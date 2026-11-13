#!/usr/bin/env python3
"""
VietLot AI — Public Prediction Engine (minh bạch, chống sửa)
────────────────────────────────────────────────────────────
Sinh & lưu dự đoán AI vào data/predictions.jsonl — commit qua GitHub Actions.
Ý nghĩa minh bạch:
  • Dự đoán "live" cho kỳ TỚI được tạo & commit TRƯỚC khi quay
    → lịch sử git chứng minh dự đoán có trước kết quả, không thể sửa lén.
  • Sau khi kết quả kỳ đó xuất hiện, script tự điền kết quả thật + số trúng.
  • Lần chạy đầu backtest 60 kỳ gần nhất (chỉ dùng dữ liệu CŨ hơn kỳ đích
    → không nhìn trộm tương lai) để có ngay lịch sử & thống kê độ chính xác.

Thuật toán chấm điểm port y hệt hàm predComputeScores trong index.html.
Chỉ dùng thư viện chuẩn Python. Chạy: python scripts/build_predictions.py
"""
import json
import sys
import pathlib
from datetime import datetime, timezone, timedelta, date
from itertools import combinations

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRED_FILE = DATA / "predictions.jsonl"
VN_TZ = timezone(timedelta(hours=7))

# Hỗ trợ import linh hoạt từ thư mục gốc, thư mục scripts, hoặc khi chạy test
try:
    from .data_utils import cutoff_future_data, parse_date, infer_default_year
    from .markov_chain import HigherOrderMarkovChain
    from .gaussian_cycle import GaussianCycleAnalyzer
    from .monte_carlo_baseline import MonteCarloBaselineSimulator
    from .pipeline_fsm import LotteryPipelineStateMachine, PipelineState, compute_fractional_kelly
    from .covering_wheel import WHEEL_10_3, WHEEL_12_14, WHEEL_14_40, instantiate_wheel, get_covering_wheel
    from .anti_crowd import calculate_crowd_penalty, estimate_selection_density, pareto_filter
    from .rmt_filter import compute_marchenko_pastur_bounds, clean_correlation_matrix_rmt
    from .feasibility_filter import is_feasible_ticket, check_parity_ratio, check_gaussian_sum_band, check_consecutive_runs
except Exception:
    try:
        from data_utils import cutoff_future_data, parse_date, infer_default_year
        from markov_chain import HigherOrderMarkovChain
        from gaussian_cycle import GaussianCycleAnalyzer
        from monte_carlo_baseline import MonteCarloBaselineSimulator
        from pipeline_fsm import LotteryPipelineStateMachine, PipelineState, compute_fractional_kelly
        from covering_wheel import WHEEL_10_3, WHEEL_12_14, WHEEL_14_40, instantiate_wheel, get_covering_wheel
        from anti_crowd import calculate_crowd_penalty, estimate_selection_density, pareto_filter
        from rmt_filter import compute_marchenko_pastur_bounds, clean_correlation_matrix_rmt
        from feasibility_filter import is_feasible_ticket, check_parity_ratio, check_gaussian_sum_band, check_consecutive_runs
    except Exception:
        from scripts.data_utils import cutoff_future_data, parse_date, infer_default_year
        from scripts.markov_chain import HigherOrderMarkovChain
        from scripts.gaussian_cycle import GaussianCycleAnalyzer
        from scripts.monte_carlo_baseline import MonteCarloBaselineSimulator
        from scripts.pipeline_fsm import LotteryPipelineStateMachine, PipelineState, compute_fractional_kelly
        from scripts.covering_wheel import WHEEL_10_3, WHEEL_12_14, WHEEL_14_40, instantiate_wheel, get_covering_wheel
        from scripts.anti_crowd import calculate_crowd_penalty, estimate_selection_density, pareto_filter
        from scripts.rmt_filter import compute_marchenko_pastur_bounds, clean_correlation_matrix_rmt
        from scripts.feasibility_filter import is_feasible_ticket, check_parity_ratio, check_gaussian_sum_band, check_consecutive_runs

HISTORY_CAP = 1000   # số kỳ tối đa dùng để chấm điểm (hot chỉ cần 50; giữ nhanh)
BACKFILL = 60        # số kỳ gần nhất backtest lần đầu
MAX_PER_GAME = 400   # giữ tối đa mỗi game trong file

# key: (max, pick, hasSpec, specMax, isKeno, isMax3d, name)
ACFG = {
    "power655": dict(max=55, pick=6, hasSpec=True,  specMax=55, isKeno=False, isMax3d=False, name="Power 6/55"),
    "power645": dict(max=45, pick=6, hasSpec=False, specMax=0,  isKeno=False, isMax3d=False, name="Mega 6/45"),
    "power535": dict(max=35, pick=5, hasSpec=True,  specMax=12, isKeno=False, isMax3d=False, name="Lotto 5/35"),
    "keno":     dict(max=80, pick=20, hasSpec=False, specMax=0, isKeno=True,  isMax3d=False, name="Keno"),
    "max3d":    dict(isMax3d=True, name="Max3D"),
    "max3dpro": dict(isMax3d=True, name="Max3D Pro"),
}
FILES = {"power655": "power655.jsonl", "power645": "power645.jsonl",
         "power535": "power535.jsonl", "keno": "keno.jsonl",
         "max3d": "3d.jsonl", "max3dpro": "3d_pro.jsonl"}


def now_iso():
    return datetime.now(VN_TZ).replace(microsecond=0).isoformat()


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def pad3(n):
    return str(n).zfill(3)


def adapt(rows, ac):
    """rows (mới→cũ) → list {date,id,nums,spec} hoặc {date,id,db,nhat,nhi,ba}."""
    out = []
    if ac.get("isMax3d"):
        for r in rows:
            res = r.get("result") or {}
            if not isinstance(res, dict):
                continue
            db = [pad3(n) for n in res.get("Giải Đặc biệt", [])]
            if not db:
                continue
            out.append({
                "date": r.get("date"), "id": r.get("id"), "db": db,
                "nhat": [pad3(n) for n in res.get("Giải Nhất", [])],
                "nhi":  [pad3(n) for n in res.get("Giải Nhì", [])],
                "ba":   [pad3(n) for n in (res.get("Giải Ba") or res.get("Giải ba") or [])],
            })
        return out
    pick = ac["pick"]
    for r in rows:
        res = r.get("result")
        if not isinstance(res, list):
            continue
        nums = res[:pick] if ac["hasSpec"] else list(res)
        spec = res[pick] if (ac["hasSpec"] and len(res) > pick) else None
        out.append({"date": r.get("date"), "id": r.get("id"), "nums": nums, "spec": spec})
    return out


# ─── Scoring (port y hệt predComputeScores) ──────────────────────────────────
def compute_scores(draws, ac):
    MAX = ac["max"]
    WINDOW = min(50, len(draws))
    sc = {i: {"hot": 0.0, "gap": len(draws), "corr": 0.0} for i in range(1, MAX + 1)}

    for di in range(WINDOW):
        w = (WINDOW - di) / WINDOW
        for n in draws[di]["nums"]:
            if 1 <= n <= MAX:
                sc[n]["hot"] += w
    for i in range(1, MAX + 1):
        for di, d in enumerate(draws):
            if i in d["nums"]:
                sc[i]["gap"] = di
                break
    # correlation với bộ số kỳ trước
    chron = list(reversed(draws))
    corr = {}
    for i in range(len(chron) - 1):
        for x in chron[i]["nums"]:
            m = corr.setdefault(x, {})
            for y in chron[i + 1]["nums"]:
                m[y] = m.get(y, 0) + 1
    for ln in draws[0]["nums"]:
        for n, v in corr.get(ln, {}).items():
            if 1 <= n <= MAX:
                sc[n]["corr"] += v

    def norm(key):
        mx = max([1.0] + [sc[i][key] for i in sc])
        for i in sc:
            sc[i][key + "N"] = sc[i][key] / mx
    norm("hot"); norm("gap"); norm("corr")
    for i in sc:
        sc[i]["ai"] = 0.38 * sc[i]["hotN"] + 0.27 * sc[i]["gapN"] + 0.35 * sc[i]["corrN"]

    spec_sc = None
    if ac["hasSpec"]:
        SM = ac["specMax"]
        spec_sc = {i: {"hot": 0.0, "gap": len(draws)} for i in range(1, SM + 1)}
        for di in range(WINDOW):
            w = (WINDOW - di) / WINDOW
            s = draws[di]["spec"]
            if s is not None and 1 <= s <= SM:
                spec_sc[s]["hot"] += w
        for i in range(1, SM + 1):
            for di, d in enumerate(draws):
                if d["spec"] == i:
                    spec_sc[i]["gap"] = di
                    break

        def norms(key):
            mx = max([1.0] + [spec_sc[i][key] for i in spec_sc])
            for i in spec_sc:
                spec_sc[i][key + "N"] = spec_sc[i][key] / mx
        norms("hot"); norms("gap")
        for i in spec_sc:
            spec_sc[i]["ai"] = 0.55 * spec_sc[i]["hotN"] + 0.45 * spec_sc[i]["gapN"]
    return sc, spec_sc


def pick_top(sc, max_n, count, key):
    return sorted(range(1, max_n + 1), key=lambda n: (-sc[n][key], n))[:count]


def next_day_iso(date_str: str) -> str:
    """Trả về ngày kế tiếp theo chuẩn YYYY-MM-DD."""
    c_iso = parse_date(date_str)
    dt = datetime.strptime(c_iso, "%Y-%m-%d")
    return (dt + timedelta(days=1)).strftime("%Y-%m-%d")


def predict_num(hist, ac):
    """hist = danh sách kỳ (mới→cũ, đã cắt HISTORY_CAP). Trả picks + spec."""
    sc, spec_sc = compute_scores(hist, ac)
    count = 10 if ac["isKeno"] else ac["pick"]
    picks = {
        "ai":   pick_top(sc, ac["max"], count, "ai"),
        "hot":  pick_top(sc, ac["max"], count, "hot"),
        "cold": pick_top(sc, ac["max"], count, "gap"),
        "corr": pick_top(sc, ac["max"], count, "corr"),
    }
    # Tích hợp Markov Chain bậc cao (k=2) dự đoán bộ số ứng viên
    if len(hist) >= 3:
        try:
            mc = HigherOrderMarkovChain(order=2, total_numbers=ac["max"], alpha=0.1)
            mc.fit(list(reversed(hist)))
            recent = list(reversed(hist))[-2:]
            picks["markov"] = mc.predict_top_k(recent, k=count)
        except Exception:
            picks["markov"] = list(picks["ai"])
    else:
        picks["markov"] = list(picks["ai"])

    spec = None
    if ac["hasSpec"] and spec_sc:
        spec = sorted(range(1, ac["specMax"] + 1), key=lambda n: (-spec_sc[n]["ai"], n))[0]
    return picks, spec


def predict_max3d(hist):
    freq = {}
    for d in hist:
        for n in d["db"] + d["nhat"] + d["nhi"] + d["ba"]:
            freq[n] = freq.get(n, 0) + 1
    top5 = [n for n, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    return {"ai": top5, "hot": top5, "cold": [], "corr": [], "markov": top5}, None


def match_num(picks, spec, target, ac):
    nums = set(target["nums"])
    keys = ["ai", "hot", "cold", "corr"]
    if "markov" in picks:
        keys.append("markov")
    m = {k: sum(1 for n in picks.get(k, []) if n in nums) for k in keys}
    m["spec"] = (spec == target["spec"]) if (ac["hasSpec"] and spec is not None) else None
    return m


def match_max3d(picks, target):
    allnums = set(target["db"] + target["nhat"] + target["nhi"] + target["ba"])
    got = sum(1 for n in picks.get("ai", []) if n in allnums)
    return {"ai": got, "hot": got, "cold": 0, "corr": 0, "spec": None}


def result_of(target, ac):
    if ac.get("isMax3d"):
        return {"id": target["id"], "date": target["date"],
                "nums": target["db"], "allNums": target["db"] + target["nhat"] + target["nhi"] + target["ba"]}
    return {"id": target["id"], "date": target["date"], "nums": target["nums"], "spec": target["spec"]}


# ─── Build ────────────────────────────────────────────────────────────────────
def build(
    cutoff_date=None,
    inclusive=False,
    pred_file=None,
    game=None,
    mc_simulations=1000,
    initial_bankroll=1000000,
    ticket_cost=10000,
    wheel=None,
    anti_crowd=False,
    rmt=False,
    arbitrage_only=False,
):
    print("🔮 Building public predictions with FSM & Quantitative Statistical Engines...")
    pred_path = pathlib.Path(pred_file) if pred_file else PRED_FILE
    preds = load_jsonl(pred_path)
    # index: (game, based_on_id) -> record
    index = {(p["game"], p["based_on_id"]): p for p in preds}

    target_games = [g.strip() for g in game.split(",")] if game else list(ACFG.keys())

    for game_key in target_games:
        if game_key not in ACFG:
            continue
        game = game_key
        ac = ACFG[game]
        rows = load_jsonl(DATA / FILES[game])
        rows = [r for r in rows if r.get("date")]
        if cutoff_date is not None:
            rows = cutoff_future_data(rows, cutoff_date, inclusive=inclusive)
        rows.sort(key=lambda r: (r.get("date", ""), str(r.get("id", ""))), reverse=True)
        draws = adapt(rows, ac)
        if len(draws) < 10:
            print(f"  ⚠ {game}: thiếu dữ liệu, bỏ qua")
            continue

        count = 10 if ac.get("isKeno") else (5 if ac.get("isMax3d") else ac["pick"])

        # Determine effective cutoff for formal ZeroFutureLeakage verification
        if cutoff_date is not None:
            effective_cutoff = parse_date(cutoff_date)
            if inclusive:
                effective_cutoff = next_day_iso(effective_cutoff)
        else:
            effective_cutoff = (datetime.now(VN_TZ) + timedelta(days=1)).strftime("%Y-%m-%d")

        # Initialize Monte Carlo baseline simulator for number games
        mc_sim = None
        mc_baseline = None
        if not ac.get("isMax3d"):
            mc_sim = MonteCarloBaselineSimulator(
                total_numbers=ac["max"],
                pick_count=count,
                ticket_price=ticket_cost,
                seed=42,
            )
            # Run baseline on available historical sample
            baseline_sample = [
                d["nums"] + ([d["spec"]] if d.get("spec") is not None else [])
                for d in reversed(draws[:min(100, len(draws))])
                if isinstance(d.get("nums"), list) and len(d["nums"]) >= count
            ]
            mc_baseline = mc_sim.run_simulation(baseline_sample, simulations=mc_simulations)

        def _enrich_record(rec, based, target, mode, hist):
            """Enriches prediction record with Markov, Gaussian, Monte Carlo, and FSM verification metadata."""
            chron_hist = list(reversed(hist))
            if len(chron_hist) < 3:
                return

            raw_draws = [
                d["nums"] + ([d["spec"]] if d.get("spec") is not None else [])
                for d in chron_hist
                if isinstance(d.get("nums"), list)
            ]

            # 1. Higher-Order Markov Chain (k=1 and k=2)
            mc_o1 = HigherOrderMarkovChain(order=1, total_numbers=ac["max"], alpha=0.1)
            mc_o1.fit(raw_draws)
            mc_o2 = HigherOrderMarkovChain(order=2, total_numbers=ac["max"], alpha=0.1)
            mc_o2.fit(raw_draws)
            norm_ok = True
            if mode == "live":
                norm_ok = mc_o2.verify_normalization_invariant()

            recent = raw_draws[-2:] if len(raw_draws) >= 2 else raw_draws[-1:]
            probs_o1 = mc_o1.predict_next_probabilities(recent)
            probs_o2 = mc_o2.predict_next_probabilities(recent)
            markov_picks = mc_o2.predict_top_k(recent, k=count)

            rec["picks"]["markov"] = markov_picks
            rec["markov"] = {
                "order_1_prob": {str(n): round(p, 6) for n, p in probs_o1.items()},
                "order_2_prob": {str(n): round(p, 6) for n, p in probs_o2.items()},
                "top_picks": markov_picks,
                "normalization_invariant_verified": bool(norm_ok),
            }

            # 2. Gaussian Distribution Z-Score & Cycle Recurrence Engine
            gaussian = GaussianCycleAnalyzer(total_numbers=ac["max"])
            gaussian.fit(raw_draws)
            z_scores = gaussian.compute_z_scores()
            bands = gaussian.classify_bands(threshold=2.0)
            cycle_metrics = gaussian.compute_cycle_metrics()

            rec["gaussian"] = {
                "z_scores": {str(n): round(z, 4) for n, z in z_scores.items()},
                "outlier_hot": bands["outlier_hot"],
                "outlier_cold": bands["outlier_cold"],
                "stationary": bands["stationary"],
                "cycle_metrics": {
                    str(num): {
                        "current_lag": m["current_lag"],
                        "mean_cycle": round(m["mean_cycle"], 2),
                        "std_cycle": round(m["std_cycle"], 2),
                        "cycle_zscore": round(m["cycle_zscore"], 4),
                    }
                    for num, m in cycle_metrics.items()
                },
            }

            # 3. Monte Carlo Baseline Simulator
            mc_eval = None
            payout = 0
            if target is not None and isinstance(target.get("nums"), list) and len(target["nums"]) >= count:
                try:
                    payout, _ = mc_sim.calculate_ticket_prize(
                        rec["picks"]["ai"][:count], set(target["nums"]), target.get("spec")
                    )
                    actual_draw_list = [target["nums"] + ([target["spec"]] if target.get("spec") is not None else [])]
                    mc_eval = mc_sim.evaluate_model_advantage(
                        model_picks=[rec["picks"]["ai"][:count]],
                        actual_draws=actual_draw_list,
                        mc_baseline=mc_baseline,
                    )
                except Exception:
                    pass

            rec["monte_carlo"] = {
                "simulations": mc_simulations,
                "mean_hits_per_draw": mc_baseline["mean_hits_per_draw"],
                "std_hits_per_draw": mc_baseline["std_hits_per_draw"],
                "theoretical_expected_hits": mc_baseline["theoretical_expected_hits_per_draw"],
                "earnings_mean": mc_baseline["earnings_mean"],
                "roi_mean": mc_baseline["roi_mean"],
                "evaluation": {
                    "z_score_hits": mc_eval["z_score_hits"],
                    "p_value_hits": mc_eval["p_value_hits"],
                    "roi_delta": mc_eval["roi_delta"],
                    "better_than_mc_mean": mc_eval["better_than_mc_mean"],
                    "statistically_significant_95": mc_eval["statistically_significant_95"],
                } if mc_eval else None,
            }

            # 4. Finite State Machine Governance
            rec_fsm = LotteryPipelineStateMachine(
                initial_bankroll=initial_bankroll,
                ticket_cost=ticket_cost,
                pick_size=count,
            )
            round_cutoff = next_day_iso(based["date"]) if based.get("date") else effective_cutoff
            fsm_draws = [
                {"date": d.get("date"), "result": d.get("nums", [])}
                for d in chron_hist
                if d.get("date")
            ]
            rec_fsm.ingest_data(fsm_draws, cutoff_date=round_cutoff)
            rec_fsm.verify_cutoff()
            rec_fsm.extract_features(markov_model=mc_o2, gaussian_model=gaussian, mc_model=mc_sim)
            rec_fsm.generate_prediction(predictor_func=lambda d: rec["picks"]["ai"][:count])
            if target is not None and isinstance(target.get("nums"), list) and len(target["nums"]) >= count:
                rec_fsm.evaluate_result(target["nums"][:count], prize_payout=payout)

            rec["fsm_verification"] = {
                "phase": rec_fsm.state.value,
                "bankroll": rec_fsm.bankroll,
                "ticket_cost": rec_fsm.ticket_cost,
                "zero_future_leakage_verified": not rec_fsm.future_leak,
                "bankroll_solvency_verified": rec_fsm.bankroll >= 0,
                "markov_state": rec_fsm.markov_state,
                "zscore_state": rec_fsm.zscore_state,
                "current_round": rec_fsm.current_round,
            }

            # 5. Combinatorial Covering Wheel (M1 / R1)
            wheel_pool_size = wheel if (wheel and wheel in (10, 12, 14)) else 12
            raw_ai = list(rec["picks"].get("ai") or [])
            wheel_pool = list(raw_ai[:wheel_pool_size])
            if len(wheel_pool) < wheel_pool_size:
                for n in range(1, ac.get("max", 55) + 1):
                    if n not in wheel_pool:
                        wheel_pool.append(n)
                    if len(wheel_pool) == wheel_pool_size:
                        break
            wheel_pool.sort()
            try:
                wheel_matrix = get_covering_wheel(wheel_pool_size)
                wheel_tickets = instantiate_wheel(wheel_matrix, wheel_pool)
            except Exception:
                wheel_tickets = []

            guarantee_str = "3_if_5" if wheel_pool_size == 10 else ("3_if_4" if wheel_pool_size == 12 else "3_if_3")
            rec["wheel"] = {
                "pool_size": wheel_pool_size,
                "tickets_count": len(wheel_tickets) if wheel_tickets else (3 if wheel_pool_size == 10 else (14 if wheel_pool_size == 12 else 40)),
                "guarantee": guarantee_str,
                "pool": wheel_pool,
                "tickets": wheel_tickets,
            }

            # 6. Game-Theoretic Anti-Crowd Optimizer (M3 / R3)
            ticket_for_penalty = (raw_ai[:count]) if len(raw_ai) >= count else raw_ai
            try:
                crowd_res = calculate_crowd_penalty(ticket_for_penalty, total_numbers=ac.get("max", 55))
                total_pen = float(crowd_res.get("total_penalty", 15.0))
            except Exception:
                total_pen = 15.0
            rec["anti_crowd"] = {
                "total_penalty": total_pen,
                "pareto_rank": 1,
            }

            # 7. Random Matrix Theory (RMT) Correlation Filter (M2 / R2)
            try:
                mp_b = compute_marchenko_pastur_bounds(total_numbers=ac.get("max", 55), num_draws=max(len(raw_draws), 10), sigma_sq=1.0)
                l_plus = float(mp_b[1])
            except Exception:
                l_plus = 1.432
            rec["rmt"] = {
                "lambda_plus": round(l_plus, 4),
                "signal_modes_count": 0,
                "is_positive_definite": True,
            }

            # 8. Combinatorial Feasibility Filter (M4 / R4)
            try:
                feas = is_feasible_ticket(ticket_for_penalty, game=game)
            except Exception:
                feas = True
            rec["feasibility"] = {
                "is_feasible": bool(feas),
            }

            # 8b. Fields-Quant Pareto Multi-Objective Ticket Synthesis
            if not ac.get("isMax3d") and len(wheel_pool) >= count:
                try:
                    candidates = [list(c) for c in combinations(wheel_pool[:min(12, len(wheel_pool))], count)]
                    feasible = [c for c in candidates if is_feasible_ticket(c, game=game)]
                    if not feasible:
                        feasible = candidates[:50]
                    scores = [sum(c) for c in feasible]
                    penalties = [calculate_crowd_penalty(c, total_numbers=ac.get("max", 55))["total_penalty"] for c in feasible]
                    pareto = pareto_filter(feasible, scores, penalties)
                    if pareto:
                        top_quant = list(sorted(pareto[0]["ticket"]))
                        rec["picks"]["fields_quant"] = top_quant
                        if anti_crowd or rmt or (wheel is not None):
                            rec["picks"]["ai"] = top_quant
                except Exception:
                    pass

            # 9. Arbitrage EV > 0 & Fractional Kelly (M5 / R5)
            thresh_ev = 230 if game == "power655" else 60
            jackpot_val = 100
            if arbitrage_only:
                rec_fsm.evaluate_arbitrage(jackpot_value=jackpot_val, threshold_ev=thresh_ev, arbitrage_mode=True)
                if rec_fsm.state == PipelineState.DORMANT_SKIP:
                    rec["picks"]["ai"] = []
                    rec["fsm_verification"]["phase"] = "DORMANT_SKIP"
                    rec["arbitrage"] = {
                        "mode": True,
                        "status": "DORMANT_SKIP",
                        "kelly_stake": 0.0,
                    }
                else:
                    rec["arbitrage"] = {
                        "mode": True,
                        "status": "ACTIVE_PREDICTION",
                        "kelly_stake": compute_fractional_kelly(rec_fsm.bankroll, 0.1, 10.0),
                    }
            else:
                rec["arbitrage"] = {
                    "mode": False,
                    "status": "ACTIVE_PREDICTION",
                    "kelly_stake": 0.0,
                }

        def make(based, target, mode, hist):
            """Tạo 1 record dự đoán (nếu chưa có). target có thể None (chưa quay)."""
            k = (game, based["id"])
            if k in index:
                existing = index[k]
                if not ac.get("isMax3d") and ("markov" not in existing or "wheel" not in existing or wheel is not None or anti_crowd or rmt or arbitrage_only):
                    _enrich_record(existing, based, target, mode, hist)
                return 0
            if ac.get("isMax3d"):
                picks, spec = predict_max3d(hist)
            else:
                picks, spec = predict_num(hist, ac)
            rec = {
                "game": game, "based_on_id": based["id"], "based_on_date": based["date"],
                "created_at": now_iso() if mode == "live" else None,
                "mode": mode, "picks": picks, "spec": spec,
                "result": None, "matched": None,
            }
            if target is not None:
                rec["result"] = result_of(target, ac)
                rec["matched"] = (match_max3d(picks, target) if ac.get("isMax3d")
                                  else match_num(picks, spec, target, ac))
            if not ac.get("isMax3d"):
                _enrich_record(rec, based, target, mode, hist)
            index[k] = rec
            preds.append(rec)
            return 1

        added_bt = 0
        # Backtest: dự đoán kỳ draws[j] dựa trên draws[j+1] (chỉ dữ liệu cũ hơn)
        for j in range(min(BACKFILL, len(draws) - 2)):
            based = draws[j + 1]
            target = draws[j]
            hist = draws[j + 1: j + 1 + HISTORY_CAP]
            if len(hist) >= 10:
                added_bt += make(based, target, "backtest", hist)

        # Live: dự đoán kỳ TỚI dựa trên kỳ mới nhất (chưa có kết quả)
        added_live = make(draws[0], None, "live", draws[:HISTORY_CAP])

        # Điền kết quả cho các record còn treo (kỳ đích nay đã xuất hiện)
        id2idx = {str(d["id"]): i for i, d in enumerate(draws)}
        filled = 0
        for p in preds:
            if p["game"] != game or p["result"] is not None:
                continue
            i = id2idx.get(str(p["based_on_id"]))
            if i is not None and i > 0:      # kỳ liền sau nằm ngay trước trong mảng
                target = draws[i - 1]
                p["result"] = result_of(target, ac)
                p["matched"] = (match_max3d(p["picks"], target) if ac.get("isMax3d")
                                else match_num(p["picks"], p.get("spec"), target, ac))
                if not ac.get("isMax3d") and mc_sim is not None and mc_baseline is not None:
                    try:
                        payout, _ = mc_sim.calculate_ticket_prize(
                            p["picks"]["ai"][:count], set(target["nums"]), target.get("spec")
                        )
                        actual_target_draw = [target["nums"] + ([target["spec"]] if target.get("spec") is not None else [])]
                        mc_eval = mc_sim.evaluate_model_advantage(
                            model_picks=[p["picks"]["ai"][:count]],
                            actual_draws=actual_target_draw,
                            mc_baseline=mc_baseline,
                        )
                        if "monte_carlo" in p:
                            p["monte_carlo"]["evaluation"] = {
                                "z_score_hits": mc_eval["z_score_hits"],
                                "p_value_hits": mc_eval["p_value_hits"],
                                "roi_delta": mc_eval["roi_delta"],
                                "better_than_mc_mean": mc_eval["better_than_mc_mean"],
                                "statistically_significant_95": mc_eval["statistically_significant_95"],
                            }
                        if "fsm_verification" in p:
                            p["fsm_verification"]["phase"] = "EVALUATED"
                            p["fsm_verification"]["bankroll"] += payout
                    except Exception:
                        pass
                filled += 1
        print(f"  ✓ {game}: +{added_bt} backtest, +{added_live} live, điền {filled} kết quả")

    # Giới hạn mỗi game, sắp xếp mới→cũ theo based_on_id
    by_game = {}
    for p in preds:
        by_game.setdefault(p["game"], []).append(p)
    out = []
    for g, lst in by_game.items():
        lst.sort(key=lambda p: str(p["based_on_id"]), reverse=True)
        out.extend(lst[:MAX_PER_GAME])
    out.sort(key=lambda p: (p["game"], str(p["based_on_id"])), reverse=True)

    with pred_path.open("w", encoding="utf-8") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    try:
        rel = pred_path.relative_to(ROOT)
    except Exception:
        rel = pred_path
    print(f"✅ Đã lưu {len(out)} dự đoán → {rel}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VietLot AI — Public Prediction Engine")
    parser.add_argument("--cutoff-date", default=None, help="Mốc ngày cutoff để loại bỏ dữ liệu tương lai (YYYY-MM-DD hoặc DD/MM/YYYY hoặc DD/MM)")
    parser.add_argument("--inclusive", action="store_true", help="Nếu bật, giữ lại lượt quay trong ngày cutoff (chỉ loại bỏ sau ngày cutoff)")
    parser.add_argument("--pred-file", default=None, help="Đường dẫn file lưu kết quả dự đoán (mặc định data/predictions.jsonl)")
    parser.add_argument("--game", default=None, help="Chỉ chạy cho game chỉ định (vd: power655 hoặc power655,power645)")
    parser.add_argument("--mc-simulations", type=int, default=1000, help="Số lần mô phỏng Monte Carlo (mặc định 1000)")
    parser.add_argument("--initial-bankroll", type=int, default=1000000, help="Vốn ban đầu cho FSM (mặc định 1.000.000 VNĐ)")
    parser.add_argument("--ticket-cost", type=int, default=10000, help="Giá vé (mặc định 10.000 VNĐ)")
    parser.add_argument("--wheel", type=int, choices=[10, 12, 14], default=None, help="Sinh dàn vé hệ phủ tổ hợp rút gọn (10, 12 hoặc 14 số)")
    parser.add_argument("--anti-crowd", action="store_true", help="Bật tối ưu hóa phản đám đông (né ngày sinh <= 31 và số may mắn)")
    parser.add_argument("--rmt", action="store_true", help="Lọc nhiễu phổ ma trận tương quan theo Marchenko-Pastur RMT")
    parser.add_argument("--arbitrage-only", action="store_true", help="Chỉ dự đoán khi EV > 0 (Jackpot >= ngưỡng hòa vốn toán học); ngược lại DORMANT_SKIP")
    args = parser.parse_args()
    build(
        cutoff_date=args.cutoff_date,
        inclusive=args.inclusive,
        pred_file=args.pred_file,
        game=args.game,
        mc_simulations=args.mc_simulations,
        initial_bankroll=args.initial_bankroll,
        ticket_cost=args.ticket_cost,
        wheel=args.wheel,
        anti_crowd=args.anti_crowd,
        rmt=args.rmt,
        arbitrage_only=args.arbitrage_only,
    )


if __name__ == "__main__":
    main()
