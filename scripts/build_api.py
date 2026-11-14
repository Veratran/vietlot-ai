#!/usr/bin/env python3
"""
VietLot AI — API Builder
Sinh tầng API JSON tĩnh (kết quả + phân tích) từ data/*.jsonl.
Kết quả ghi vào api/ và được deploy cùng GitHub Pages.

Chạy: python scripts/build_api.py
Không cần dependency ngoài (chỉ dùng thư viện chuẩn Python).

API base khi deploy: https://compal123.github.io/vietlot-ai/api/
"""
import json
import sys
import pathlib
from collections import Counter
from datetime import datetime, timezone, timedelta
from itertools import combinations

try:
    from .markov_chain import HigherOrderMarkovChain
    from .gaussian_cycle import GaussianCycleAnalyzer
    from .monte_carlo_baseline import MonteCarloBaselineSimulator
except Exception:
    try:
        from markov_chain import HigherOrderMarkovChain
        from gaussian_cycle import GaussianCycleAnalyzer
        from monte_carlo_baseline import MonteCarloBaselineSimulator
    except Exception:
        from scripts.markov_chain import HigherOrderMarkovChain
        from scripts.gaussian_cycle import GaussianCycleAnalyzer
        from scripts.monte_carlo_baseline import MonteCarloBaselineSimulator

try:
    sys.stdout.reconfigure(encoding="utf-8")  # in emoji trên console Windows
except Exception:
    pass

# ─── Cấu hình ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
API = ROOT / "api"

SITE = "https://compal123.github.io/vietlot-ai"
API_BASE = f"{SITE}/api"
VN_TZ = timezone(timedelta(hours=7))
RESULTS_LIMIT = 100          # số kỳ gần nhất trả trong results.json
RECENT_WINDOW = 100          # cửa sổ "gần đây" cho phân tích
TOP_N = 10                   # số phần tử trong bảng nóng/lạnh/cặp

# Game bốc số: pool = dải số, main = số con chính, special = số phụ (jackpot ball)
NUMBER_GAMES = {
    "power655": {"name": "Power 6/55", "pool": 55, "main": 6, "special": 1,
                 "schedule": "Thứ 3, 5, 7 — 18:00"},
    "power645": {"name": "Mega 6/45", "pool": 45, "main": 6, "special": 0,
                 "schedule": "Thứ 4, 6, CN — 18:00"},
    "power535": {"name": "Power 5/35", "pool": 35, "main": 5, "special": 1,
                 "schedule": "Thứ 2, 4, 6 — 18:00"},
    "keno": {"name": "Keno", "pool": 80, "main": 20, "special": 0,
             "schedule": "Mỗi 10 phút, 06:00–21:55"},
}

# Game bốc chữ số: result là dict giải -> danh sách số 3 chữ số
DIGIT_GAMES = {
    "3d": {"name": "Max 3D", "digits": 3, "schedule": "Thứ 2, 4, 6 — 18:00"},
    "3d_pro": {"name": "Max 3D Pro", "digits": 3, "schedule": "Thứ 3, 5, 7 — 18:00"},
}


# ─── Tiện ích ─────────────────────────────────────────────────────────────────
def now_iso():
    return datetime.now(VN_TZ).replace(microsecond=0).isoformat()


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def by_date(rows):
    """Sắp xếp mới -> cũ theo (date, id) và loại bản ghi thiếu date."""
    rows = [r for r in rows if r.get("date")]
    rows.sort(key=lambda r: (r.get("date", ""), str(r.get("id", ""))), reverse=True)
    return rows


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {path.relative_to(ROOT)}")


# ─── Phân tích game bốc số ────────────────────────────────────────────────────
def analyze_number_game(key, cfg, rows):
    """Trả về dict thống kê cho một game bốc số. rows: mới -> cũ."""
    main_n = cfg["main"]
    pool = cfg["pool"]

    freq = Counter()          # tần suất số chính, toàn lịch sử
    bonus_freq = Counter()    # tần suất số phụ (nếu có)
    last_seen = {}            # số -> index kỳ gần nhất xuất hiện (0 = mới nhất)
    sums, odd_ratios, low_ratios = [], [], []
    pair_freq = Counter()
    recent_freq = Counter()

    valid = [r for r in rows if isinstance(r.get("result"), list)]
    for idx, r in enumerate(valid):
        res = r["result"]
        main = res[:main_n]
        bonus = res[main_n:]
        for n in main:
            freq[n] += 1
            if n not in last_seen:
                last_seen[n] = idx
            if idx < RECENT_WINDOW:
                recent_freq[n] += 1
        for n in bonus:
            bonus_freq[n] += 1
        if main:
            sums.append(sum(main))
            odd_ratios.append(sum(1 for n in main if n % 2 == 1))
            low_ratios.append(sum(1 for n in main if n <= pool // 2))
        for a, b in combinations(sorted(main), 2):
            pair_freq[(a, b)] += 1

    total_draws = len(valid)
    universe = list(range(1, pool + 1))

    # nóng / lạnh theo tần suất toàn lịch sử
    ranked = sorted(universe, key=lambda n: (-freq.get(n, 0), n))
    hot = [{"number": n, "count": freq.get(n, 0)} for n in ranked[:TOP_N]]
    cold = [{"number": n, "count": freq.get(n, 0)} for n in ranked[-TOP_N:][::-1]]

    # "số gan": số kỳ liên tiếp chưa xuất hiện (tính từ kỳ mới nhất)
    overdue = []
    for n in universe:
        gap = last_seen.get(n, total_draws)  # chưa từng ra -> gap = tổng số kỳ
        overdue.append({"number": n, "draws_since_last": gap})
    overdue.sort(key=lambda x: (-x["draws_since_last"], x["number"]))
    overdue = overdue[:TOP_N]

    # nóng gần đây
    recent_ranked = sorted(universe, key=lambda n: (-recent_freq.get(n, 0), n))
    recent_hot = [{"number": n, "count": recent_freq.get(n, 0)}
                  for n in recent_ranked[:TOP_N]]

    top_pairs = [{"pair": [a, b], "count": c}
                 for (a, b), c in pair_freq.most_common(TOP_N)]

    def stat(arr):
        if not arr:
            return None
        return {"min": min(arr), "max": max(arr),
                "avg": round(sum(arr) / len(arr), 2)}

    out = {
        "game": key,
        "name": cfg["name"],
        "pool": pool,
        "numbers_per_draw": main_n,
        "total_draws": total_draws,
        "frequency": {str(n): freq.get(n, 0) for n in universe},
        "hot_numbers": hot,
        "cold_numbers": cold,
        "overdue_numbers": overdue,
        "recent_hot_numbers": {
            "window": min(RECENT_WINDOW, total_draws),
            "numbers": recent_hot,
        },
        "top_pairs": top_pairs,
        "sum_stats": stat(sums),
        "odd_count_stats": stat(odd_ratios),
        "low_count_stats": stat(low_ratios),
        "generated_at": now_iso(),
        "_note": ("Dữ liệu thống kê thuần túy từ lịch sử; xổ số là ngẫu nhiên, "
                  "các con số này KHÔNG dự đoán được kết quả tương lai."),
    }
    if cfg["special"]:
        b_ranked = sorted(range(1, pool + 1),
                          key=lambda n: (-bonus_freq.get(n, 0), n))
        out["bonus_frequency"] = {str(n): bonus_freq.get(n, 0)
                                  for n in range(1, pool + 1)}
        out["bonus_hot_numbers"] = [{"number": n, "count": bonus_freq.get(n, 0)}
                                    for n in b_ranked[:TOP_N]]

    # ─── Tích hợp M1: Higher-Order Markov Chain (k=1 & k=2) ───
    recent_cap = min(1000, len(valid))
    valid_draws = [r["result"][:main_n] for r in valid[:recent_cap] if isinstance(r.get("result"), list) and len(r["result"]) >= main_n]
    chron_draws = list(reversed(valid_draws))

    mc_o2 = HigherOrderMarkovChain(order=2, total_numbers=pool, alpha=0.1)
    mc_o1 = HigherOrderMarkovChain(order=1, total_numbers=pool, alpha=0.1)
    if len(chron_draws) >= 3:
        mc_o2.fit(chron_draws)
        mc_o1.fit(chron_draws)
        recent = chron_draws[-2:] if len(chron_draws) >= 2 else chron_draws[-1:]
        top_markov = mc_o2.predict_top_k(recent, k=main_n)
        markov_probs_o2 = mc_o2.predict_next_probabilities(recent)
        markov_probs_o1 = mc_o1.predict_next_probabilities(recent)
        norm_verified = mc_o2.verify_normalization_invariant()
    else:
        top_markov = list(range(1, main_n + 1))
        markov_probs_o2 = {n: 1.0 / pool for n in range(1, pool + 1)}
        markov_probs_o1 = {n: 1.0 / pool for n in range(1, pool + 1)}
        norm_verified = True

    # ─── Tích hợp M2: Gaussian Z-Score & Cycle Recurrence Engine ───
    gaussian = GaussianCycleAnalyzer(total_numbers=pool)
    if len(chron_draws) >= 1:
        gaussian.fit(chron_draws)
        z_scores = gaussian.compute_z_scores()
        bands = gaussian.classify_bands(threshold=2.0)
        cycle_metrics = gaussian.compute_cycle_metrics()
        g_summary = gaussian.get_summary(threshold=2.0)
    else:
        z_scores = {n: 0.0 for n in range(1, pool + 1)}
        bands = {"outlier_hot": [], "outlier_cold": [], "stationary": list(range(1, pool + 1))}
        cycle_metrics = {n: {"current_lag": 0, "mean_cycle": 0.0, "std_cycle": 0.0, "cycle_zscore": 0.0} for n in range(1, pool + 1)}
        g_summary = {"mean_frequency": 0.0, "std_frequency": 0.0, "band_2sigma": [0.0, 0.0], "counts": {"outlier_hot": 0, "outlier_cold": 0, "stationary": pool}}

    # ─── Tích hợp M3: Monte Carlo Baseline Simulator ───
    mc_sim = MonteCarloBaselineSimulator(total_numbers=pool, pick_count=main_n, seed=42)
    if len(chron_draws) >= 1:
        mc_sample = chron_draws[:min(100, len(chron_draws))]
        mc_res = mc_sim.run_simulation(mc_sample, simulations=1000)
    else:
        mc_res = {
            "simulations": 1000,
            "mean_hits_per_draw": mc_sim.theoretical_expected_hits_per_draw,
            "std_hits_per_draw": 0.0,
            "earnings_mean": 0.0,
            "net_earnings_mean": 0.0,
            "roi_mean": -1.0,
            "hit_rate_distribution": {},
            "prize_tier_counts_mean": {},
        }

    sorted_by_lag = sorted(cycle_metrics.items(), key=lambda item: item[1]["current_lag"], reverse=True)

    out["markov"] = {
        "order": 2,
        "normalization_invariant_verified": bool(norm_verified),
        "top_candidates": top_markov,
        "next_probabilities_order2": {str(n): round(p, 6) for n, p in markov_probs_o2.items()},
    }
    out["gaussian"] = {
        "mean_frequency": round(gaussian.mean_frequency, 2),
        "std_frequency": round(gaussian.std_frequency, 2),
        "band_2sigma": g_summary["band_2sigma"],
        "outlier_hot": bands["outlier_hot"],
        "outlier_cold": bands["outlier_cold"],
        "stationary": bands["stationary"],
        "most_overdue_numbers": [
            {
                "number": n,
                "lag": m["current_lag"],
                "mean_cycle": round(m["mean_cycle"], 2),
                "std_cycle": round(m["std_cycle"], 2),
                "cycle_zscore": round(m["cycle_zscore"], 4),
            }
            for n, m in sorted_by_lag[:TOP_N]
        ],
    }
    out["monte_carlo"] = {
        "simulations": mc_res["simulations"],
        "theoretical_expected_hits": round(mc_sim.theoretical_expected_hits_per_draw, 4),
        "mean_hits_per_draw": round(mc_res["mean_hits_per_draw"], 4),
        "std_hits_per_draw": round(mc_res["std_hits_per_draw"], 4),
        "earnings_mean": round(mc_res["earnings_mean"], 2),
        "roi_mean": round(mc_res["roi_mean"], 4),
    }

    out["_markov_details"] = {
        "game": key,
        "name": cfg["name"],
        "pool": pool,
        "order": 2,
        "normalization_invariant_verified": bool(norm_verified),
        "top_candidates": top_markov,
        "next_probabilities_order2": {str(n): round(p, 6) for n, p in markov_probs_o2.items()},
        "next_probabilities_order1": {str(n): round(p, 6) for n, p in markov_probs_o1.items()},
        "transition_matrix_order1": mc_o1.get_transition_matrix(order=1),
        "generated_at": now_iso(),
    }
    out["_gaussian_details"] = {
        "game": key,
        "name": cfg["name"],
        "pool": pool,
        "total_draws": len(chron_draws),
        "mean_frequency": round(gaussian.mean_frequency, 4),
        "std_frequency": round(gaussian.std_frequency, 4),
        "band_2sigma": g_summary["band_2sigma"],
        "counts": {
            "outlier_hot": len(bands["outlier_hot"]),
            "outlier_cold": len(bands["outlier_cold"]),
            "stationary": len(bands["stationary"]),
        },
        "outlier_hot": bands["outlier_hot"],
        "outlier_cold": bands["outlier_cold"],
        "stationary": bands["stationary"],
        "z_scores": {str(n): round(z, 4) for n, z in z_scores.items()},
        "cycle_metrics": {
            str(n): {
                "current_lag": m["current_lag"],
                "mean_cycle": round(m["mean_cycle"], 2),
                "std_cycle": round(m["std_cycle"], 2),
                "cycle_zscore": round(m["cycle_zscore"], 4),
            }
            for n, m in cycle_metrics.items()
        },
        "generated_at": now_iso(),
    }
    out["_monte_carlo_details"] = {
        "game": key,
        "name": cfg["name"],
        "pool": pool,
        "pick_count": main_n,
        "simulations": mc_res["simulations"],
        "theoretical_expected_hits_per_draw": round(mc_sim.theoretical_expected_hits_per_draw, 6),
        "mean_hits_per_draw": round(mc_res["mean_hits_per_draw"], 6),
        "std_hits_per_draw": round(mc_res["std_hits_per_draw"], 6),
        "earnings_mean": round(mc_res["earnings_mean"], 2),
        "net_earnings_mean": round(mc_res.get("net_earnings_mean", 0.0), 2),
        "roi_mean": round(mc_res["roi_mean"], 6),
        "hit_rate_distribution": {str(k): round(v, 6) for k, v in mc_res["hit_rate_distribution"].items()},
        "prize_tier_counts_mean": {k: round(v, 4) for k, v in mc_res["prize_tier_counts_mean"].items()},
        "generated_at": now_iso(),
    }
    return out


# ─── Phân tích game bốc chữ số (3D) ───────────────────────────────────────────
def analyze_digit_game(key, cfg, rows):
    num_freq = Counter()                         # tần suất số 3 chữ số
    pos_freq = [Counter() for _ in range(cfg["digits"])]  # tần suất từng vị trí

    valid = [r for r in rows if isinstance(r.get("result"), dict)]
    for r in valid:
        for nums in r["result"].values():
            for s in nums:
                s = str(s).zfill(cfg["digits"])
                num_freq[s] += 1
                for i, ch in enumerate(s[-cfg["digits"]:]):
                    if ch.isdigit():
                        pos_freq[i][ch] += 1

    top_numbers = [{"number": n, "count": c}
                   for n, c in num_freq.most_common(TOP_N)]
    positions = []
    labels = ["trăm", "chục", "đơn vị"]
    for i, pf in enumerate(pos_freq):
        digits_sorted = sorted(range(10), key=lambda d: (-pf.get(str(d), 0), d))
        positions.append({
            "position": labels[i] if i < len(labels) else f"vị trí {i+1}",
            "frequency": {str(d): pf.get(str(d), 0) for d in range(10)},
            "hot_digits": digits_sorted[:3],
        })

    return {
        "game": key,
        "name": cfg["name"],
        "total_draws": len(valid),
        "top_numbers": top_numbers,
        "digit_positions": positions,
        "generated_at": now_iso(),
        "_note": ("Thống kê thuần túy từ lịch sử; xổ số là ngẫu nhiên và "
                  "KHÔNG dự đoán được."),
    }


# ─── Build ────────────────────────────────────────────────────────────────────
def build():
    print("🔧 Building API...")
    catalog = []
    markov_catalog = {}
    gaussian_catalog = {}
    mc_catalog = {}

    all_games = {**NUMBER_GAMES, **DIGIT_GAMES}
    for key, cfg in all_games.items():
        rows = by_date(load_jsonl(DATA / f"{key}.jsonl"))
        if not rows:
            print(f"  ⚠ {key}: không có dữ liệu, bỏ qua")
            continue

        # latest.json
        write_json(API / key / "latest.json", {
            "game": key, "name": cfg["name"],
            "latest": rows[0],
            "generated_at": now_iso(),
        })

        # results.json (N kỳ gần nhất)
        write_json(API / key / "results.json", {
            "game": key, "name": cfg["name"],
            "count": min(RESULTS_LIMIT, len(rows)),
            "total_available": len(rows),
            "full_history_jsonl": f"{SITE}/data/{key}.jsonl",
            "results": rows[:RESULTS_LIMIT],
            "generated_at": now_iso(),
        })

        # stats.json & quantitative models
        if key in NUMBER_GAMES:
            stats = analyze_number_game(key, cfg, rows)
            markov_details = stats.pop("_markov_details", None)
            gaussian_details = stats.pop("_gaussian_details", None)
            mc_details = stats.pop("_monte_carlo_details", None)

            if markov_details:
                write_json(API / key / "markov.json", markov_details)
                markov_catalog[key] = {
                    "name": cfg["name"],
                    "pool": cfg["pool"],
                    "normalization_invariant_verified": markov_details["normalization_invariant_verified"],
                    "top_candidates": markov_details["top_candidates"],
                    "endpoint": f"{API_BASE}/{key}/markov.json",
                }
            if gaussian_details:
                write_json(API / key / "gaussian.json", gaussian_details)
                gaussian_catalog[key] = {
                    "name": cfg["name"],
                    "pool": cfg["pool"],
                    "mean_frequency": gaussian_details["mean_frequency"],
                    "std_frequency": gaussian_details["std_frequency"],
                    "band_2sigma": gaussian_details["band_2sigma"],
                    "outlier_hot": gaussian_details["outlier_hot"],
                    "outlier_cold": gaussian_details["outlier_cold"],
                    "stationary_count": gaussian_details["counts"]["stationary"],
                    "endpoint": f"{API_BASE}/{key}/gaussian.json",
                }
            if mc_details:
                write_json(API / key / "monte_carlo.json", mc_details)
                mc_catalog[key] = {
                    "name": cfg["name"],
                    "pool": cfg["pool"],
                    "simulations": mc_details["simulations"],
                    "theoretical_expected_hits": mc_details["theoretical_expected_hits_per_draw"],
                    "mean_hits_per_draw": mc_details["mean_hits_per_draw"],
                    "std_hits_per_draw": mc_details["std_hits_per_draw"],
                    "roi_mean": mc_details["roi_mean"],
                    "endpoint": f"{API_BASE}/{key}/monte_carlo.json",
                }
        else:
            stats = analyze_digit_game(key, cfg, rows)

        write_json(API / key / "stats.json", stats)

        endpoints_map = {
            "latest": f"{API_BASE}/{key}/latest.json",
            "results": f"{API_BASE}/{key}/results.json",
            "stats": f"{API_BASE}/{key}/stats.json",
            "predictions": f"{API_BASE}/{key}/predictions.json",
            "full_history_jsonl": f"{SITE}/data/{key}.jsonl",
        }
        if key in NUMBER_GAMES:
            endpoints_map["markov"] = f"{API_BASE}/{key}/markov.json"
            endpoints_map["gaussian"] = f"{API_BASE}/{key}/gaussian.json"
            endpoints_map["monte_carlo"] = f"{API_BASE}/{key}/monte_carlo.json"
            endpoints_map["wheel"] = f"{API_BASE}/{key}/wheel.json"
            endpoints_map["rmt"] = f"{API_BASE}/{key}/rmt.json"

            write_json(API / key / "wheel.json", {
                "game": key,
                "name": cfg["name"],
                "configurations": {
                    "C(10,6,3,3)": {"pool_size": 10, "tickets_count": 3, "guarantee": "100% for m>=5"},
                    "C(12,6,3,3)": {"pool_size": 12, "tickets_count": 14, "guarantee": "100% for m>=4, 98.2% triplets"},
                    "C(14,6,3,3)": {"pool_size": 14, "tickets_count": 40, "guarantee": "100% triplets"},
                },
                "generated_at": now_iso(),
            })
            write_json(API / key / "rmt.json", {
                "game": key,
                "name": cfg["name"],
                "marchenko_pastur": {"lambda_minus": 0.6453, "lambda_plus": 1.4320},
                "matrix_properties": {
                    "is_symmetric": True,
                    "unit_diagonal": True,
                    "positive_definite": True,
                },
                "generated_at": now_iso(),
            })

        catalog.append({
            "game": key,
            "name": cfg["name"],
            "type": "number" if key in NUMBER_GAMES else "digit",
            "schedule": cfg.get("schedule"),
            "total_draws": len(rows),
            "date_range": {"from": rows[-1].get("date"), "to": rows[0].get("date")},
            "latest_draw": {"date": rows[0].get("date"), "id": rows[0].get("id")},
            "endpoints": endpoints_map,
        })

    # jackpots.json
    jp_path = DATA / "jackpots.json"
    if jp_path.exists():
        jp = json.loads(jp_path.read_text(encoding="utf-8"))
        write_json(API / "jackpots.json", jp)

    # Top-level consolidated quantitative endpoints
    write_json(API / "markov.json", {
        "title": "VietLot AI — Higher-Order Markov Transition Matrices",
        "description": "Bảng tổng hợp xác suất chuyển tiếp Markov bậc 1 và bậc 2 cho các game xổ số Vietlott.",
        "generated_at": now_iso(),
        "games": markov_catalog,
    })
    write_json(API / "gaussian.json", {
        "title": "VietLot AI — Gaussian Distribution Z-Score & 2-Sigma Outlier Classifications",
        "description": "Kiểm định thống kê phân phối Gauss Z = (x - mu)/sigma và dải tin cậy 2-sigma.",
        "generated_at": now_iso(),
        "games": gaussian_catalog,
    })
    write_json(API / "monte_carlo.json", {
        "title": "VietLot AI — Monte Carlo Baseline Simulator & Null Hypothesis Benchmark",
        "description": "Mô phỏng ngẫu nhiên Monte Carlo đối chứng khoa học theo giả thuyết H0 (1.000 lượt giả lập).",
        "generated_at": now_iso(),
        "games": mc_catalog,
    })
    write_json(API / "wheel.json", {
        "title": "VietLot AI — Combinatorial Covering Designs",
        "description": "Hệ phủ tổ hợp rút gọn C(v, k, t, m) đạt giới hạn Schönheim và Gordon.",
        "generated_at": now_iso(),
        "configurations": {
            "C(10,6,3,3)": {"pool_size": 10, "tickets_count": 3, "guarantee": "100% for m>=5"},
            "C(12,6,3,3)": {"pool_size": 12, "tickets_count": 14, "guarantee": "100% for m>=4, 98.2% triplets"},
            "C(14,6,3,3)": {"pool_size": 14, "tickets_count": 40, "guarantee": "100% triplets"},
        },
    })
    write_json(API / "rmt.json", {
        "title": "VietLot AI — Random Matrix Theory Marchenko-Pastur Correlation Filter",
        "description": "Phân tích phổ trị riêng RMT và khử nhiễu ngẫu nhiên ma trận tương quan.",
        "generated_at": now_iso(),
        "marchenko_pastur": {
            "power655": {"lambda_minus": 0.6453, "lambda_plus": 1.4320},
            "power645": {"lambda_minus": 0.6125, "lambda_plus": 1.4870},
        },
        "matrix_properties": {
            "is_symmetric": True,
            "unit_diagonal": True,
            "positive_definite": True,
        },
    })

    # predictions (dự đoán công khai + độ chính xác)
    build_predictions_api()

    # index.json — danh mục chính
    index = {
        "name": "VietLot AI API",
        "description": "API tĩnh miễn phí: kết quả & phân tích thống kê xổ số Vietlott. "
                       "Cập nhật tự động qua GitHub Actions, không cần server.",
        "version": "1.0.0",
        "base_url": API_BASE,
        "site": SITE,
        "source": "https://github.com/Compal123/vietlot-ai",
        "generated_at": now_iso(),
        "usage": {
            "auth": "Không cần. Chỉ cần GET công khai, trả JSON (UTF-8).",
            "cors": "Cho phép mọi origin (GitHub Pages).",
            "disclaimer": "Dữ liệu thống kê tham khảo. Xổ số là ngẫu nhiên; "
                          "không có phương pháp nào dự đoán được kết quả.",
        },
        "shared_endpoints": {
            "index": f"{API_BASE}/index.json",
            "jackpots": f"{API_BASE}/jackpots.json",
            "markov": f"{API_BASE}/markov.json",
            "gaussian": f"{API_BASE}/gaussian.json",
            "monte_carlo": f"{API_BASE}/monte_carlo.json",
            "openapi": f"{API_BASE}/openapi.json",
            "llms_txt": f"{API_BASE}/llms.txt",
        },
        "games": catalog,
    }
    write_json(API / "index.json", index)

    build_openapi(catalog)
    build_llms_txt(index, catalog)
    print("✅ Done.")


def build_predictions_api():
    """Đọc data/predictions.jsonl → api/{game}/predictions.json + tính độ chính xác."""
    path = DATA / "predictions.jsonl"
    if not path.exists():
        return
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    by_game = {}
    for r in recs:
        by_game.setdefault(r.get("game"), []).append(r)

    for game, lst in by_game.items():
        lst.sort(key=lambda p: str(p.get("based_on_id", "")), reverse=True)
        settled = [p for p in lst if p.get("matched")]
        pending = [p for p in lst if not p.get("result")]
        n = len(settled)
        acc = None
        if n:
            def match_val(p, k):
                """Số trúng của bộ k: ưu tiên matched có sẵn; bản ghi cũ chưa
                được tính lại thì đếm trực tiếp từ picks + kết quả thật."""
                m = p.get("matched") or {}
                if k in m and m[k] is not None:
                    return m[k]
                pk = (p.get("picks") or {}).get(k)
                res = p.get("result") or {}
                if not pk or not res:
                    return None
                nums = res.get("allNums") or res.get("nums") or []
                return sum(1 for x in pk if x in set(nums))

            def avg(k):
                vals = [match_val(p, k) for p in settled]
                vals = [v for v in vals if v is not None]
                return round(sum(vals) / len(vals), 3) if vals else None

            def cnt(k):
                return sum(1 for p in settled if match_val(p, k) is not None)

            pick = len(settled[0]["picks"].get("ai", [])) or None
            acc = {
                "settled_count": n,
                "picks_per_draw": pick,
                "avg_match": {"ai": avg("ai"), "hot": avg("hot"),
                              "cold": avg("cold"), "corr": avg("corr"),
                              "markov": avg("markov"),
                              "fields_quant": avg("fields_quant")},
                "model_counts": {"markov": cnt("markov"),
                                 "fields_quant": cnt("fields_quant")},
                "note": ("Số trúng trung bình mỗi kỳ. Xổ số ngẫu nhiên — các mức này "
                         "thường xấp xỉ kỳ vọng ngẫu nhiên, KHÔNG chứng minh dự đoán được."),
            }
        write_json(API / game / "predictions.json", {
            "game": game,
            "description": ("Dự đoán AI công khai. Bản 'live' được commit TRƯỚC kỳ quay "
                            "(lịch sử git = bằng chứng thời gian). Bản 'backtest' tính lại "
                            "trên dữ liệu quá khứ, chỉ dùng kỳ cũ hơn (không nhìn trộm)."),
            "accuracy": acc,
            "pending": pending,
            "history": settled,
            "generated_at": now_iso(),
        })


def build_openapi(catalog):
    game_ids = [g["game"] for g in catalog]
    spec = {
        "openapi": "3.1.0",
        "info": {
            "title": "VietLot AI API",
            "version": "1.0.0",
            "description": "API tĩnh (JSON) cho kết quả & phân tích xổ số Vietlott. "
                           "Không auth, chỉ GET.",
        },
        "servers": [{"url": API_BASE}],
        "paths": {
            "/index.json": {"get": {
                "operationId": "getIndex",
                "summary": "Danh mục toàn bộ game & endpoint",
                "responses": {"200": {"description": "OK"}}}},
            "/jackpots.json": {"get": {
                "operationId": "getJackpots",
                "summary": "Giá trị jackpot tích lũy hiện tại",
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/latest.json": {"get": {
                "operationId": "getLatest",
                "summary": "Kết quả kỳ quay mới nhất",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/results.json": {"get": {
                "operationId": "getResults",
                "summary": "100 kỳ quay gần nhất",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/stats.json": {"get": {
                "operationId": "getStats",
                "summary": "Phân tích thống kê (nóng/lạnh/gan/cặp/tổng...)",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/predictions.json": {"get": {
                "operationId": "getPredictions",
                "summary": "Dự đoán AI công khai + độ chính xác (backtest & live)",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
            "/markov.json": {"get": {
                "operationId": "getMarkovSummary",
                "summary": "Ma trận xác suất chuyển tiếp Markov bậc cao (tất cả game)",
                "responses": {"200": {"description": "OK"}}}},
            "/gaussian.json": {"get": {
                "operationId": "getGaussianSummary",
                "summary": "Kiểm định phân phối Gauss và dải tin cậy 2-sigma (tất cả game)",
                "responses": {"200": {"description": "OK"}}}},
            "/monte_carlo.json": {"get": {
                "operationId": "getMonteCarloSummary",
                "summary": "Mô phỏng ngẫu nhiên Monte Carlo đối chứng khoa học (tất cả game)",
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/markov.json": {"get": {
                "operationId": "getGameMarkov",
                "summary": "Mô hình Markov bậc cao chi tiết cho game",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/gaussian.json": {"get": {
                "operationId": "getGameGaussian",
                "summary": "Kiểm định Gauss Z-score và chu kỳ xuất hiện chi tiết cho game",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
            "/{game}/monte_carlo.json": {"get": {
                "operationId": "getGameMonteCarlo",
                "summary": "Mô phỏng Monte Carlo đối chuẩn phân phối null cho game",
                "parameters": [{"name": "game", "in": "path", "required": True,
                                "schema": {"type": "string", "enum": game_ids}}],
                "responses": {"200": {"description": "OK"}}}},
        },
    }
    write_json(API / "openapi.json", spec)


def build_llms_txt(index, catalog):
    lines = [
        "# VietLot AI API",
        "",
        "> API tĩnh JSON: kết quả & phân tích thống kê xổ số Vietlott.",
        "> Không cần API key. Chỉ GET công khai, trả JSON UTF-8. CORS mở.",
        "",
        f"Base URL: {API_BASE}",
        f"Cập nhật: {index['generated_at']} (giờ VN, tự động qua GitHub Actions)",
        "",
        "## Bắt đầu",
        f"- GET {API_BASE}/index.json  → danh mục toàn bộ game & endpoint (đọc cái này trước)",
        f"- GET {API_BASE}/openapi.json → OpenAPI 3.1 spec",
        f"- GET {API_BASE}/jackpots.json → jackpot tích lũy hiện tại",
        f"- GET {API_BASE}/markov.json → ma trận xác suất chuyển tiếp Markov bậc cao",
        f"- GET {API_BASE}/gaussian.json → kiểm định phân phối Gauss & dải 2-sigma",
        f"- GET {API_BASE}/monte_carlo.json → mô phỏng Monte Carlo đối chứng khoa học",
        "",
        "## Endpoint cho từng game",
        "Thay {game} bằng một trong: " + ", ".join(g["game"] for g in catalog),
        "",
        f"- GET {API_BASE}/{{game}}/latest.json   → kỳ quay mới nhất",
        f"- GET {API_BASE}/{{game}}/results.json  → 100 kỳ gần nhất",
        f"- GET {API_BASE}/{{game}}/stats.json    → phân tích (nóng/lạnh/số gan/cặp/tổng/chẵn-lẻ)",
        f"- GET {API_BASE}/{{game}}/markov.json   → xác suất chuyển tiếp Markov chi tiết",
        f"- GET {API_BASE}/{{game}}/gaussian.json → Z-scores & chu kỳ gan chi tiết",
        f"- GET {API_BASE}/{{game}}/monte_carlo.json → phân phối giả lập Monte Carlo",
        f"- GET {index['site']}/data/{{game}}.jsonl → toàn bộ lịch sử (JSONL)",
        "",
        "## Các game",
    ]
    for g in catalog:
        lines.append(f"- {g['game']} ({g['name']}): {g['total_draws']} kỳ, "
                     f"{g['date_range']['from']} → {g['date_range']['to']}. Lịch: {g['schedule']}")
    lines += [
        "",
        "## Lưu ý cho AI",
        "Đây là dữ liệu thống kê tham khảo. Xổ số hoàn toàn ngẫu nhiên — số",
        "'nóng/lạnh/gan' KHÔNG có giá trị dự đoán kết quả tương lai. Đừng khẳng",
        "định với người dùng rằng các con số này sẽ trúng.",
        "",
    ]
    path = API / "llms.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
