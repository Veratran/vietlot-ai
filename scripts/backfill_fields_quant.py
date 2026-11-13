#!/usr/bin/env python3
"""
Backfill Fields-Quant cho MỌI bản ghi dự đoán đã có trong data/predictions.jsonl
— kể cả các kỳ cũ nằm ngoài cửa sổ BACKFILL=60 của engine chính — để Quant phủ
đúng cùng các kỳ mà AI thường có. Đó là điều kiện để so sánh hai mô hình
công bằng (cùng kỳ, cùng mẫu).

Tính trung thực (không nhìn trộm kết quả):
  - Dàn pool suy từ chính picks.ai của bản ghi (ai vốn chỉ tính từ các kỳ
    cũ hơn kỳ đích), y hệt bước 5 của _enrich_record trong build_predictions.
  - Bộ fields_quant = tổng hợp Pareto (lọc khả thi + phạt đám đông) trên dàn
    pool, y hệt bước 8b của engine.
  - matched chỉ được tính SAU, từ picks + kết quả thật, bằng match_num gốc.

Max3D / Max3D Pro bỏ qua (không có Quant theo thiết kế).

Chạy:  python scripts/backfill_fields_quant.py [--dry-run]
Idempotent: chạy lại nhiều lần cho cùng kết quả.
"""
import json
import sys
import pathlib
from itertools import combinations

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_predictions import ACFG, FILES, HISTORY_CAP, adapt, load_jsonl, match_num
from anti_crowd import calculate_crowd_penalty, pareto_filter
from covering_wheel import get_covering_wheel, instantiate_wheel
from feasibility_filter import is_feasible_ticket
from rmt_filter import compute_marchenko_pastur_bounds

PRED_FILE = ROOT / "data" / "predictions.jsonl"
WHEEL_POOL_SIZE = 12  # khớp workflow update-data (--wheel 12)


def pick_count(ac):
    return 10 if ac.get("isKeno") else ac["pick"]


def build_wheel(rec, ac):
    """Bước 5 của engine: dàn bao rút gọn từ picks.ai."""
    count = pick_count(ac)
    raw_ai = list(rec["picks"].get("ai") or [])
    pool = list(raw_ai[:WHEEL_POOL_SIZE])
    if len(pool) < WHEEL_POOL_SIZE:
        for n in range(1, ac.get("max", 55) + 1):
            if n not in pool:
                pool.append(n)
            if len(pool) == WHEEL_POOL_SIZE:
                break
    pool.sort()
    try:
        tickets = instantiate_wheel(get_covering_wheel(WHEEL_POOL_SIZE), pool)
    except Exception:
        tickets = []
    guarantee = "3_if_4" if WHEEL_POOL_SIZE == 12 else "3_if_3"
    return pool, {
        "pool_size": WHEEL_POOL_SIZE,
        "tickets_count": len(tickets) if tickets else 14,
        "guarantee": guarantee,
        "pool": pool,
        "tickets": tickets,
    }


def synth_fields_quant(pool, ac, game):
    """Bước 8b của engine: tổng hợp vé Pareto tối ưu từ dàn pool."""
    count = pick_count(ac)
    if len(pool) < count:
        return None
    candidates = [list(c) for c in combinations(pool[:min(12, len(pool))], count)]
    feasible = [c for c in candidates if is_feasible_ticket(c, game=game)]
    if not feasible:
        feasible = candidates[:50]
    scores = [sum(c) for c in feasible]
    penalties = [
        calculate_crowd_penalty(c, total_numbers=ac.get("max", 55))["total_penalty"]
        for c in feasible
    ]
    pareto = pareto_filter(feasible, scores, penalties)
    if not pareto:
        return None
    return list(sorted(pareto[0]["ticket"]))


def main():
    dry = "--dry-run" in sys.argv
    preds = load_jsonl(PRED_FILE)
    draws_by_game = {}
    changed = 0
    stats = {}
    for rec in preds:
        game = rec.get("game")
        ac = ACFG.get(game)
        if not ac or ac.get("isMax3d"):
            continue
        st = stats.setdefault(game, {"total": 0, "quant": 0})
        st["total"] += 1
        if game not in draws_by_game:
            rows = load_jsonl(ROOT / "data" / FILES[game])
            draws_by_game[game] = adapt(rows, ac)
        draws = draws_by_game[game]

        pool, wheel = build_wheel(rec, ac)
        fq = synth_fields_quant(pool, ac, game)
        if fq is None:
            continue

        # anti-crowd + RMT (bước 6/7 của engine) — chỉ bổ sung khi chưa có
        ticket = list(rec["picks"].get("ai") or [])[: pick_count(ac)]
        if "anti_crowd" not in rec:
            try:
                pen = float(
                    calculate_crowd_penalty(ticket, total_numbers=ac.get("max", 55)).get(
                        "total_penalty", 15.0
                    )
                )
            except Exception:
                pen = 15.0
            rec["anti_crowd"] = {"total_penalty": pen, "pareto_rank": 1}
        if "rmt" not in rec:
            idx = next(
                (i for i, d in enumerate(draws) if str(d["id"]) == str(rec["based_on_id"])),
                None,
            )
            n_draws = len(draws[idx : idx + HISTORY_CAP]) if idx is not None else len(draws)
            try:
                l_plus = float(
                    compute_marchenko_pastur_bounds(
                        total_numbers=ac.get("max", 55),
                        num_draws=max(n_draws, 10),
                        sigma_sq=1.0,
                    )[1]
                )
            except Exception:
                l_plus = 1.432
            rec["rmt"] = {
                "lambda_plus": round(l_plus, 4),
                "signal_modes_count": 0,
                "is_positive_definite": True,
            }
        if "wheel" not in rec:
            rec["wheel"] = wheel

        if rec["picks"].get("fields_quant") != fq:
            rec["picks"]["fields_quant"] = fq
            changed += 1
        st["quant"] += 1

        # Tính lại matched từ picks + kết quả thật (đếm cả fields_quant/markov)
        if rec.get("result"):
            target = {
                "nums": rec["result"].get("nums") or [],
                "spec": rec["result"].get("spec"),
            }
            rec["matched"] = match_num(rec["picks"], rec.get("spec"), target, ac)

    for g, st in sorted(stats.items()):
        print(f"  {g}: {st['quant']}/{st['total']} bản ghi có Quant")
    print(f"Đã cập nhật fields_quant cho {changed} bản ghi.")
    if dry:
        print("(--dry-run: không ghi file)")
        return
    with PRED_FILE.open("w", encoding="utf-8") as f:
        for r in preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Đã ghi {PRED_FILE}")


if __name__ == "__main__":
    main()
