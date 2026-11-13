#!/usr/bin/env python3
"""
VietLot AI — Game-Theoretic Anti-Crowd Optimizer
────────────────────────────────────────────────
Milestone M3: Game-Theoretic Anti-Crowd & Feasibility Filter.

Modules:
1. estimate_selection_density(historical_draws, total_numbers=55, alpha=0.30):
   Estimates empirical number selection density phi(n) calibrated on historical winner statistics:
   - Calendar bias rho_cal ~ 1.222 reflecting +123% winner surge for balls <= 31.
   - Vietnamese lucky ending digits (6, 8, 9) boost 1.08.
   - Bayesian shrinkage alpha=0.30 when winner stats exist.
   - Fallback to calibrated prior when winner stats are absent, empty, or plain number sequences.
   - Guaranteed density sum equals 1.0 +- 1e-6.

2. calculate_crowd_penalty(ticket, total_numbers=55):
   Calculates crowd collision penalty across 4 distinct behavioral vectors:
   - Birthday bias: c <= 2 -> 0.0; c = 3 -> 15.0; c = 4 -> 30.0; c = 5 -> 70.0; c = 6 -> 100.0.
   - Lucky digits: 5.0 * count(ending in 6, 8, 9).
   - Arithmetic progressions: full AP (5 equal differences) -> 100.0; partial AP (4 equal) -> 40.0; else 0.0.
   - Grid geometry on 5-wide slip: >= 4 in same column -> 25.0; monotonic diagonal sweep -> 20.0; combined -> 45.0.
   - Returns dict with keys: total_penalty, birthday_penalty, lucky_digit_penalty, ap_penalty, geometry_penalty.

3. pareto_filter(candidates, statistical_scores, crowd_penalties):
   Rank-1 multi-objective non-dominated sorting:
   - Maximize statistical score, minimize crowd penalty.
   - Deterministic tie-breaking (-score, penalty, sorted ticket).
   - Returns list of dicts with keys: ticket, statistical_score, crowd_penalty, rank, pareto_rank, original_index.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def estimate_selection_density(
    historical_draws: Union[Sequence[Dict[str, Any]], Sequence[Sequence[int]]],
    total_numbers: int = 55,
    alpha: float = 0.30,
) -> Dict[int, float]:
    """
    Estimates empirical number selection density phi(n) calibrated on historical winner statistics.

    Args:
        historical_draws: Historical lottery draws with or without winner metadata.
        total_numbers: Total number universe (default 55 for Power 6/55, 45 for Mega 6/45).
        alpha: Bayesian shrinkage power factor (default 0.30).

    Returns:
        Dict mapping ball number (1..total_numbers) to normalized selection density phi(n).
        Guarantees sum(phi(n)) == 1.0 +- 1e-6.
    """
    if total_numbers <= 0:
        raise ValueError("total_numbers must be positive")

    N = total_numbers

    # 1. Calibrated Prior:
    # rho_cal = (25.50 / 11.43)**(1/4) ~ 1.222 (+123% surge for balls <= 31)
    # Lucky endings 6, 8, 9 boost 1.08
    rho_cal = 1.222
    prior_w: Dict[int, float] = {}
    for n in range(1, N + 1):
        w = 1.0
        if n <= 31:
            w *= rho_cal
        if (n % 10) in (6, 8, 9):
            w *= 1.08
        prior_w[n] = w

    # 2. Check for winner statistics in historical_draws
    valid_draws: List[Dict[str, Any]] = []
    if historical_draws:
        for d in historical_draws:
            if isinstance(d, dict) and "winners" in d and isinstance(d["winners"], dict):
                w = d["winners"]
                has_any_winner = any(
                    isinstance(w.get(k), (int, float)) and w[k] > 0
                    for k in ("Giải nhất", "Giải nhì", "Giải ba", "Jackpot 1", "Jackpot 2")
                )
                if has_any_winner:
                    valid_draws.append(d)

    # 3. Fallback to calibrated prior if winner statistics are absent, empty, or plain number sequences
    if not valid_draws:
        total_prior = sum(prior_w.values())
        return {n: float(prior_w[n] / total_prior) for n in range(1, N + 1)}

    # 4. Bayesian attribution with shrinkage when winner data exists
    nhat_vals = [
        float(d["winners"]["Giải nhất"])
        for d in valid_draws
        if "Giải nhất" in d["winners"] and isinstance(d["winners"]["Giải nhất"], (int, float))
    ]
    ba_vals = [
        float(d["winners"]["Giải ba"])
        for d in valid_draws
        if "Giải ba" in d["winners"] and isinstance(d["winners"]["Giải ba"], (int, float))
    ]
    nhi_vals = [
        float(d["winners"]["Giải nhì"])
        for d in valid_draws
        if "Giải nhì" in d["winners"] and isinstance(d["winners"]["Giải nhì"], (int, float))
    ]

    mean_nhat = float(np.mean(nhat_vals)) if nhat_vals and np.mean(nhat_vals) > 0 else 0.0
    mean_ba = float(np.mean(ba_vals)) if ba_vals and np.mean(ba_vals) > 0 else 0.0
    mean_nhi = float(np.mean(nhi_vals)) if nhi_vals and np.mean(nhi_vals) > 0 else 0.0

    ball_multipliers: Dict[int, List[float]] = {n: [] for n in range(1, N + 1)}

    for d in valid_draws:
        res = d.get("result", [])[:6]
        w = d["winners"]

        r_parts: List[float] = []
        if mean_nhat > 0 and "Giải nhất" in w and isinstance(w["Giải nhất"], (int, float)):
            r_parts.append(float(w["Giải nhất"]) / mean_nhat)
        if mean_ba > 0 and "Giải ba" in w and isinstance(w["Giải ba"], (int, float)):
            r_parts.append(float(w["Giải ba"]) / mean_ba)
        if mean_nhi > 0 and "Giải nhì" in w and isinstance(w["Giải nhì"], (int, float)):
            r_parts.append(float(w["Giải nhì"]) / mean_nhi)

        r_draw = float(np.mean(r_parts)) if r_parts else 1.0

        for b in res:
            if isinstance(b, (int, np.integer)) and 1 <= b <= N:
                ball_multipliers[int(b)].append(r_draw)

    emp_w: Dict[int, float] = {}
    for n in range(1, N + 1):
        if ball_multipliers[n]:
            y_bar = float(np.mean(ball_multipliers[n]))
            emp_w[n] = prior_w[n] * (y_bar ** alpha)
        else:
            emp_w[n] = prior_w[n]

    total_emp = sum(emp_w.values())
    return {n: float(emp_w[n] / total_emp) for n in range(1, N + 1)}


def calculate_crowd_penalty(ticket: Sequence[int], total_numbers: int = 55) -> Dict[str, Any]:
    """
    Calculates crowd collision penalty (Crowd Collision Penalty) for a ticket.

    Args:
        ticket: Sequence of 6 unique integers in range [1, total_numbers].
        total_numbers: Upper bound of lottery pool (default 55).

    Returns:
        Dict with keys:
            - total_penalty: float
            - birthday_penalty: float
            - lucky_digit_penalty: float
            - ap_penalty: float
            - geometry_penalty: float
    """
    if ticket is None:
        raise TypeError("Ticket cannot be None")
    if not isinstance(ticket, (list, tuple, Sequence)):
        raise TypeError(f"Ticket must be a sequence of integers, got {type(ticket).__name__}")

    n_ticket = len(ticket)
    if n_ticket < 1:
        raise ValueError("Ticket must not be empty")

    for x in ticket:
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            raise TypeError(f"All ticket elements must be integers, got {type(x).__name__}: {x}")
        if x < 1 or x > total_numbers:
            raise ValueError(f"Number {x} out of valid range [1, {total_numbers}]")

    if len(set(ticket)) != n_ticket:
        raise ValueError("Ticket numbers must be unique, duplicates detected")

    s = sorted(int(x) for x in ticket)

    # 1. Birthday / Calendar bias penalty (balls <= 31)
    # Theo TỈ LỆ số <= 31 trong vé (bản gốc cho vé 6 số: <=2 -> 0; 3 -> 15;
    # 4 -> 30; 5 -> 70; 6 -> 100 — tương đương ngưỡng tỉ lệ 1/3, 1/2, 2/3, 5/6)
    c_cal = sum(1 for x in s if x <= 31)
    r_cal = c_cal / n_ticket
    if r_cal <= 1 / 3:
        p_birthday = 0.0
    elif r_cal <= 1 / 2:
        p_birthday = 15.0
    elif r_cal <= 2 / 3:
        p_birthday = 30.0
    elif r_cal < 1:
        p_birthday = 70.0
    else:
        p_birthday = 100.0

    # 2. Lucky ending digits penalty (6, 8, 9)
    # 5.0 * count(ending in 6, 8, 9)
    k_lucky = sum(1 for x in s if (x % 10) in (6, 8, 9))
    p_lucky = float(5.0 * k_lucky)

    # 3. Arithmetic progressions penalty (AP)
    deltas = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    delta_counts = Counter(deltas)
    f_max = max(delta_counts.values()) if delta_counts else 0
    if f_max == len(s) - 1 and len(s) > 1:
        p_ap = 100.0
    elif f_max == len(s) - 2 and len(s) > 2:
        p_ap = 40.0
    else:
        p_ap = 0.0

    # 4. Grid geometry penalty on 5-wide Vietlott play slip
    # col = (n - 1) % 5, row = (n - 1) // 5
    cols = [(x - 1) % 5 for x in s]
    rows = [(x - 1) // 5 for x in s]

    # 4a. Column alignment: >= 4 numbers in same column -> 25.0
    col_counts = Counter(cols)
    p_col = 25.0 if any(cnt >= 4 for cnt in col_counts.values()) else 0.0

    # 4b. Monotonic diagonal sweep: monotonic sweep across 6 distinct rows OR >= 3 steps of 4 or 6 -> 20.0
    p_diag = 0.0
    if len(set(rows)) == len(s):
        inc = all(cols[i] <= cols[i + 1] for i in range(len(s) - 1)) and (cols[-1] - cols[0] >= 2)
        dec = all(cols[i] >= cols[i + 1] for i in range(len(s) - 1)) and (cols[0] - cols[-1] >= 2)
        if inc or dec:
            p_diag = 20.0

    diag_step_6 = sum(1 for d in deltas if d == 6)
    diag_step_4 = sum(1 for d in deltas if d == 4)
    if diag_step_6 >= 3 or diag_step_4 >= 3:
        p_diag = 20.0

    p_geometry = float(p_col + p_diag)
    total_penalty = float(p_birthday + p_lucky + p_ap + p_geometry)

    return {
        "total_penalty": total_penalty,
        "birthday_penalty": float(p_birthday),
        "lucky_digit_penalty": float(p_lucky),
        "ap_penalty": float(p_ap),
        "geometry_penalty": float(p_geometry),
    }


def pareto_filter(
    candidates: Sequence[Sequence[int]],
    statistical_scores: Sequence[float],
    crowd_penalties: Sequence[float],
) -> List[Dict[str, Any]]:
    """
    Rank-1 multi-objective non-dominated Pareto front filter.

    Objectives:
      - Maximize statistical_scores
      - Minimize crowd_penalties

    Args:
        candidates: Sequence of candidate lottery tickets.
        statistical_scores: Corresponding statistical scores (higher is better).
        crowd_penalties: Corresponding crowd collision penalties (lower is better).

    Returns:
        List of dicts representing rank-1 non-dominated candidates sorted deterministically by
        (-statistical_score, crowd_penalty, sorted ticket).
    """
    if len(candidates) != len(statistical_scores) or len(candidates) != len(crowd_penalties):
        raise ValueError(
            f"Input lengths do not match: candidates={len(candidates)}, "
            f"statistical_scores={len(statistical_scores)}, "
            f"crowd_penalties={len(crowd_penalties)}"
        )

    if not candidates:
        return []

    M = len(candidates)
    dominated = [False] * M

    for i in range(M):
        s_i = float(statistical_scores[i])
        p_i = float(crowd_penalties[i])
        for j in range(M):
            if i == j:
                continue
            s_j = float(statistical_scores[j])
            p_j = float(crowd_penalties[j])
            # Candidate j strictly dominates i if j is at least as good in both objectives
            # and strictly better in at least one objective:
            if (s_j >= s_i and p_j <= p_i) and (s_j > s_i or p_j < p_i):
                dominated[i] = True
                break

    pareto_indices = [i for i in range(M) if not dominated[i]]

    # Deterministic sorting: (-score, penalty, sorted_ticket)
    pareto_indices.sort(
        key=lambda idx: (
            -float(statistical_scores[idx]),
            float(crowd_penalties[idx]),
            tuple(sorted(candidates[idx])),
        )
    )

    return [
        {
            "ticket": list(candidates[idx]),
            "statistical_score": float(statistical_scores[idx]),
            "crowd_penalty": float(crowd_penalties[idx]),
            "rank": 1,
            "pareto_rank": 1,
            "original_index": int(idx),
        }
        for idx in pareto_indices
    ]
