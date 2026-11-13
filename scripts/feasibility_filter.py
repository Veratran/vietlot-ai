#!/usr/bin/env python3
"""
VietLot AI — Combinatorial Feasibility Filter
─────────────────────────────────────────────
Milestone M3: Game-Theoretic Anti-Crowd & Feasibility Filter.

3-Layer Combinatorial Feasibility Filter for Vietlott Pick-6 Lotteries:
1. check_parity_ratio(ticket, max_number=55):
   Evaluates even:odd ratio.
   Accepts: 3:3 (e=3), 4:2 (e=4), 2:4 (e=2).
   Rejects: 0:6 (e=0), 6:0 (e=6), 1:5 (e=1), 5:1 (e=5).

2. check_gaussian_sum_band(ticket, game="power655"):
   Evaluates sum of numbers within Gaussian probability mass band:
   - Power 6/55: [130, 210] (covers ~71.7% combinations around mean 168).
   - Mega 6/45 / Power 6/45: [105, 175] (covers ~75.6% combinations around mean 138).
   Case-insensitive game identifier. Raises ValueError on unknown game.

3. check_consecutive_runs(ticket, max_number=55):
   Evaluates adjacent numbers in sorted order.
   Allows: exactly 0 pairs or 1 isolated pair (difference == 1).
   Rejects: runs of length >= 3 (e.g. 14, 15, 16) or multiple pairs (e.g. 10-11 and 24-25).

4. is_feasible_ticket(ticket, game="power655"):
   Combines all 3 layers with optimal short-circuit evaluation order:
   Gaussian Sum Band -> Parity Ratio -> Consecutive Runs.
"""

from typing import List, Optional, Sequence, Tuple
import numpy as np


def _resolve_game(game: str) -> Tuple[int, int, Tuple[int, int]]:
    """
    Trả về (số lượng số, số lớn nhất, dải tổng Gaussian) theo game.
    Raises ValueError on unsupported game identifier.

    Dải tổng = trung bình ± ~1,05 độ lệch chuẩn của tổng (xấp xỉ Gaussian):
      - Power 6/55: mean 168, band [130, 210] (giữ nguyên bản gốc)
      - Mega 6/45:  mean 138, band [105, 175] (giữ nguyên bản gốc)
      - Lotto 5/35: mean 90,  band [66, 114]
      - Keno 10/80: mean 405, band [328, 482]
    """
    if not isinstance(game, str):
        raise ValueError(f"Game identifier must be a string, got {type(game).__name__}")

    g = game.strip().lower()
    if g in ("power655", "power_655", "655"):
        return (6, 55, (130, 210))
    elif g in ("mega645", "power645", "mega_645", "power_645", "645"):
        return (6, 45, (105, 175))
    elif g in ("power535", "lotto535", "power_535", "535"):
        return (5, 35, (66, 114))
    elif g in ("keno",):
        return (10, 80, (328, 482))
    else:
        raise ValueError(
            f"Unsupported game identifier: '{game}'. Supported games: "
            "'power655', 'mega645' (or 'power645'), 'power535', 'keno'"
        )


def _get_game_sum_band(game: str) -> Tuple[int, int]:
    """
    Normalizes game identifier and returns (min_sum, max_sum).
    Raises ValueError on unsupported game identifier.
    """
    return _resolve_game(game)[2]


def _validate_ticket_schema(
    ticket: Sequence[int],
    game: str = "power655",
    max_number: Optional[int] = None,
) -> List[int]:
    """
    Validates ticket sequence, uniqueness, integer types, and valid range.
    Raises TypeError on invalid type, ValueError on invalid values.
    """
    if ticket is None:
        raise TypeError("Ticket cannot be None")
    if not isinstance(ticket, (list, tuple, Sequence)):
        raise TypeError(f"Ticket must be a sequence of integers, got {type(ticket).__name__}")

    # Số lượng số theo game (mặc định 6/55 khi game không xác định được —
    # lỗi game lạ sẽ do _get_game_sum_band / _resolve_game bắt ở lớp trên).
    try:
        pick_count, game_max, _band = _resolve_game(game)
    except (ValueError, TypeError):
        pick_count, game_max = 6, 55

    if len(ticket) != pick_count:
        raise ValueError(f"Ticket must contain exactly {pick_count} numbers, got {len(ticket)}")

    for x in ticket:
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            raise TypeError(f"All ticket elements must be integers, got {type(x).__name__}: {x}")

    if len(set(ticket)) != pick_count:
        raise ValueError("Ticket numbers must be unique, duplicates detected")

    if max_number is None:
        max_number = game_max

    for x in ticket:
        if x < 1 or x > max_number:
            raise ValueError(f"Number {x} out of valid range [1, {max_number}] for game '{game}'")

    return [int(x) for x in ticket]


def check_parity_ratio(ticket: Sequence[int], max_number: int = 55) -> bool:
    """
    Evaluates even:odd ratio of the ticket.

    Accepts:
      - 3:3 (3 even, 3 odd)
      - 4:2 (4 even, 2 odd)
      - 2:4 (2 even, 4 odd)

    Rejects:
      - 0:6 (0 even, 6 odd)
      - 6:0 (6 even, 0 odd)
      - 1:5 (1 even, 5 odd)
      - 5:1 (5 even, 1 odd)

    Args:
        ticket: Sequence of 6 unique integers.
        max_number: Upper bound of lottery pool (default 55).

    Returns:
        bool: True if parity ratio is accepted, False otherwise.
    """
    clean_ticket = _validate_ticket_schema(ticket, max_number=max_number)
    even_count = sum(1 for x in clean_ticket if x % 2 == 0)
    # Chẵn/lẻ cân bằng: số chẵn within ±1 of half the ticket (6 số -> 2..4)
    return abs(even_count - len(clean_ticket) / 2) <= 1


def check_gaussian_sum_band(ticket: Sequence[int], game: str = "power655") -> bool:
    """
    Checks if sum of numbers falls within Gaussian bounds:
      - Power 6/55: [130, 210]
      - Mega 6/45 / Power 6/45: [105, 175]

    Args:
        ticket: Sequence of 6 unique integers.
        game: Game identifier ('power655', 'mega645', 'power645').

    Returns:
        bool: True if min_sum <= sum(ticket) <= max_sum (inclusive), False otherwise.
    """
    band = _get_game_sum_band(game)
    clean_ticket = _validate_ticket_schema(ticket, game=game)
    s = sum(clean_ticket)
    return band[0] <= s <= band[1]


def check_consecutive_runs(ticket: Sequence[int], max_number: int = 55) -> bool:
    """
    Checks for consecutive number runs in sorted order.

    Allows:
      - Exactly 0 pairs (all differences >= 2)
      - 1 isolated pair (one difference == 1, all others >= 2)

    Rejects:
      - Runs of length >= 3 (e.g. 14, 15, 16)
      - Multiple pairs (e.g. 10-11 and 24-25)

    Args:
        ticket: Sequence of 6 unique integers.
        max_number: Upper bound of lottery pool (default 55).

    Returns:
        bool: True if consecutive runs satisfy constraints, False otherwise.
    """
    clean_ticket = _validate_ticket_schema(ticket, max_number=max_number)
    sorted_ticket = sorted(clean_ticket)
    diffs = [sorted_ticket[i + 1] - sorted_ticket[i] for i in range(len(sorted_ticket) - 1)]

    pair_count = sum(1 for d in diffs if d == 1)
    has_run_3 = any(diffs[i] == 1 and diffs[i + 1] == 1 for i in range(len(diffs) - 1))

    if pair_count > 1 or has_run_3:
        return False
    return True


def is_feasible_ticket(ticket: Sequence[int], game: str = "power655") -> bool:
    """
    Combines all 3 feasibility filters with short-circuit evaluation order:
      1. Game band resolution & schema validation
      2. Layer 1: Gaussian sum band (O(1), eliminates ~28% tickets)
      3. Layer 2: Parity ratio (O(1), eliminates ~19% tickets)
      4. Layer 3: Consecutive runs (O(K log K) sort, eliminates remaining ~8-12%)

    Args:
        ticket: Sequence of 6 unique integers.
        game: Game identifier ('power655', 'mega645', 'power645').

    Returns:
        bool: True if ticket satisfies all 3 filters, False otherwise.
    """
    band = _get_game_sum_band(game)
    clean_ticket = _validate_ticket_schema(ticket, game=game)

    # Short-circuit Layer 1: Gaussian Sum Band
    s = sum(clean_ticket)
    if not (band[0] <= s <= band[1]):
        return False

    # Short-circuit Layer 2: Parity Ratio
    even_count = sum(1 for x in clean_ticket if x % 2 == 0)
    if abs(even_count - len(clean_ticket) / 2) > 1:
        return False

    # Short-circuit Layer 3: Consecutive Runs
    sorted_ticket = sorted(clean_ticket)
    diffs = [sorted_ticket[i + 1] - sorted_ticket[i] for i in range(len(sorted_ticket) - 1)]
    pair_count = sum(1 for d in diffs if d == 1)
    if pair_count > 1:
        return False

    return True
