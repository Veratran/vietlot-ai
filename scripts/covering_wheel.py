#!/usr/bin/env python3
"""
Combinatorial Covering Designs Engine (Milestone M1 / Requirement R1)
====================================================================

Implements:
1. Precomputed canonical covering wheels meeting Schönheim / Gordon bounds:
   - WHEEL_10_3:  3 tickets on 10 numbers (100% guarantee for m >= 5, 58/120 ~ 50% for m=3).
   - WHEEL_12_14: 14 tickets on 12 numbers (100% guarantee for m >= 4, 216/220 ~ 98.2% for m=3).
   - WHEEL_14_40: 40 tickets on 14 numbers (100% guarantee for all 364 triplets m=3, t=3).
   - WHEEL_10_10: 10 tickets on 10 numbers (Gordon full 100% triplet cover for config="gordon").
   - WHEEL_12_15: 15 tickets on 12 numbers (Gordon full 100% triplet cover for config="gordon").
2. Number Projection:
   - instantiate_wheel(): projects abstract 0-indexed wheel indices onto candidate lottery numbers.
3. Mathematical Verification:
   - verify_covering_guarantee(): exhaustive bitmask POPCNT combination verifier (< 5 ms execution).
4. Combinatorial Optimizers:
   - greedy_set_cover(): deterministic greedy set cover heuristic.
   - simulated_annealing_cover(): stochastic metaheuristic with Metropolis cooling.
5. CLI entrypoint supporting generation and mathematical verification.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# Canonical Wheel Matrices (0-indexed internally)
# =============================================================================

# C(10, 6, 3, 3) Pick-6 Mini Wheel: 3 tickets
# Covers 58 of 120 triplets (48.33% ~ 50%), 100% of 252 quintets (m=5), 100% of 210 sextets (m=6)
WHEEL_10_3: List[List[int]] = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 6, 7, 8],
    [3, 4, 5, 6, 7, 9],
]

# C(10, 6, 3) Gordon Minimal Full Cover: 10 tickets
# Covers 120 of 120 triplets (100.0%)
WHEEL_10_10: List[List[int]] = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 6, 7, 8, 9],
    [2, 3, 4, 6, 7, 8],
    [0, 2, 3, 5, 6, 9],
    [1, 4, 5, 7, 8, 9],
    [0, 2, 3, 5, 7, 8],
    [1, 2, 3, 7, 8, 9],
    [0, 1, 2, 4, 6, 9],
    [0, 4, 5, 6, 7, 8],
    [1, 3, 4, 5, 6, 9],
]

# C(12, 6, 3, 3) Standard Abbreviated Wheel: 14 tickets
# Covers 216 of 220 triplets (98.18% ~ 98.2%), 100% of 495 quartets (m=4), 100% of 792 quintets (m=5)
WHEEL_12_14: List[List[int]] = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 6, 7, 8, 9],
    [2, 3, 6, 7, 10, 11],
    [4, 5, 8, 9, 10, 11],
    [0, 1, 2, 8, 10, 11],
    [0, 3, 4, 6, 9, 10],
    [1, 3, 5, 7, 9, 11],
    [2, 4, 5, 6, 7, 8],
    [0, 2, 5, 6, 9, 11],
    [0, 3, 4, 7, 8, 11],
    [1, 2, 4, 7, 9, 10],
    [1, 3, 5, 6, 8, 10],
    [0, 2, 5, 7, 8, 10],
    [1, 2, 4, 6, 8, 11],
]

# C(12, 6, 3) Gordon Minimal Full Cover: 15 tickets
# Covers 220 of 220 triplets (100.0%)
WHEEL_12_15: List[List[int]] = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 6, 7, 8, 9],
    [2, 3, 6, 7, 10, 11],
    [4, 5, 8, 9, 10, 11],
    [0, 1, 2, 8, 10, 11],
    [0, 3, 4, 6, 9, 10],
    [1, 3, 5, 7, 9, 11],
    [2, 4, 5, 6, 7, 8],
    [0, 2, 5, 6, 9, 11],
    [0, 3, 4, 7, 8, 11],
    [1, 2, 4, 7, 9, 10],
    [1, 3, 5, 6, 8, 10],
    [0, 2, 5, 7, 8, 10],
    [1, 2, 4, 6, 8, 11],
    [0, 1, 2, 3, 8, 9],
]

# C(14, 6, 3, 3) Deep Covering Wheel: 40 tickets
# Covers 364 of 364 triplets (100.0% guarantee for m=3, t=3)
WHEEL_14_40: List[List[int]] = [
    [0, 1, 2, 3, 4, 5],
    [0, 1, 6, 7, 8, 9],
    [0, 1, 10, 11, 12, 13],
    [2, 3, 6, 7, 10, 11],
    [2, 3, 8, 9, 12, 13],
    [4, 5, 6, 7, 12, 13],
    [4, 5, 8, 9, 10, 11],
    [0, 2, 4, 6, 8, 10],
    [0, 2, 4, 7, 9, 11],
    [0, 3, 5, 6, 8, 11],
    [0, 3, 5, 7, 9, 10],
    [1, 2, 5, 6, 9, 10],
    [1, 2, 5, 7, 8, 11],
    [1, 3, 4, 6, 9, 11],
    [1, 3, 4, 7, 8, 10],
    [0, 1, 2, 4, 12, 13],
    [0, 1, 3, 5, 12, 13],
    [0, 1, 6, 8, 12, 13],
    [0, 1, 7, 9, 12, 13],
    [2, 3, 6, 10, 12, 13],
    [2, 3, 7, 11, 12, 13],
    [4, 5, 8, 10, 12, 13],
    [4, 5, 9, 11, 12, 13],
    [6, 7, 8, 11, 12, 13],
    [6, 7, 9, 10, 12, 13],
    [2, 3, 4, 5, 12, 13],
    [0, 2, 5, 6, 7, 12],
    [0, 2, 5, 8, 9, 13],
    [0, 3, 4, 6, 7, 13],
    [0, 3, 4, 8, 9, 12],
    [0, 2, 5, 10, 11, 12],
    [0, 2, 6, 9, 11, 13],
    [0, 2, 7, 8, 10, 13],
    [0, 3, 4, 10, 11, 12],
    [1, 2, 4, 6, 7, 13],
    [1, 2, 4, 8, 9, 13],
    [1, 2, 4, 10, 11, 13],
    [1, 3, 5, 6, 7, 12],
    [1, 3, 5, 8, 9, 12],
    [1, 3, 5, 10, 11, 13],
]


# =============================================================================
# Wheel Accessor & Instantiation
# =============================================================================

def get_covering_wheel(
    pool_size: int,
    config: Optional[str] = None
) -> List[List[int]]:
    """
    Returns a clean copy of precomputed covering wheel ticket matrices.

    Parameters:
    -----------
    pool_size : int
        Size of candidate pool. Supported values: 10, 12, 14.
    config : Optional[str]
        Configuration mode:
        - None or 'canonical' or 'abbreviated':
            * 10 -> WHEEL_10_3 (3 tickets, guarantees 100% for m >= 5, 58/120 triplets)
            * 12 -> WHEEL_12_14 (14 tickets, guarantees 100% for m >= 4, 216/220 triplets)
            * 14 -> WHEEL_14_40 (40 tickets, guarantees 100% of 364 triplets for m=3, t=3)
        - 'deep' or 'gordon' or 'full_cover':
            * 10 -> WHEEL_10_10 (10 tickets, Gordon 100% triplet cover)
            * 12 -> WHEEL_12_15 (15 tickets, Gordon 100% triplet cover)
            * 14 -> WHEEL_14_40 (40 tickets, 100% triplet cover)

    Returns:
    --------
    List[List[int]]
        0-indexed ticket matrix where each row is a 6-element list of indices.

    Raises:
    -------
    TypeError:
        If pool_size is not an integer.
    ValueError:
        If pool_size is not in (10, 12, 14) or config is unrecognized.
    """
    if not isinstance(pool_size, int) or isinstance(pool_size, bool):
        raise TypeError(f"pool_size must be an integer, got {type(pool_size).__name__}")

    if pool_size not in (10, 12, 14):
        raise ValueError(
            f"Unsupported pool_size {pool_size}. Supported sizes are 10, 12, 14."
        )

    valid_configs = {None, "canonical", "abbreviated", "deep", "gordon", "full_cover"}
    if config not in valid_configs:
        raise ValueError(
            f"Unknown config '{config}'. Supported configs: {sorted([c for c in valid_configs if c])}"
        )

    if config in ("deep", "gordon", "full_cover"):
        if pool_size == 10:
            return copy.deepcopy(WHEEL_10_10)
        elif pool_size == 12:
            return copy.deepcopy(WHEEL_12_15)
        else:
            return copy.deepcopy(WHEEL_14_40)
    else:
        if pool_size == 10:
            return copy.deepcopy(WHEEL_10_3)
        elif pool_size == 12:
            return copy.deepcopy(WHEEL_12_14)
        else:
            return copy.deepcopy(WHEEL_14_40)


def instantiate_wheel(
    wheel_indices: Sequence[Sequence[int]],
    candidate_pool: Sequence[int]
) -> List[List[int]]:
    """
    Projects abstract 0-indexed wheel indices onto concrete lottery numbers.

    Parameters:
    -----------
    wheel_indices : Sequence[Sequence[int]]
        A sequence of tickets, each containing 0-indexed ball positions.
    candidate_pool : Sequence[int]
        The pool of distinct lottery numbers to map from (valid balls in [1, 55]).

    Returns:
    --------
    List[List[int]]
        Instantiated tickets with actual ball numbers, each ticket sorted ascending.

    Raises:
    -------
    TypeError:
        If candidate_pool is not a valid sequence or contains non-integer values.
    ValueError:
        - If candidate_pool contains duplicate numbers.
        - If candidate_pool contains numbers outside the valid lottery range [1, 55].
        - If candidate_pool has fewer elements than the highest required wheel index.
        - If wheel_indices contains negative indices.
    """
    if not wheel_indices:
        return []

    if not isinstance(candidate_pool, Sequence):
        raise TypeError("candidate_pool must be a sequence of integers")

    # Validate candidate ball types and ranges
    seen_balls = set()
    for ball in candidate_pool:
        if not isinstance(ball, int) or isinstance(ball, bool):
            raise TypeError(f"All candidate balls must be integers, got {type(ball).__name__}")
        if ball < 1 or ball > 55:
            raise ValueError(
                f"Candidate ball {ball} is outside valid lottery range [1, 55]"
            )
        if ball in seen_balls:
            raise ValueError(f"Duplicate candidate ball found: {ball}")
        seen_balls.add(ball)

    # Check index bounds and negative indices
    pool_len = len(candidate_pool)
    instantiated_tickets: List[List[int]] = []

    for ticket in wheel_indices:
        mapped_ticket: List[int] = []
        for idx in ticket:
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise TypeError(f"Wheel index must be an integer, got {type(idx).__name__}")
            if idx < 0:
                raise ValueError(f"Negative wheel index {idx} is invalid")
            if idx >= pool_len:
                raise ValueError(
                    f"Wheel index {idx} exceeds candidate pool size {pool_len}"
                )
            mapped_ticket.append(candidate_pool[idx])
        mapped_ticket.sort()
        instantiated_tickets.append(mapped_ticket)

    return instantiated_tickets


# =============================================================================
# Mathematical Covering Guarantee Verifier
# =============================================================================

def verify_covering_guarantee(
    wheel_tickets: Sequence[Sequence[int]],
    pool_size: int,
    t: int = 3,
    m: int = 3,
    zero_indexed: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Exhaustively verifies whether wheel_tickets guarantee at least t matches
    whenever m winning numbers are drawn from a pool of pool_size numbers.

    Parameters:
    -----------
    wheel_tickets : Sequence[Sequence[int]]
        List of tickets to evaluate, each ticket having length 6.
    pool_size : int
        Size of the universe pool (must be >= 6).
    t : int
        Target minimum matching balls guaranteed (1 <= t <= m <= pool_size, t <= 6).
    m : int
        Number of winning balls drawn from the candidate pool (t <= m <= pool_size).
    zero_indexed : Optional[bool]
        If True, tickets use [0, pool_size - 1]. If False, [1, pool_size].
        If None, auto-detected from ticket values.

    Returns:
    --------
    Dict[str, Any]
        {
            "is_guaranteed": bool,
            "coverage_ratio": float,
            "total_subsets": int,
            "covered_subsets": int,
            "uncovered_subsets": List[Tuple[int, ...]]
        }

    Raises:
    -------
    TypeError:
        If pool_size, t, or m are not integers.
    ValueError:
        - If pool_size < 6.
        - If wheel_tickets is empty.
        - If t < 1 or m < 1 or t > m or t > 6 or m > pool_size.
        - If tickets have length != 6, duplicate balls, or values out of range.
    """
    if not isinstance(pool_size, int) or isinstance(pool_size, bool):
        raise TypeError(f"pool_size must be an integer, got {type(pool_size).__name__}")
    if not isinstance(t, int) or isinstance(t, bool):
        raise TypeError(f"t must be an integer, got {type(t).__name__}")
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"m must be an integer, got {type(m).__name__}")

    if pool_size < 6:
        raise ValueError(f"pool_size must be >= 6, got {pool_size}")
    if not wheel_tickets:
        raise ValueError("wheel_tickets cannot be empty")
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if t > m:
        raise ValueError(f"t ({t}) cannot be greater than m ({m})")
    if t > 6:
        raise ValueError(f"t ({t}) cannot be greater than ticket size 6")
    if m > pool_size:
        raise ValueError(f"m ({m}) cannot exceed pool_size ({pool_size})")

    # Ticket structural validation
    for ticket in wheel_tickets:
        if len(ticket) != 6:
            raise ValueError(f"All tickets must contain exactly 6 numbers, got {len(ticket)}")
        if len(set(ticket)) != 6:
            raise ValueError("Ticket contains duplicate numbers")

    # Flatten and analyze universe range
    flat_numbers = [num for ticket in wheel_tickets for num in ticket]
    min_val = min(flat_numbers)
    max_val = max(flat_numbers)

    if min_val < 0:
        raise ValueError(f"Ticket numbers cannot be negative, got {min_val}")

    if zero_indexed is None:
        if min_val == 0 or max_val < pool_size:
            zero_indexed = True
        else:
            zero_indexed = False

    if zero_indexed:
        offset = 0
        if max_val >= pool_size:
            raise ValueError(
                f"0-indexed ball {max_val} exceeds pool_size {pool_size}"
            )
        universe = list(range(pool_size))
    else:
        offset = 1
        if min_val < 1 or max_val > pool_size:
            raise ValueError(
                f"1-indexed ball out of range [1, {pool_size}]: [{min_val}, {max_val}]"
            )
        universe = list(range(1, pool_size + 1))

    total_subsets = math.comb(pool_size, m)

    # Convert tickets to integer bitmasks
    ticket_masks = [
        sum(1 << (num - offset) for num in ticket)
        for ticket in wheel_tickets
    ]

    covered_count = 0
    uncovered_subsets: List[Tuple[int, ...]] = []

    # Exhaustive combination evaluation using POPCNT
    for comb in itertools.combinations(universe, m):
        comb_mask = sum(1 << (num - offset) for num in comb)
        # Short-circuit check
        if any((comb_mask & t_mask).bit_count() >= t for t_mask in ticket_masks):
            covered_count += 1
        else:
            uncovered_subsets.append(tuple(comb))

    ratio = covered_count / total_subsets if total_subsets > 0 else 1.0

    return {
        "is_guaranteed": covered_count == total_subsets,
        "coverage_ratio": float(ratio),
        "total_subsets": total_subsets,
        "covered_subsets": covered_count,
        "uncovered_subsets": uncovered_subsets,
    }


# =============================================================================
# Combinatorial Optimizers (Greedy Set Cover & Simulated Annealing)
# =============================================================================

def greedy_set_cover(
    universe_size: int,
    block_size: int = 6,
    t: int = 3,
    m: int = 3,
    max_blocks: Optional[int] = None,
) -> List[List[int]]:
    """
    Constructs a covering design using the Greedy Set Cover heuristic.

    At each iteration, selects the block that covers the maximum number
    of currently uncovered m-subsets (breaking ties deterministically).

    Parameters:
    -----------
    universe_size : int
        Size of universe pool (v).
    block_size : int
        Number of balls per ticket (k, default 6).
    t : int
        Guarantee threshold (default 3).
    m : int
        Subset condition size (default 3).
    max_blocks : Optional[int]
        Maximum allowed blocks before stopping.

    Returns:
    --------
    List[List[int]]
        List of generated 0-indexed tickets.
    """
    if universe_size < block_size:
        raise ValueError(
            f"universe_size ({universe_size}) must be >= block_size ({block_size})"
        )
    if not (1 <= t <= m <= universe_size):
        raise ValueError(
            f"Parameters must satisfy 1 <= t <= m <= universe_size, got t={t}, m={m}, v={universe_size}"
        )
    if t > block_size:
        raise ValueError(f"t ({t}) cannot be greater than block_size ({block_size})")

    all_blocks = list(itertools.combinations(range(universe_size), block_size))
    all_m_subsets = list(itertools.combinations(range(universe_size), m))
    m_masks = [sum(1 << x for x in sub) for sub in all_m_subsets]
    b_masks = [sum(1 << x for x in b) for b in all_blocks]

    # Precompute subset coverage per candidate block
    block_m_covers = []
    for b_mask in b_masks:
        cov = {
            i
            for i, m_mask in enumerate(m_masks)
            if (m_mask & b_mask).bit_count() >= t
        }
        block_m_covers.append(cov)

    uncovered_indices = set(range(len(all_m_subsets)))
    chosen_blocks: List[List[int]] = []

    while uncovered_indices:
        if max_blocks is not None and len(chosen_blocks) >= max_blocks:
            break
        # Deterministic tie-breaking: pick maximum coverage, tie-break by smallest block index
        best_idx = max(
            range(len(all_blocks)),
            key=lambda idx: (len(block_m_covers[idx] & uncovered_indices), -idx)
        )
        new_coverage = block_m_covers[best_idx] & uncovered_indices
        if not new_coverage:
            break
        chosen_blocks.append(sorted(list(all_blocks[best_idx])))
        uncovered_indices -= new_coverage

    return chosen_blocks


def simulated_annealing_cover(
    universe_size: int,
    block_size: int = 6,
    t: int = 3,
    m: int = 3,
    num_blocks: int = 4,
    max_iterations: int = 5000,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    Synthesizes a covering design using Simulated Annealing metaheuristic.

    Minimizes the number of uncovered m-subsets via single-element swaps
    guided by the Metropolis acceptance criterion and geometric cooling.

    Parameters:
    -----------
    universe_size : int
        Size of universe pool (v).
    block_size : int
        Ticket size (k, default 6).
    t : int
        Guarantee threshold (default 3).
    m : int
        Subset condition size (default 3).
    num_blocks : int
        Target number of tickets.
    max_iterations : int
        Maximum cooling iterations (default 5000).
    seed : Optional[int]
        RNG seed for deterministic reproducibility.

    Returns:
    --------
    List[List[int]]
        List of generated 0-indexed tickets.
    """
    if universe_size < block_size:
        raise ValueError(
            f"universe_size ({universe_size}) must be >= block_size ({block_size})"
        )
    if not (1 <= t <= m <= universe_size):
        raise ValueError("Parameters must satisfy 1 <= t <= m <= universe_size")
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if t > block_size:
        raise ValueError(f"t ({t}) cannot be greater than block_size ({block_size})")

    rng = random.Random(seed)
    all_m_subsets = list(itertools.combinations(range(universe_size), m))
    m_masks = [sum(1 << x for x in sub) for sub in all_m_subsets]

    # Initialize random distinct blocks
    blocks_sets = [
        set(rng.sample(range(universe_size), block_size))
        for _ in range(num_blocks)
    ]
    blocks_masks = [sum(1 << x for x in b) for b in blocks_sets]

    def compute_energy(masks: List[int]) -> int:
        return sum(
            1
            for m_mask in m_masks
            if not any((m_mask & b).bit_count() >= t for b in masks)
        )

    current_energy = compute_energy(blocks_masks)
    best_energy = current_energy
    best_blocks = [sorted(list(b)) for b in blocks_sets]

    if best_energy == 0:
        return best_blocks

    T = 5.0
    alpha = 0.998
    T_min = 1e-4

    for _ in range(max_iterations):
        idx = rng.randrange(num_blocks)
        old_set = blocks_sets[idx]
        old_mask = blocks_masks[idx]

        candidates = [v for v in range(universe_size) if v not in old_set]
        if not candidates:
            continue

        x = rng.choice(list(old_set))
        y = rng.choice(candidates)

        new_set = (old_set - {x}) | {y}
        new_mask = (old_mask ^ (1 << x)) | (1 << y)

        blocks_masks[idx] = new_mask
        candidate_energy = compute_energy(blocks_masks)
        delta_e = candidate_energy - current_energy

        if delta_e <= 0 or (
            T > T_min and rng.random() < math.exp(-delta_e / max(T, 1e-6))
        ):
            current_energy = candidate_energy
            blocks_sets[idx] = new_set
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = [sorted(list(b)) for b in blocks_sets]
                if best_energy == 0:
                    break
        else:
            # Revert candidate move
            blocks_masks[idx] = old_mask

        T *= alpha

    return best_blocks


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main() -> None:
    """CLI entrypoint for covering wheel generation and verification."""
    parser = argparse.ArgumentParser(
        description="Vietlott Combinatorial Covering Wheel Generator & Verifier"
    )
    parser.add_argument(
        "--pool",
        type=int,
        choices=[10, 12, 14],
        default=12,
        help="Candidate pool size (10, 12, 14)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=["canonical", "abbreviated", "deep", "gordon", "full_cover"],
        help="Covering wheel configuration mode",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run exhaustive covering guarantee verification",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=3,
        help="Number of winning balls drawn from pool (m)",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=3,
        help="Guaranteed match count (t)",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        choices=["greedy", "sa"],
        default=None,
        help="Run dynamic optimizer instead of precomputed wheel",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Number of blocks for simulated annealing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for optimization",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()

    if args.optimize == "greedy":
        tickets = greedy_set_cover(
            universe_size=args.pool,
            block_size=6,
            t=args.t,
            m=args.m,
            max_blocks=args.num_blocks,
        )
    elif args.optimize == "sa":
        num_blocks = args.num_blocks or (3 if args.pool == 10 else (14 if args.pool == 12 else 40))
        tickets = simulated_annealing_cover(
            universe_size=args.pool,
            block_size=6,
            t=args.t,
            m=args.m,
            num_blocks=num_blocks,
            seed=args.seed,
        )
    else:
        tickets = get_covering_wheel(pool_size=args.pool, config=args.config)

    output_data: Dict[str, Any] = {
        "pool_size": args.pool,
        "config": args.config or "canonical",
        "tickets_count": len(tickets),
        "tickets": tickets,
    }

    if args.verify:
        v_res = verify_covering_guarantee(
            wheel_tickets=tickets,
            pool_size=args.pool,
            t=args.t,
            m=args.m,
        )
        output_data["verification"] = {
            "t": args.t,
            "m": args.m,
            "is_guaranteed": v_res["is_guaranteed"],
            "coverage_ratio": v_res["coverage_ratio"],
            "total_subsets": v_res["total_subsets"],
            "covered_subsets": v_res["covered_subsets"],
            "uncovered_count": len(v_res["uncovered_subsets"]),
        }

    if args.json:
        print(json.dumps(output_data, indent=2))
    else:
        print(f"=== Covering Wheel: pool={args.pool}, tickets={len(tickets)} ===")
        for i, t in enumerate(tickets, 1):
            # Display as 1-indexed for human readability
            print(f"Ticket #{i:02d}: {[x + 1 for x in t]}")
        if args.verify:
            vf = output_data["verification"]
            print(f"\nVerification (t={vf['t']}, m={vf['m']}):")
            print(f"  Guaranteed: {vf['is_guaranteed']}")
            print(f"  Coverage:   {vf['covered_subsets']}/{vf['total_subsets']} ({vf['coverage_ratio']:.2%})")
            print(f"  Uncovered:  {vf['uncovered_count']}")


if __name__ == "__main__":
    main()
