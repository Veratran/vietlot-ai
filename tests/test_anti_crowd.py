#!/usr/bin/env python3
"""
Unit Test Suite for scripts/anti_crowd.py
────────────────────────────────────────
Milestone M3: Game-Theoretic Anti-Crowd & Feasibility Filter.
Tests:
  - estimate_selection_density
  - calculate_crowd_penalty
  - pareto_filter
"""

import math
from typing import Any, Dict, List
import numpy as np
import pytest

from scripts.anti_crowd import (
    calculate_crowd_penalty,
    estimate_selection_density,
    pareto_filter,
)


class TestEstimateSelectionDensity:
    """Test suite for estimate_selection_density."""

    def test_estimate_selection_density_fallback_sum_to_one(self):
        """Empty input falls back to calibrated prior with sum == 1.0 +- 1e-6."""
        density = estimate_selection_density([], total_numbers=55)
        assert len(density) == 55
        assert math.isclose(sum(density.values()), 1.0, rel_tol=1e-6)
        # All densities must be strictly positive
        assert all(v > 0 for v in density.values())

    def test_estimate_selection_density_fallback_calendar_bias(self):
        """Fallback calibrated prior embodies the +123% calendar bias."""
        density = estimate_selection_density([], total_numbers=55)
        mean_cal = np.mean([density[n] for n in range(1, 32)])
        mean_non_cal = np.mean([density[n] for n in range(32, 56)])
        ratio = mean_cal / mean_non_cal
        # ratio should reflect rho_cal ~ 1.222
        assert 1.15 <= ratio <= 1.30

    def test_estimate_selection_density_with_winners(self):
        """Draws with winner data scale selection density according to winner counts."""
        draws = [
            {
                "result": [1, 5, 10, 15, 20, 25],
                "winners": {"Giải nhất": 25, "Giải nhì": 1500, "Giải ba": 28000},
            },
            {
                "result": [35, 40, 45, 50, 52, 55],
                "winners": {"Giải nhất": 11, "Giải nhì": 800, "Giải ba": 18000},
            },
        ]
        density = estimate_selection_density(draws, total_numbers=55)
        assert len(density) == 55
        assert math.isclose(sum(density.values()), 1.0, rel_tol=1e-6)
        # Number 1 (high-winner draw) must receive higher density than number 35
        assert density[1] > density[35]

    def test_estimate_selection_density_empirical_bias(self):
        """Empirical draws with winner statistics reflect strong calendar preference."""
        draws = [
            {"result": [2, 5, 12, 19, 25, 31], "winners": {"Giải ba": 28782, "Giải nhất": 25}},
            {"result": [33, 38, 42, 47, 51, 55], "winners": {"Giải ba": 18360, "Giải nhất": 11}},
        ]
        density = estimate_selection_density(draws, total_numbers=55)
        cal_density = sum(density[n] for n in range(1, 32)) / 31
        non_cal_density = sum(density[n] for n in range(32, 56)) / 24
        assert cal_density > non_cal_density

    def test_estimate_selection_density_without_winner_metadata(self):
        """Sequence of raw ticket integer lists falls back cleanly to calibrated prior."""
        draws = [
            [1, 2, 3, 4, 5, 6],
            [10, 20, 30, 40, 50, 55],
            [7, 14, 21, 28, 35, 42],
        ]
        density = estimate_selection_density(draws, total_numbers=55)
        assert len(density) == 55
        assert math.isclose(sum(density.values()), 1.0, rel_tol=1e-6)
        # Verify calibrated prior values
        mean_cal = np.mean([density[n] for n in range(1, 32)])
        mean_non_cal = np.mean([density[n] for n in range(32, 56)])
        assert mean_cal > mean_non_cal

    def test_estimate_selection_density_custom_pool(self):
        """Works cleanly for Mega 6/45 with total_numbers=45."""
        density = estimate_selection_density([], total_numbers=45)
        assert len(density) == 45
        assert min(density.keys()) == 1
        assert max(density.keys()) == 45
        assert math.isclose(sum(density.values()), 1.0, rel_tol=1e-6)

    def test_estimate_selection_density_invalid_total_numbers(self):
        """Zero or negative total_numbers raises ValueError."""
        with pytest.raises(ValueError):
            estimate_selection_density([], total_numbers=0)
        with pytest.raises(ValueError):
            estimate_selection_density([], total_numbers=-10)


class TestCalculateCrowdPenalty:
    """Test suite for calculate_crowd_penalty."""

    @pytest.mark.parametrize(
        "c_cal,ticket,expected_penalty",
        [
            (0, [32, 35, 40, 45, 50, 55], 0.0),
            (1, [10, 35, 40, 45, 50, 55], 0.0),
            (2, [10, 20, 35, 40, 45, 50], 0.0),
            (3, [5, 12, 28, 35, 42, 50], 15.0),
            (4, [3, 11, 21, 31, 44, 52], 30.0),
            (5, [1, 7, 15, 23, 30, 48], 70.0),
            (6, [2, 9, 14, 19, 25, 31], 100.0),
        ],
    )
    def test_birthday_penalty_scaling(self, c_cal, ticket, expected_penalty):
        """Verifies progressive birthday penalty curve [0, 0, 0, 15, 30, 70, 100]."""
        pen = calculate_crowd_penalty(ticket)
        assert pen["birthday_penalty"] == pytest.approx(expected_penalty)

    @pytest.mark.parametrize(
        "ticket,expected_lucky_pen",
        [
            ([1, 2, 3, 4, 5, 7], 0.0),  # 0 lucky
            ([1, 2, 3, 4, 5, 6], 5.0),  # 1 lucky (6)
            ([1, 2, 3, 4, 8, 9], 10.0),  # 2 lucky (8, 9)
            ([6, 8, 9, 11, 21, 31], 15.0),  # 3 lucky (6, 8, 9)
            ([6, 16, 26, 36, 46, 5], 25.0),  # 5 lucky (6, 16, 26, 36, 46)
            ([6, 16, 26, 38, 48, 49], 30.0),  # 6 lucky
        ],
    )
    def test_lucky_digits_penalty(self, ticket, expected_lucky_pen):
        """Verifies 5.0 points per lucky ending digit (6, 8, 9)."""
        pen = calculate_crowd_penalty(ticket)
        assert pen["lucky_digit_penalty"] == pytest.approx(expected_lucky_pen)

    def test_arithmetic_progression_penalty(self):
        """Verifies full AP (100.0), partial AP (40.0), and non-AP (0.0)."""
        # Full AP step 1 -> 100.0
        assert calculate_crowd_penalty([1, 2, 3, 4, 5, 6])["ap_penalty"] == 100.0
        # Full AP step 5 -> 100.0
        assert calculate_crowd_penalty([5, 10, 15, 20, 25, 30])["ap_penalty"] == 100.0
        # Full AP step 7 -> 100.0
        assert calculate_crowd_penalty([7, 14, 21, 28, 35, 42])["ap_penalty"] == 100.0
        # Partial AP (first 4 differences equal) -> 40.0
        assert calculate_crowd_penalty([2, 4, 6, 8, 10, 25])["ap_penalty"] == 40.0
        # Partial AP (last 4 differences equal) -> 40.0
        assert calculate_crowd_penalty([1, 10, 15, 20, 25, 30])["ap_penalty"] == 40.0
        # No AP -> 0.0
        assert calculate_crowd_penalty([3, 7, 12, 19, 31, 44])["ap_penalty"] == 0.0

    def test_grid_geometry_penalty(self):
        """Verifies column alignment (25.0), diagonal sweep (20.0), combined (45.0), scattered (0.0)."""
        # Column alignment: 4 numbers in column 0 (1, 6, 11, 16)
        pen_col = calculate_crowd_penalty([1, 6, 11, 16, 22, 33])
        assert pen_col["geometry_penalty"] >= 25.0

        # Monotonic diagonal: 1, 7, 13, 19 (step 6, 4 balls)
        pen_diag = calculate_crowd_penalty([1, 7, 13, 19, 35, 42])
        assert pen_diag["geometry_penalty"] >= 20.0

        # Combined geometry: column + diagonal
        # e.g. 4 in col 0 (1, 6, 11, 16) plus diagonal elements (1, 7, 13, 19)
        # Ticket: [1, 7, 13, 19, 25, 31] is both full AP step 6 and diagonal sweep
        pen_diag_full = calculate_crowd_penalty([1, 7, 13, 19, 25, 31])
        assert pen_diag_full["geometry_penalty"] >= 20.0

        # Clean scattered ticket: no column clustering, no diagonal sweep
        pen_clean = calculate_crowd_penalty([2, 9, 17, 24, 38, 51])
        assert pen_clean["geometry_penalty"] == 0.0

    def test_crowd_penalty_schema_and_sum(self):
        """Schema has exactly 5 keys, and total_penalty equals sum of subcomponents."""
        ticket = [6, 16, 26, 36, 46, 55]
        res = calculate_crowd_penalty(ticket)
        expected_keys = {
            "total_penalty",
            "birthday_penalty",
            "lucky_digit_penalty",
            "ap_penalty",
            "geometry_penalty",
        }
        assert set(res.keys()) == expected_keys
        for k in expected_keys:
            assert isinstance(res[k], float)
        assert res["total_penalty"] == pytest.approx(
            res["birthday_penalty"]
            + res["lucky_digit_penalty"]
            + res["ap_penalty"]
            + res["geometry_penalty"]
        )

    def test_unsorted_ticket_invariance(self):
        """Permutation of ticket numbers produces identical penalty breakdown."""
        t1 = [30, 10, 25, 5, 20, 15]
        t2 = [5, 10, 15, 20, 25, 30]
        assert calculate_crowd_penalty(t1) == calculate_crowd_penalty(t2)

    def test_calculate_crowd_penalty_validation(self):
        """Comprehensive input validation: length, uniqueness, bounds, types."""
        # Length: hàm đã tổng quát cho mọi độ dài vé (5/35, 10/80...)
        # — vé 5 và 7 số hợp lệ, chỉ vé rỗng mới lỗi.
        assert calculate_crowd_penalty([1, 2, 3, 4, 5])["total_penalty"] >= 0
        assert calculate_crowd_penalty([1, 2, 3, 4, 5, 6, 7])["total_penalty"] >= 0
        with pytest.raises(ValueError):
            calculate_crowd_penalty([])

        # Duplicate elements
        with pytest.raises(ValueError):
            calculate_crowd_penalty([1, 2, 3, 4, 5, 5])

        # Bounds errors
        with pytest.raises(ValueError):
            calculate_crowd_penalty([0, 2, 3, 4, 5, 6])
        with pytest.raises(ValueError):
            calculate_crowd_penalty([1, 2, 3, 4, 5, 56])

        # Type errors
        with pytest.raises(TypeError):
            calculate_crowd_penalty(None)
        with pytest.raises(TypeError):
            calculate_crowd_penalty([1, 2, 3, 4, 5, "6"])
        with pytest.raises(TypeError):
            calculate_crowd_penalty([1, 2, 3, 4, 5, 6.0])
        with pytest.raises(TypeError):
            calculate_crowd_penalty([1, 2, 3, 4, 5, True])


class TestParetoFilter:
    """Test suite for pareto_filter."""

    def test_pareto_filter_single_dominant_candidate(self):
        """Candidate A strictly dominates Candidate B -> only Candidate A retained."""
        candidates = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 55]]
        scores = [0.9, 0.4]
        penalties = [10.0, 50.0]
        front = pareto_filter(candidates, scores, penalties)
        assert len(front) == 1
        assert front[0]["ticket"] == [1, 2, 3, 4, 5, 6]
        assert front[0]["rank"] == 1
        assert front[0]["pareto_rank"] == 1
        assert front[0]["original_index"] == 0

    def test_pareto_filter_trade_off(self):
        """Candidate A (higher score, higher penalty) vs Candidate B (lower score, lower penalty) -> both retained."""
        candidates = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 55]]
        scores = [0.9, 0.5]
        penalties = [60.0, 10.0]
        front = pareto_filter(candidates, scores, penalties)
        assert len(front) == 2
        # Deterministic sorting: higher score first
        assert front[0]["ticket"] == [1, 2, 3, 4, 5, 6]
        assert front[1]["ticket"] == [10, 20, 30, 40, 50, 55]

    def test_pareto_filter_multi_candidate(self):
        """Filters dominated candidates in a 5-candidate pool."""
        candidates = [
            [1, 2, 3, 4, 5, 6],    # score=0.90, pen=40 (Front)
            [10, 20, 30, 40, 50, 55], # score=0.95, pen=60 (Front)
            [5, 10, 15, 20, 25, 30],  # score=0.50, pen=10 (Front)
            [7, 14, 21, 28, 35, 42],  # score=0.40, pen=50 (Dominated by #1, #3)
            [2, 4, 6, 8, 10, 12],    # score=0.80, pen=50 (Dominated by #0)
        ]
        scores = [0.90, 0.95, 0.50, 0.40, 0.80]
        penalties = [40.0, 60.0, 10.0, 50.0, 50.0]

        front = pareto_filter(candidates, scores, penalties)
        assert len(front) == 3
        front_tickets = [f["ticket"] for f in front]
        assert [10, 20, 30, 40, 50, 55] in front_tickets
        assert [1, 2, 3, 4, 5, 6] in front_tickets
        assert [5, 10, 15, 20, 25, 30] in front_tickets
        assert [7, 14, 21, 28, 35, 42] not in front_tickets
        assert [2, 4, 6, 8, 10, 12] not in front_tickets

    def test_pareto_filter_empty_and_mismatch(self):
        """Empty input returns [], mismatched array lengths raise ValueError."""
        assert pareto_filter([], [], []) == []
        with pytest.raises(ValueError):
            pareto_filter([[1, 2, 3, 4, 5, 6]], [0.5], [10.0, 20.0])
        with pytest.raises(ValueError):
            pareto_filter([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], [0.5], [10.0])

    def test_pareto_filter_all_identical(self):
        """Identical candidates are all retained in rank-1 front."""
        candidates = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 55]]
        scores = [0.8, 0.8]
        penalties = [20.0, 20.0]
        front = pareto_filter(candidates, scores, penalties)
        assert len(front) == 2

    def test_pareto_filter_deterministic_ordering(self):
        """Tie-breaking sorts by (-score, penalty, sorted_ticket)."""
        candidates = [
            [2, 4, 6, 8, 10, 12],
            [1, 3, 5, 7, 9, 11],
        ]
        scores = [0.8, 0.8]
        penalties = [20.0, 20.0]
        front = pareto_filter(candidates, scores, penalties)
        assert len(front) == 2
        # Ticket [1, 3, 5, 7, 9, 11] comes before [2, 4, 6, 8, 10, 12] lexicographically
        assert front[0]["ticket"] == [1, 3, 5, 7, 9, 11]
        assert front[1]["ticket"] == [2, 4, 6, 8, 10, 12]

