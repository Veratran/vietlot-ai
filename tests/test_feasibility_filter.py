#!/usr/bin/env python3
"""
Unit Test Suite for scripts/feasibility_filter.py
─────────────────────────────────────────────────
Milestone M3: Game-Theoretic Anti-Crowd & Feasibility Filter.
Tests:
  - check_parity_ratio
  - check_gaussian_sum_band
  - check_consecutive_runs
  - is_feasible_ticket
  - All 27 edge cases (EC01–EC27) documented in Explorer 2 handoff
"""

import json
from pathlib import Path
import pytest

from scripts.feasibility_filter import (
    check_consecutive_runs,
    check_gaussian_sum_band,
    check_parity_ratio,
    is_feasible_ticket,
)


class TestCheckParityRatio:
    """Test suite for check_parity_ratio and its combinatorial edge cases."""

    @pytest.mark.parametrize(
        "ticket,expected",
        [
            ([2, 4, 6, 1, 3, 5], True),   # EC13: 3:3 (3 even, 3 odd)
            ([2, 4, 6, 8, 1, 3], True),   # EC14: 4:2 (4 even, 2 odd)
            ([2, 4, 1, 3, 5, 7], True),   # EC15: 2:4 (2 even, 4 odd)
            ([1, 3, 5, 7, 9, 11], False), # EC16: 0:6 (0 even, 6 odd)
            ([2, 4, 6, 8, 10, 12], False),# EC17: 6:0 (6 even, 0 odd)
            ([2, 1, 3, 5, 7, 9], False),  # EC18: 1:5 (1 even, 5 odd)
            ([2, 4, 6, 8, 10, 1], False), # EC19: 5:1 (5 even, 1 odd)
        ],
    )
    def test_parity_ratio_all_combinations(self, ticket, expected):
        """Verifies exact acceptance of 3:3, 4:2, 2:4 and rejection of 0:6, 6:0, 1:5, 5:1."""
        assert check_parity_ratio(ticket) is expected

    def test_parity_validation_guards(self):
        """Verifies schema validation guards (EC01–EC11)."""
        # EC01: None guard
        with pytest.raises(TypeError):
            check_parity_ratio(None)

        # EC02: primitive scalar
        with pytest.raises(TypeError):
            check_parity_ratio(123456)

        # EC03: under-length (5)
        with pytest.raises(ValueError):
            check_parity_ratio([1, 2, 3, 4, 5])

        # EC04: over-length (7)
        with pytest.raises(ValueError):
            check_parity_ratio([1, 2, 3, 4, 5, 6, 7])

        # EC05: empty container
        with pytest.raises(ValueError):
            check_parity_ratio([])

        # EC06: duplicate numbers
        with pytest.raises(ValueError):
            check_parity_ratio([1, 1, 2, 3, 4, 5])

        # EC07: string element
        with pytest.raises(TypeError):
            check_parity_ratio([1, 2, 3, 4, 5, "6"])

        # EC08: float element
        with pytest.raises(TypeError):
            check_parity_ratio([1, 2, 3, 4, 5, 6.0])

        # EC09: boolean element (subclass of int in Python)
        with pytest.raises(TypeError):
            check_parity_ratio([1, 2, 3, 4, 5, True])

        # EC10: number underflow (< 1)
        with pytest.raises(ValueError):
            check_parity_ratio([0, 2, 3, 4, 5, 6])

        # EC11: number overflow (> 55)
        with pytest.raises(ValueError):
            check_parity_ratio([1, 2, 3, 4, 5, 56])

        # EC12: number overflow for Mega 6/45 (> 45)
        with pytest.raises(ValueError):
            check_parity_ratio([1, 2, 3, 4, 5, 46], max_number=45)


class TestCheckGaussianSumBand:
    """Test suite for check_gaussian_sum_band."""

    @pytest.mark.parametrize(
        "ticket,game,expected",
        [
            # Power 6/55: band [130, 210]
            ([10, 15, 20, 25, 28, 32], "power655", True),  # EC20: sum = 130 (exact min)
            ([25, 30, 35, 38, 40, 42], "power655", True),  # EC21: sum = 210 (exact max)
            ([10, 20, 25, 30, 38, 45], "power655", True),  # sum = 168 (mean)
            ([10, 15, 20, 25, 28, 31], "power655", False), # EC22: sum = 129 (< 130)
            ([25, 30, 35, 38, 40, 43], "power655", False), # EC23: sum = 211 (> 210)
            ([1, 2, 3, 4, 5, 6], "power655", False),       # sum = 21 (extreme min)
            ([50, 51, 52, 53, 54, 55], "power655", False), # sum = 315 (extreme max)

            # Mega 6/45: band [105, 175]
            ([10, 12, 15, 20, 22, 26], "mega645", True),   # EC24: sum = 105 (exact min)
            ([20, 25, 30, 31, 34, 35], "mega645", True),   # EC25: sum = 175 (exact max)
            ([12, 18, 22, 25, 29, 32], "mega645", True),   # sum = 138 (mean)
            ([10, 12, 15, 20, 22, 25], "mega645", False),  # sum = 104 (< 105)
            ([20, 25, 30, 31, 34, 36], "mega645", False),  # sum = 176 (> 175)
        ],
    )
    def test_gaussian_sum_band_boundaries(self, ticket, game, expected):
        """Verifies boundary inclusive sums and off-by-one rejections."""
        assert check_gaussian_sum_band(ticket, game=game) is expected

    def test_gaussian_sum_band_aliases_and_case(self):
        """Verifies game identifier normalization and aliases."""
        ticket = [10, 20, 25, 30, 38, 45]  # sum = 168
        assert check_gaussian_sum_band(ticket, game="Power655") is True
        assert check_gaussian_sum_band(ticket, game="POWER655") is True
        assert check_gaussian_sum_band(ticket, game="power_655") is True
        assert check_gaussian_sum_band(ticket, game="655") is True

        mega_ticket = [12, 18, 22, 25, 29, 32]  # sum = 138
        assert check_gaussian_sum_band(mega_ticket, game="Mega645") is True
        assert check_gaussian_sum_band(mega_ticket, game="power645") is True
        assert check_gaussian_sum_band(mega_ticket, game="POWER_645") is True
        assert check_gaussian_sum_band(mega_ticket, game="645") is True

    def test_gaussian_sum_band_overflow_mega(self):
        """EC12: Ball > 45 in Mega 6/45 raises ValueError."""
        with pytest.raises(ValueError):
            check_gaussian_sum_band([1, 2, 3, 4, 5, 46], game="mega645")

    def test_gaussian_sum_band_invalid_game(self):
        """EC26: Unknown game identifier raises ValueError."""
        ticket = [10, 20, 25, 30, 38, 45]
        with pytest.raises(ValueError):
            check_gaussian_sum_band(ticket, game="invalid_game")
        with pytest.raises(ValueError):
            check_gaussian_sum_band(ticket, game="keno")
        with pytest.raises(ValueError):
            check_gaussian_sum_band(ticket, game=123)


class TestCheckConsecutiveRuns:
    """Test suite for check_consecutive_runs."""

    @pytest.mark.parametrize(
        "ticket,expected",
        [
            ([5, 12, 19, 28, 37, 46], True),   # 0 pairs
            ([14, 15, 27, 34, 42, 51], True),  # 1 isolated pair (14, 15)
            ([14, 15, 16, 30, 40, 50], False), # run of 3 (14, 15, 16)
            ([14, 15, 16, 17, 30, 40], False), # run of 4
            ([14, 15, 16, 17, 18, 40], False), # run of 5
            ([14, 15, 16, 17, 18, 19], False), # run of 6
            ([10, 11, 24, 25, 38, 50], False), # 2 disjoint pairs
            ([10, 11, 20, 21, 30, 31], False), # 3 disjoint pairs
            ([51, 15, 42, 14, 34, 27], True),  # EC27: unsorted ticket with 1 pair (14, 15)
            ([50, 16, 40, 14, 30, 15], False), # unsorted ticket with run of 3 (14, 15, 16)
        ],
    )
    def test_consecutive_runs_patterns(self, ticket, expected):
        """Verifies acceptance of <= 1 pair, rejection of runs >= 3 and multiple pairs."""
        assert check_consecutive_runs(ticket) is expected

    def test_consecutive_runs_validation(self):
        """Input validation on consecutive runs."""
        with pytest.raises(ValueError):
            check_consecutive_runs([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            check_consecutive_runs([1, 1, 3, 4, 5, 6])


class TestIsFeasibleTicket:
    """Test suite for combined 3-layer validator is_feasible_ticket."""

    def test_is_feasible_ticket_all_pass(self):
        """Ticket passing parity, gaussian sum band, and consecutive runs -> True."""
        # 3:3 parity, sum 157 in [130, 210], 0 pairs
        assert is_feasible_ticket([10, 15, 22, 27, 38, 45], game="power655") is True

    def test_is_feasible_ticket_with_one_pair(self):
        """Ticket passing all 3 with 1 isolated pair -> True."""
        # 4:2 parity, sum 174 in [130, 210], 1 pair (14, 15)
        assert is_feasible_ticket([14, 15, 24, 33, 40, 48], game="power655") is True

    def test_is_feasible_ticket_fails_only_parity(self):
        """Ticket failing only parity filter -> False."""
        # 6:0 parity (fail), 0 pairs (pass), sum 162 in [130, 210] (pass)
        assert is_feasible_ticket([12, 18, 24, 30, 36, 42], game="power655") is False

    def test_is_feasible_ticket_fails_only_consecutive(self):
        """Ticket failing only consecutive runs filter -> False."""
        # 3:3 parity (pass), run of 3 (fail), sum 167 in [130, 210] (pass)
        assert is_feasible_ticket([20, 21, 22, 27, 34, 43], game="power655") is False

    def test_is_feasible_ticket_fails_only_sum(self):
        """Ticket failing only gaussian sum band filter -> False."""
        # 3:3 parity (pass), 0 pairs (pass), sum 57 < 130 (fail)
        assert is_feasible_ticket([2, 5, 8, 11, 14, 17], game="power655") is False

    def test_is_feasible_ticket_fails_all(self):
        """Ticket failing all 3 filters -> False."""
        # 5:1 parity (fail), 2 pairs (fail), sum 37 < 130 (fail)
        assert is_feasible_ticket([2, 4, 6, 7, 8, 10], game="power655") is False

    def test_is_feasible_ticket_real_historical_data(self):
        """Tests feasibility on sample of real historical Power 6/55 draws."""
        data_file = Path("D:/vietlot-ai/data/power655.jsonl")
        if not data_file.exists():
            pytest.skip("data/power655.jsonl not found")

        draws: list = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    draws.append(json.loads(line))
                if len(draws) >= 50:
                    break

        evaluated = 0
        feasible_count = 0
        for d in draws:
            res = d.get("result", [])[:6]
            if len(res) == 6:
                evaluated += 1
                if is_feasible_ticket(res, game="power655"):
                    feasible_count += 1

        # Real historical pass rate is empirically ~55%
        assert evaluated == 50
        assert feasible_count > 15
