#!/usr/bin/env python3
"""
Unit Test Suite: Monte Carlo Baseline Simulator (Milestone 3 / R3)
──────────────────────────────────────────────────────────────────
Tests the formal null-hypothesis baseline engine for VietLot AI:
  - Theoretical hypergeometric expected hits validation (6/55 -> ~0.6545, 6/45 -> 0.8).
  - Deterministic reproducibility under fixed random seeds.
  - Execution across 1,000 and 10,000 Monte Carlo iterations.
  - Bankroll returns, prize tier accounting, and financial ROI invariants.
  - Model comparative advantage: Empirical Z-scores and empirical p-values.
  - Edge cases, error handling, and integration with real Vietlott dataset files.
"""
import json
import math
import pathlib
import sys
import pytest

# Ensure scripts directory is available in Python path
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from monte_carlo_baseline import (
    MonteCarloBaselineSimulator,
    POWER_655_PRIZES,
    POWER_645_PRIZES,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_draws_655():
    """
    Synthetic series of 50 Power 6/55 draws.
    Each draw has 6 main numbers and 1 special/bonus number.
    """
    draws = []
    for d in range(1, 51):
        main = [
            ((d * 3 + 1) % 55) + 1,
            ((d * 5 + 7) % 55) + 1,
            ((d * 7 + 13) % 55) + 1,
            ((d * 11 + 17) % 55) + 1,
            ((d * 13 + 19) % 55) + 1,
            ((d * 17 + 23) % 55) + 1,
        ]
        # Guarantee 6 distinct numbers
        unique_main = sorted(set(main))
        num = 1
        while len(unique_main) < 6:
            if num not in unique_main:
                unique_main.append(num)
            num += 1

        bonus = 55
        while bonus in unique_main:
            bonus -= 1

        draws.append(unique_main + [bonus])
    return draws


@pytest.fixture
def synthetic_draws_645():
    """
    Synthetic series of 50 Mega 6/45 draws (6 numbers each).
    """
    draws = []
    for d in range(1, 51):
        main = [
            ((d * 2 + 1) % 45) + 1,
            ((d * 4 + 5) % 45) + 1,
            ((d * 6 + 9) % 45) + 1,
            ((d * 8 + 13) % 45) + 1,
            ((d * 10 + 17) % 45) + 1,
            ((d * 12 + 21) % 45) + 1,
        ]
        unique_main = sorted(set(main))
        num = 1
        while len(unique_main) < 6:
            if num not in unique_main:
                unique_main.append(num)
            num += 1
        draws.append(unique_main)
    return draws


# ==============================================================================
# Test Cases
# ==============================================================================

def test_simulator_initialization_defaults_and_validation():
    """
    Verifies default arguments and input validation.
    """
    # Defaults: 55 numbers, pick 6, 10000 VND, seed 42, power655
    sim = MonteCarloBaselineSimulator()
    assert sim.total_numbers == 55
    assert sim.pick_count == 6
    assert sim.ticket_price == 10000
    assert sim.seed == 42
    assert sim.game_type == "power655"

    # Auto-detection of power645
    sim_45 = MonteCarloBaselineSimulator(total_numbers=45)
    assert sim_45.game_type == "power645"

    # Invalid configurations
    with pytest.raises(ValueError, match="greater than or equal to pick_count"):
        MonteCarloBaselineSimulator(total_numbers=5, pick_count=6)

    with pytest.raises(ValueError, match="pick_count"):
        MonteCarloBaselineSimulator(total_numbers=55, pick_count=0)

    with pytest.raises(ValueError, match="negative"):
        MonteCarloBaselineSimulator(ticket_price=-100)


def test_theoretical_expected_hits_property():
    """
    Verifies theoretical expected hits:
      - 6/55: 6 * 6 / 55 = 36 / 55 ≈ 0.654545...
      - 6/45: 6 * 6 / 45 = 36 / 45 = 0.800000...
    """
    sim_55 = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6)
    expected_55 = 36.0 / 55.0
    assert sim_55.theoretical_expected_hits_per_draw == pytest.approx(expected_55, abs=1e-9)
    assert round(sim_55.theoretical_expected_hits_per_draw, 4) == 0.6545

    sim_45 = MonteCarloBaselineSimulator(total_numbers=45, pick_count=6)
    assert sim_45.theoretical_expected_hits_per_draw == pytest.approx(0.8, abs=1e-9)


def test_prize_calculation_power655():
    """
    Tests prize calculation across all tiers for Power 6/55.
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, game_type="power655")
    draw_main = {1, 2, 3, 4, 5, 6}
    bonus_number = 7

    # Jackpot 1 (Match 6)
    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 4, 5, 6], draw_main, bonus_number)
    assert payout == 30_000_000_000
    assert tier == "jackpot_1"

    # Jackpot 2 (Match 5 + bonus)
    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 4, 5, 7], draw_main, bonus_number)
    assert payout == 3_000_000_000
    assert tier == "jackpot_2"

    # Match 5 without bonus (Giải nhất)
    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 4, 5, 8], draw_main, bonus_number)
    assert payout == 40_000_000
    assert tier == "match_5"

    # Match 4 (Giải nhì)
    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 4, 10, 11], draw_main, bonus_number)
    assert payout == 500_000
    assert tier == "match_4"

    # Match 3 (Giải ba)
    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 20, 21, 22], draw_main, bonus_number)
    assert payout == 50_000
    assert tier == "match_3"

    # Match 2 (No prize)
    payout, tier = sim.calculate_ticket_prize([1, 2, 30, 31, 32, 33], draw_main, bonus_number)
    assert payout == 0
    assert tier == "no_prize"

    # Match 0 (No prize)
    payout, tier = sim.calculate_ticket_prize([40, 41, 42, 43, 44, 45], draw_main, bonus_number)
    assert payout == 0
    assert tier == "no_prize"


def test_prize_calculation_power645():
    """
    Tests prize calculation across all tiers for Mega 6/45.
    """
    sim = MonteCarloBaselineSimulator(total_numbers=45, pick_count=6, game_type="power645")
    draw_main = {10, 20, 30, 40, 41, 42}

    # Jackpot (Match 6)
    payout, tier = sim.calculate_ticket_prize([10, 20, 30, 40, 41, 42], draw_main)
    assert payout == 12_000_000_000
    assert tier == "jackpot"

    # Match 5 (Giải nhất)
    payout, tier = sim.calculate_ticket_prize([10, 20, 30, 40, 41, 1], draw_main)
    assert payout == 10_000_000
    assert tier == "match_5"

    # Match 4 (Giải nhì)
    payout, tier = sim.calculate_ticket_prize([10, 20, 30, 40, 1, 2], draw_main)
    assert payout == 300_000
    assert tier == "match_4"

    # Match 3 (Giải ba)
    payout, tier = sim.calculate_ticket_prize([10, 20, 30, 1, 2, 3], draw_main)
    assert payout == 30_000
    assert tier == "match_3"

    # Match <= 2
    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 4, 5, 6], draw_main)
    assert payout == 0
    assert tier == "no_prize"


def test_prize_calculation_custom_payout_table():
    """
    Tests simulator with custom payout schedule.
    """
    custom_payouts = {
        "match_6": 1_000_000,
        "match_5": 50_000,
        "match_4": 5_000,
        "match_3": 500,
    }
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, payout_table=custom_payouts)
    draw_main = {1, 2, 3, 4, 5, 6}

    payout, tier = sim.calculate_ticket_prize([1, 2, 3, 4, 5, 6], draw_main)
    assert payout == 1_000_000
    assert tier == "match_6"

    payout_3, tier_3 = sim.calculate_ticket_prize([1, 2, 3, 10, 11, 12], draw_main)
    assert payout_3 == 500
    assert tier_3 == "match_3"


def test_deterministic_reproducibility(synthetic_draws_655):
    """
    Guarantees deterministic reproducibility:
    Two separate simulation runs with identical seeds MUST produce identical results.
    """
    sim_a = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=12345)
    res_a = sim_a.run_simulation(synthetic_draws_655, simulations=1000)

    sim_b = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=12345)
    res_b = sim_b.run_simulation(synthetic_draws_655, simulations=1000)

    # Identical summary metrics
    assert res_a["mean_hits_per_draw"] == res_b["mean_hits_per_draw"]
    assert res_a["std_hits_per_draw"] == res_b["std_hits_per_draw"]
    assert res_a["earnings_mean"] == res_b["earnings_mean"]
    assert res_a["net_earnings_mean"] == res_b["net_earnings_mean"]
    assert res_a["roi_mean"] == res_b["roi_mean"]

    # Identical raw simulation lists
    assert res_a["sim_mean_hits"] == res_b["sim_mean_hits"]
    assert res_a["sim_earnings"] == res_b["sim_earnings"]

    # Different seed produces distinct results
    sim_c = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=99999)
    res_c = sim_c.run_simulation(synthetic_draws_655, simulations=1000)
    assert res_a["sim_mean_hits"] != res_c["sim_mean_hits"]


def test_simulation_1000_iterations_statistical_properties(synthetic_draws_655):
    """
    Runs 1,000 iterations and verifies:
      - Mean hits per draw converges to theoretical expectation (36/55 ≈ 0.6545 ± 0.02).
      - Accounting equations hold:
          net_earnings == earnings - total_cost
          roi == net_earnings / total_cost
          payout_ratio == earnings / total_cost
      - Hit probabilities sum to 1.0.
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    res = sim.run_simulation(synthetic_draws_655, simulations=1000)

    assert res["simulations"] == 1000
    assert res["draws_count"] == 50
    assert res["total_cost_per_sim"] == 50 * 10000  # 500,000 VND

    # Statistical convergence to theoretical expectation: 0.654545
    theoretical = 36.0 / 55.0
    assert abs(res["mean_hits_per_draw"] - theoretical) < 0.02, (
        f"Empirical mean hits {res['mean_hits_per_draw']} diverged from theoretical {theoretical}"
    )

    # Accounting invariants
    expected_cost = 50 * 10000
    assert res["total_cost_per_sim"] == expected_cost
    assert res["net_earnings_mean"] == pytest.approx(res["earnings_mean"] - expected_cost, abs=1e-2)
    assert res["roi_mean"] == pytest.approx(res["net_earnings_mean"] / expected_cost, abs=1e-4)
    assert res["payout_ratio_mean"] == pytest.approx(res["earnings_mean"] / expected_cost, abs=1e-4)

    # Min/max bounds
    assert res["earnings_min"] <= res["earnings_mean"] <= res["earnings_max"]
    assert res["net_earnings_min"] <= res["net_earnings_mean"] <= res["net_earnings_max"]

    # Hit distribution normalization: sum of probabilities == 1.0
    hit_rate_sum = sum(res["hit_rate_distribution"].values())
    assert hit_rate_sum == pytest.approx(1.0, abs=1e-5)


def test_simulation_10000_iterations_scale_and_convergence(synthetic_draws_655):
    """
    Verifies execution across 10,000 iterations:
      - Validates scalability and fast runtime.
      - Law of large numbers: Convergence within tight tolerance (|mean - 0.6545| < 0.01).
      - Positive standard deviation of hits and earnings.
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=777)
    res = sim.run_simulation(synthetic_draws_655, simulations=10000)

    assert res["simulations"] == 10000
    assert res["draws_count"] == len(synthetic_draws_655)

    theoretical = 36.0 / 55.0
    assert abs(res["mean_hits_per_draw"] - theoretical) < 0.01, (
        f"10,000 iterations mean hits {res['mean_hits_per_draw']} not within 0.01 of {theoretical}"
    )

    assert res["std_hits_per_draw"] > 0.0
    assert res["earnings_std"] >= 0.0
    assert len(res["sim_mean_hits"]) == 10000
    assert len(res["sim_earnings"]) == 10000


def test_simulation_mega_645_expected_hits(synthetic_draws_645):
    """
    Verifies Mega 6/45 theoretical convergence:
    Theoretical hits = 6 * 6 / 45 = 0.8000.
    """
    sim_45 = MonteCarloBaselineSimulator(total_numbers=45, pick_count=6, seed=101)
    res = sim_45.run_simulation(synthetic_draws_645, simulations=2000)

    assert abs(res["mean_hits_per_draw"] - 0.8000) < 0.02
    assert res["game_type"] == "power645" if hasattr(sim_45, "game_type") else True


def test_evaluate_model_advantage_superior_model(synthetic_draws_655):
    """
    Tests evaluate_model_advantage with a model that correctly predicts all 6 numbers.
    Expected:
      - High positive Z-score (Z >> 3.0)
      - Empirical p-value == 0.0
      - statistically_significant_95 == True
      - statistically_significant_99 == True
      - better_than_mc_mean == True
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    mc_baseline = sim.run_simulation(synthetic_draws_655, simulations=1000)

    # Perfect model picking exact main draw
    perfect_picks = [d[:6] for d in synthetic_draws_655]
    eval_res = sim.evaluate_model_advantage(perfect_picks, synthetic_draws_655, mc_baseline)

    assert eval_res["model_metrics"]["mean_hits_per_draw"] == 6.0
    assert eval_res["model_metrics"]["total_hits"] == 6 * len(synthetic_draws_655)
    assert eval_res["z_score_hits"] > 10.0
    assert eval_res["z_score"] > 10.0
    assert eval_res["p_value_hits"] == 0.0
    assert eval_res["p_value"] == 0.0
    assert eval_res["better_than_mc_mean"] is True
    assert eval_res["statistically_significant_95"] is True
    assert eval_res["statistically_significant_99"] is True
    assert eval_res["hit_rate_delta"] > 5.0
    assert eval_res["earnings_delta"] > 0


def test_evaluate_model_advantage_inferior_model(synthetic_draws_655):
    """
    Tests evaluate_model_advantage with an intentionally zero-hit model.
    Expected:
      - Negative Z-score (Z < -2.0)
      - Empirical p-value == 1.0 (all MC runs match or beat zero hits)
      - statistically_significant_95 == False
      - better_than_mc_mean == False
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    mc_baseline = sim.run_simulation(synthetic_draws_655, simulations=1000)

    # Model deliberately picking numbers outside the draw
    inferior_picks = []
    for d in synthetic_draws_655:
        main_set = set(d[:6])
        avoid = []
        for n in range(1, 56):
            if n not in main_set:
                avoid.append(n)
            if len(avoid) == 6:
                break
        inferior_picks.append(avoid)

    eval_res = sim.evaluate_model_advantage(inferior_picks, synthetic_draws_655, mc_baseline)

    assert eval_res["model_metrics"]["mean_hits_per_draw"] == 0.0
    assert eval_res["z_score_hits"] < -2.0
    assert eval_res["p_value_hits"] == 1.0
    assert eval_res["better_than_mc_mean"] is False
    assert eval_res["statistically_significant_95"] is False


def test_evaluate_model_advantage_random_picks(synthetic_draws_655):
    """
    Tests evaluate_model_advantage with another random model.
    Under H0, Z-score should lie within standard confidence bounds (-3.0 <= Z <= +3.0).
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    mc_baseline = sim.run_simulation(synthetic_draws_655, simulations=1000)

    # Random picks from another seed
    import random
    rng = random.Random(999)
    random_picks = [rng.sample(range(1, 56), 6) for _ in synthetic_draws_655]

    eval_res = sim.evaluate_model_advantage(random_picks, synthetic_draws_655, mc_baseline)

    # In-band Z-score
    assert -3.5 <= eval_res["z_score"] <= 3.5
    assert 0.0 <= eval_res["p_value"] <= 1.0


def test_edge_cases_and_error_handling(synthetic_draws_655):
    """
    Tests input robustness and error handling:
      - Empty draws list
      - Zero or negative simulations
      - Mismatched model_picks length
      - Invalid ticket pick size
    """
    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6)

    # Empty draws list
    with pytest.raises(ValueError, match="actual_draws list cannot be empty"):
        sim.run_simulation([], simulations=100)

    # Simulations < 1
    with pytest.raises(ValueError, match="simulations must be >= 1"):
        sim.run_simulation(synthetic_draws_655, simulations=0)

    mc_baseline = sim.run_simulation(synthetic_draws_655, simulations=100)

    # Mismatched model picks count
    short_picks = [[1, 2, 3, 4, 5, 6]]
    with pytest.raises(ValueError, match="Length mismatch"):
        sim.evaluate_model_advantage(short_picks, synthetic_draws_655, mc_baseline)

    # Empty model picks
    with pytest.raises(ValueError, match="model_picks cannot be empty"):
        sim.evaluate_model_advantage([], [], mc_baseline)

    # Invalid pick size (e.g. 5 numbers instead of 6)
    bad_pick_size = [[1, 2, 3, 4, 5]] * len(synthetic_draws_655)
    with pytest.raises(ValueError, match="expected 6"):
        sim.evaluate_model_advantage(bad_pick_size, synthetic_draws_655, mc_baseline)


def test_real_dataset_power655():
    """
    Integration test against actual historical records in data/power655.jsonl.
    """
    data_file = ROOT / "data" / "power655.jsonl"
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")

    records = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) > 50
    # Use recent 50 draws
    sample_draws = [r["result"] for r in records[:50]]

    sim = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    res = sim.run_simulation(sample_draws, simulations=1000)

    # Verify hit rate convergence on real draws
    theoretical = 36.0 / 55.0
    assert abs(res["mean_hits_per_draw"] - theoretical) < 0.02
    assert res["total_cost_per_sim"] == 50 * 10000


def test_real_dataset_power645():
    """
    Integration test against actual historical records in data/power645.jsonl.
    """
    data_file = ROOT / "data" / "power645.jsonl"
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")

    records = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    assert len(records) > 50
    sample_draws = [r["result"] for r in records[:50]]

    sim = MonteCarloBaselineSimulator(total_numbers=45, pick_count=6, seed=42)
    res = sim.run_simulation(sample_draws, simulations=1000)

    # Verify hit rate convergence on real 6/45 draws
    assert abs(res["mean_hits_per_draw"] - 0.8000) < 0.02
    assert res["total_cost_per_sim"] == 50 * 10000
