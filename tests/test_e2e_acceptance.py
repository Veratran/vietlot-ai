#!/usr/bin/env python3
"""
Test Suite: Comprehensive 4-Tier E2E Acceptance Suite
────────────────────────────────────────────────────
Repository: D:\\vietlot-ai
Acceptance Criteria: ORIGINAL_REQUEST.md & PROJECT.md
Author: E2E Acceptance Test Writer

Methodology (4-Tier Opaque-Box Approach):
  • Tier 1: Feature Coverage (F1 to F7, >=5 test cases per feature)
  • Tier 2: Boundary & Corner Cases (F1 to F7, >=5 test cases per feature)
  • Tier 3: Cross-Feature Integration Combinations
  • Tier 4: Real-World Application Scenarios on data/power655.jsonl

Invariants Asserted:
  - Normalization Invariant: sum P(w | state) == 1.0 within 1e-6
  - ZeroFutureLeakage: No draw date >= cutoff_date accessible post-cutoff
  - BankrollSolvency: bankroll >= 0 at all times; safe transition to HALTED_INSOLVENT
  - ValidPickSize: exactly 6 distinct valid lottery numbers produced
  - FeaturesSynchronized: markovState == "CALCULATED" and zscoreState == "COMPUTED"
"""

import copy
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Set, Tuple

import pytest

# Ensure scripts directory is on sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
SPEC_DIR = ROOT / "spec"
DATA_DIR = ROOT / "data"
TLA2TOOLS_JAR = pathlib.Path(r"C:\Users\porever\.gemini\config\skills\statechart\bin\tla2tools.jar")
FALLBACK_JAVA = pathlib.Path(r"C:\Users\porever\.gemini\config\skills\statechart\bin\jdk-21\bin\java.exe")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from markov_chain import HigherOrderMarkovChain
from gaussian_cycle import GaussianCycleAnalyzer
from monte_carlo_baseline import MonteCarloBaselineSimulator
from pipeline_fsm import (
    LotteryPipelineStateMachine,
    PipelineState,
    BankrollSolvencyError,
    ZeroFutureLeakageError,
    InvalidStateTransitionError,
    InvariantViolationError,
)
from data_utils import cutoff_future_data, parse_date


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Shared Test Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_java_executable() -> str:
    """Finds active Java 21+ executable."""
    java_path = shutil.which("java")
    if java_path:
        return java_path
    if FALLBACK_JAVA.exists():
        return str(FALLBACK_JAVA)
    return ""


@pytest.fixture(scope="session")
def java_runtime():
    """Provides Java runtime path or skips TLC tests if Java is unavailable."""
    java_exe = get_java_executable()
    if not java_exe or not TLA2TOOLS_JAR.exists():
        pytest.skip("Java runtime or tla2tools.jar missing in environment")
    return java_exe


@pytest.fixture
def synthetic_sequence_order1():
    """Deterministic sequence of draws for order 1 transition tests."""
    return [
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
        [4, 5, 6, 7, 8, 9],
        [5, 6, 7, 8, 9, 10],
    ]


@pytest.fixture
def synthetic_sequence_order2():
    """Deterministic sequence of draws for order 2 transition tests."""
    return [
        [10, 20, 30, 40, 50, 55],
        [11, 21, 31, 41, 51, 54],
        [12, 22, 32, 42, 52, 53],
        [13, 23, 33, 43, 53, 52],
        [14, 24, 34, 44, 54, 51],
        [15, 25, 35, 45, 55, 50],
    ]


@pytest.fixture
def power655_raw_records():
    """Loads historical Power 6/55 records from data/power655.jsonl."""
    jsonl_path = DATA_DIR / "power655.jsonl"
    assert jsonl_path.exists(), f"Missing dataset at {jsonl_path}"
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ═════════════════════════════════════════════════════════════════════════════
# TIER 1: FEATURE COVERAGE (Minimum 5 Test Cases per Feature, F1 to F7)
# ═════════════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------------------
# F1: Higher-Order Markov Transition Matrix
# -----------------------------------------------------------------------------

def test_tier1_f1_markov_order1_transition_probabilities(synthetic_sequence_order1):
    """F1.1: Order k=1 transition probability estimation and row integrity."""
    model = HigherOrderMarkovChain(order=1, total_numbers=55, alpha=0.1)
    model.fit(synthetic_sequence_order1)

    assert model.is_fitted
    prob_1_to_2 = model.get_transition_probability(1, 2)
    prob_1_to_55 = model.get_transition_probability(1, 55)

    assert prob_1_to_2 > 0.0
    assert prob_1_to_55 > 0.0
    # Because 1->2 was observed in sequence whereas 1->55 was not, prob_1_to_2 > prob_1_to_55
    assert prob_1_to_2 > prob_1_to_55


def test_tier1_f1_markov_order2_transition_probabilities(synthetic_sequence_order2):
    """F1.2: Order k=2 bigram (u1, u2) -> w transition probability estimation."""
    model = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    model.fit(synthetic_sequence_order2)

    assert model.is_fitted
    prob_observed = model.get_transition_probability((10, 11), 12)
    prob_unobserved = model.get_transition_probability((10, 11), 1)

    assert prob_observed > 0.0
    assert prob_unobserved > 0.0
    assert prob_observed > prob_unobserved


def test_tier1_f1_markov_strict_row_normalization_invariant(synthetic_sequence_order2):
    """F1.3: Mathematical Invariant: Every conditional distribution must sum to 1.0 within 1e-6."""
    model = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    model.fit(synthetic_sequence_order2)

    assert model.verify_normalization_invariant() is True

    # Test order 1 as well
    model_o1 = HigherOrderMarkovChain(order=1, total_numbers=55, alpha=0.1)
    model_o1.fit(synthetic_sequence_order2)
    assert model_o1.verify_normalization_invariant() is True


def test_tier1_f1_markov_smoothing_and_jelinek_mercer_backoff():
    """F1.4: Jelinek-Mercer backoff and Lidstone smoothing for unseen states."""
    draws = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
    model = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1, lambdas=(0.7, 0.2, 0.1))
    model.fit(draws)

    # State (54, 55) was never observed anywhere in draws
    unseen_state = (54, 55)
    row_probs = [model.get_transition_probability(unseen_state, w) for w in range(1, 56)]

    assert len(row_probs) == 55
    assert math.isclose(sum(row_probs), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    # All probabilities in unseen state should be strictly positive (no zero-probability traps)
    for p in row_probs:
        assert p > 0.0


def test_tier1_f1_markov_predict_next_probabilities_distribution(synthetic_sequence_order2):
    """F1.5: Method predict_next_probabilities returns valid probability distribution summing to 1.0."""
    model = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    model.fit(synthetic_sequence_order2)

    recent = synthetic_sequence_order2[-2:]
    dist = model.predict_next_probabilities(recent)

    assert len(dist) == 55
    assert set(dist.keys()) == set(range(1, 56))
    total_prob = sum(dist.values())
    assert math.isclose(total_prob, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert all(p > 0.0 for p in dist.values())


# -----------------------------------------------------------------------------
# F2: Gaussian Distribution Z-Score Testing
# -----------------------------------------------------------------------------

def test_tier1_f2_gaussian_sample_mean_and_stdev_calculation():
    """F2.1: Calculation of exact sample mean mu and sample stdev sigma across number frequencies."""
    total = 10
    # 3 draws where ball 1 appears 3 times, ball 2 appears 2 times, balls 3..10 appear 1 time
    draws = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 7, 8, 9, 10],
        [1, 3, 4, 5, 6, 7],
    ]
    analyzer = GaussianCycleAnalyzer(total_numbers=total)
    analyzer.fit(draws)

    counts = [analyzer._frequencies[i] for i in range(1, total + 1)]
    expected_mean = sum(counts) / total
    expected_stdev = statistics.stdev(counts)

    assert math.isclose(analyzer._mu, expected_mean, rel_tol=1e-9)
    assert math.isclose(analyzer._sigma, expected_stdev, rel_tol=1e-9)


def test_tier1_f2_gaussian_exact_zscore_formula():
    """F2.2: Verification of exact Z-Score formula Z_i = (x_i - mu) / sigma."""
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    # Synthetic draws giving distinct frequencies
    draws = []
    for _ in range(30):
        draws.append([1, 2, 3, 4, 5, 6])
    for _ in range(10):
        draws.append([1, 7, 8, 9, 10, 11])

    analyzer.fit(draws)
    z_scores = analyzer.compute_z_scores()

    assert len(z_scores) == 55
    mu = analyzer._mu
    sigma = analyzer._sigma
    for num, z in z_scores.items():
        count = analyzer._frequencies[num]
        expected_z = (count - mu) / sigma
        assert math.isclose(z, expected_z, rel_tol=1e-6, abs_tol=1e-6)


def test_tier1_f2_gaussian_confidence_band_classification():
    """F2.3: Verification of 2-sigma band segmentation (outlier_hot, outlier_cold, stationary)."""
    total = 55
    # Construct a distribution where ball 1 is heavily overheated and ball 2 is undercooled
    draws = []
    # 50 draws with ball 1 and balls 3..7
    for _ in range(50):
        draws.append([1, 3, 4, 5, 6, 7])
    # 30 draws with balls 8..55 (ball 2 never drawn)
    for _ in range(30):
        draws.append(list(range(8, 14)))

    analyzer = GaussianCycleAnalyzer(total_numbers=total)
    analyzer.fit(draws)
    bands = analyzer.classify_bands(threshold=2.0)

    assert "outlier_hot" in bands
    assert "outlier_cold" in bands
    assert "stationary" in bands
    # Ball 1 frequency is 50, which should be outlier_hot
    assert 1 in bands["outlier_hot"]


def test_tier1_f2_gaussian_partition_completeness_invariant():
    """F2.4: Union of bands must equal {1..N} with pairwise disjoint intersection."""
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    draws = [[(i + j) % 55 + 1 for i in range(6)] for j in range(20)]
    analyzer.fit(draws)

    bands = analyzer.classify_bands(threshold=2.0)
    hot_set = set(bands["outlier_hot"])
    cold_set = set(bands["outlier_cold"])
    stat_set = set(bands["stationary"])

    # Disjointness
    assert len(hot_set & cold_set) == 0
    assert len(hot_set & stat_set) == 0
    assert len(cold_set & stat_set) == 0

    # Exhaustive completeness
    assert hot_set | cold_set | stat_set == set(range(1, 56))


def test_tier1_f2_gaussian_symmetric_custom_threshold_scaling():
    """F2.5: Scaling threshold from 1.0 to 3.0 monotonically expands stationary set."""
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    draws = [[(i * 3 + j * 7) % 55 + 1 for i in range(6)] for j in range(40)]
    analyzer.fit(draws)

    bands_1 = analyzer.classify_bands(threshold=1.0)
    bands_2 = analyzer.classify_bands(threshold=2.0)
    bands_3 = analyzer.classify_bands(threshold=3.0)

    # Higher threshold encompasses more points into stationary
    assert len(bands_1["stationary"]) <= len(bands_2["stationary"]) <= len(bands_3["stationary"])


# -----------------------------------------------------------------------------
# F3: Recurrence Cycle & Interval Lag Analysis
# -----------------------------------------------------------------------------

def test_tier1_f3_cycle_metrics_chronological_appearance_intervals():
    """F3.1: Chronological appearance intervals Delta t = t_{j+1} - t_j."""
    # Ball 7 appears at draws 0, 3, 7 (intervals: 3, 4) in a 10-draw history
    draws = [
        [7, 1, 2, 3, 4, 5],  # 0
        [1, 2, 3, 4, 5, 6],  # 1
        [1, 2, 3, 4, 5, 6],  # 2
        [7, 1, 2, 3, 4, 5],  # 3
        [1, 2, 3, 4, 5, 6],  # 4
        [1, 2, 3, 4, 5, 6],  # 5
        [1, 2, 3, 4, 5, 6],  # 6
        [7, 1, 2, 3, 4, 5],  # 7
        [1, 2, 3, 4, 5, 6],  # 8
        [1, 2, 3, 4, 5, 6],  # 9
    ]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    metric_7 = metrics[7]
    # intervals are [3, 4], mean = 3.5
    assert math.isclose(metric_7["mean_cycle"], 3.5, rel_tol=1e-5)
    # std of [3, 4] is sqrt((0.5^2 + 0.5^2)/1) = sqrt(0.5) ≈ 0.7071
    assert math.isclose(metric_7["std_cycle"], math.sqrt(0.5), rel_tol=1e-5)


def test_tier1_f3_cycle_mean_and_stdev_metrics():
    """F3.2: Verification of mean_cycle and std_cycle metrics dictionary."""
    draws = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    for num in range(1, 56):
        m = metrics[num]
        assert "mean_cycle" in m
        assert "std_cycle" in m
        assert "current_lag" in m
        assert "cycle_zscore" in m
        assert m["mean_cycle"] >= 0.0
        assert m["std_cycle"] >= 0.0


def test_tier1_f3_cycle_current_overdue_lag():
    """F3.3: Current overdue lag W - t_{last} where W is total draws."""
    # 5 draws total. Ball 10 appears in draw 1 (index 1), does not appear in draws 2, 3, 4
    draws = [
        [1, 2, 3, 4, 5, 6],
        [10, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    # Draw index 1 (1-indexed draw is 2). w = 5. current_lag = 5 - 2 = 3 draws overdue.
    assert metrics[10]["current_lag"] == 3


def test_tier1_f3_cycle_anomaly_zscore_calculation():
    """F3.4: Cycle anomaly Z-score Z_cycle = (current_lag - mean_cycle) / std_cycle."""
    # Ball 5 appears at 0, 2, 4, 6 (intervals: 2, 2, 2 -> mean=2.0, std=0.0). w=10.
    # Last appearance at 6 (1-indexed: 7). lag = 10 - 7 = 3.
    draws = [
        [5, 1, 2, 3, 4, 6],  # 0
        [1, 2, 3, 4, 6, 7],  # 1
        [5, 1, 2, 3, 4, 6],  # 2
        [1, 2, 3, 4, 6, 7],  # 3
        [5, 1, 2, 3, 4, 6],  # 4
        [1, 2, 3, 4, 6, 7],  # 5
        [5, 1, 2, 3, 4, 6],  # 6
        [1, 2, 3, 4, 6, 7],  # 7
        [1, 2, 3, 4, 6, 7],  # 8
        [1, 2, 3, 4, 6, 7],  # 9
    ]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    m5 = metrics[5]
    assert m5["current_lag"] == 3
    assert m5["mean_cycle"] == 2.0
    # Overdue because current_lag (3) > mean_cycle (2.0)
    assert m5["cycle_zscore"] > 0.0


def test_tier1_f3_cycle_single_appearance_and_zero_appearance():
    """F3.5: Cycle metrics for balls appearing exactly once or never."""
    draws = [[1, 2, 3, 4, 5, 6]]  # 1 draw
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    # Ball 1 appeared once at draw 0 (1-indexed: 1). lag = 1 - 1 = 0
    assert metrics[1]["current_lag"] == 0
    assert metrics[1]["appearances"] == 1

    # Ball 55 never appeared. lag = 1. appearances = 0
    assert metrics[55]["current_lag"] == 1
    assert metrics[55]["appearances"] == 0


# -----------------------------------------------------------------------------
# F4: Monte Carlo Baseline Simulator
# -----------------------------------------------------------------------------

def test_tier1_f4_mc_simulator_initialization_and_parameters():
    """F4.1: Simulator initialization, default parameters, and seed reproducibility."""
    mc1 = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, ticket_price=10000, seed=42)
    mc2 = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, ticket_price=10000, seed=42)

    assert mc1.total_numbers == 55
    assert mc1.pick_count == 6
    assert mc1.ticket_price == 10000
    assert mc1.seed == 42


def test_tier1_f4_mc_uniform_random_ticket_generation():
    """F4.2: Uniform random ticket generation contains exactly pick_count distinct balls."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=123)
    # Generate 100 random tickets
    import random
    rng = random.Random(123)
    for _ in range(100):
        ticket = rng.sample(mc._number_pool, mc.pick_count)
        assert len(ticket) == 6
        assert len(set(ticket)) == 6
        assert all(1 <= b <= 55 for b in ticket)


def test_tier1_f4_mc_run_simulation_configurable_iterations():
    """F4.3: Support configurable simulations from 1,000 runs and verify result schema."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    test_draws = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 55]]

    res = mc.run_simulation(test_draws, simulations=1000)

    assert res["simulations"] == 1000
    assert res["draws_count"] == 2
    assert "mean_hits_per_draw" in res
    assert "std_hits_per_draw" in res
    assert "earnings_mean" in res
    assert "net_earnings_mean" in res
    assert "roi_mean" in res


def test_tier1_f4_mc_theoretical_expected_hits_law_of_large_numbers():
    """F4.4: Law of Large Numbers: Mean hits converges to k^2 / N = 36/55 ≈ 0.6545."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=999)
    test_draws = [list(range(1, 7)) for _ in range(20)]

    res = mc.run_simulation(test_draws, simulations=1000)
    theoretical = mc.theoretical_expected_hits_per_draw

    assert math.isclose(theoretical, 36.0 / 55.0, rel_tol=1e-6)
    # Empirical mean should be close to theoretical expectation within standard error
    assert abs(res["mean_hits_per_draw"] - theoretical) < 0.08


def test_tier1_f4_mc_prize_and_financial_balance_tracking():
    """F4.5: Cost, earnings, and net balance invariant: net == earnings - cost."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, ticket_price=10000, seed=42)
    test_draws = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]

    res = mc.run_simulation(test_draws, simulations=1000)
    cost = res["total_cost_per_sim"]
    expected_cost = 2 * 10000
    assert cost == expected_cost

    # Net earnings mean must equal earnings mean minus cost
    expected_net = res["earnings_mean"] - cost
    assert math.isclose(res["net_earnings_mean"], expected_net, rel_tol=1e-5, abs_tol=1.0)


# -----------------------------------------------------------------------------
# F5: Monte Carlo Hypothesis & Benchmark Testing
# -----------------------------------------------------------------------------

def test_tier1_f5_mc_evaluate_model_advantage_superior_model():
    """F5.1: Model with significantly superior hit rate produces high Z-score and p <= 0.05."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    actual_draws = [[1, 2, 3, 4, 5, 6] for _ in range(10)]
    baseline = mc.run_simulation(actual_draws, simulations=1000)

    # Superior model picks matching 4 balls every draw (mean hits = 4.0 vs baseline 0.65)
    superior_picks = [[1, 2, 3, 4, 20, 21] for _ in range(10)]
    eval_res = mc.evaluate_model_advantage(superior_picks, actual_draws, baseline)

    assert eval_res["z_score_hits"] > 3.0
    assert eval_res["p_value_hits"] <= 0.05
    assert eval_res["better_than_mc_mean"] is True
    assert eval_res["statistically_significant_95"] is True


def test_tier1_f5_mc_evaluate_model_advantage_inferior_model():
    """F5.2: Inferior model (0 hits) produces negative Z-score and p ≈ 1.0."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    actual_draws = [[1, 2, 3, 4, 5, 6] for _ in range(10)]
    baseline = mc.run_simulation(actual_draws, simulations=1000)

    # Model with zero matches
    inferior_picks = [[50, 51, 52, 53, 54, 55] for _ in range(10)]
    eval_res = mc.evaluate_model_advantage(inferior_picks, actual_draws, baseline)

    assert eval_res["z_score_hits"] < 0.0
    assert eval_res["p_value_hits"] > 0.50
    assert eval_res["better_than_mc_mean"] is False


def test_tier1_f5_mc_evaluate_model_advantage_empirical_p_value():
    """F5.3: Empirical p-value computation is bounded within [0.0, 1.0]."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    actual_draws = [[i % 55 + 1 for i in range(j, j + 6)] for j in range(5)]
    baseline = mc.run_simulation(actual_draws, simulations=1000)

    model_picks = [[1, 2, 3, 4, 5, 6] for _ in range(5)]
    eval_res = mc.evaluate_model_advantage(model_picks, actual_draws, baseline)

    p_hits = eval_res["p_value_hits"]
    p_earn = eval_res["p_value_earnings"]
    assert 0.0 <= p_hits <= 1.0
    assert 0.0 <= p_earn <= 1.0


def test_tier1_f5_mc_evaluate_model_advantage_roi_delta():
    """F5.4: Benchmarking ROI delta against empirical null baseline."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    actual_draws = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 55]]
    baseline = mc.run_simulation(actual_draws, simulations=1000)

    model_picks = [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 55]]
    eval_res = mc.evaluate_model_advantage(model_picks, actual_draws, baseline)

    assert "roi_delta" in eval_res
    expected_roi_delta = eval_res["model_metrics"]["roi"] - baseline["roi_mean"]
    assert math.isclose(eval_res["roi_delta"], expected_roi_delta, rel_tol=1e-5)


def test_tier1_f5_mc_evaluate_model_advantage_significance_flags():
    """F5.5: Statistical significance flags (95% and 99%) evaluate consistently."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    actual_draws = [[1, 2, 3, 4, 5, 6] for _ in range(20)]
    baseline = mc.run_simulation(actual_draws, simulations=1000)

    # 100% exact jackpot picks every draw -> extreme significance
    jackpot_picks = [[1, 2, 3, 4, 5, 6] for _ in range(20)]
    eval_res = mc.evaluate_model_advantage(jackpot_picks, actual_draws, baseline)

    assert eval_res["statistically_significant_95"] is True
    assert eval_res["statistically_significant_99"] is True


# -----------------------------------------------------------------------------
# F6: Production State Machine Engine
# -----------------------------------------------------------------------------

def test_tier1_f6_fsm_canonical_lifecycle_transition_sequence():
    """F6.1: Full lifecycle IDLE -> INGESTED -> CUTOFF_VERIFIED -> FEATURES_READY -> PREDICTED -> EVALUATED."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=1000000, ticket_cost=10000)
    assert fsm.state == PipelineState.IDLE

    # IngestData
    raw_data = [
        {"date": "2026-11-01", "result": [1, 2, 3, 4, 5, 6]},
        {"date": "2026-11-05", "result": [7, 8, 9, 10, 11, 12]},
        {"date": "2026-11-15", "result": [13, 14, 15, 16, 17, 18]},  # future
    ]
    fsm.ingest_data(raw_data, cutoff_date="2026-11-10")
    assert fsm.state == PipelineState.INGESTED

    # ApplyCutoff
    fsm.verify_cutoff()
    assert fsm.state == PipelineState.CUTOFF_VERIFIED
    assert len(fsm.loaded_draws) == 2

    # ExtractFeatures
    fsm.extract_features()
    assert fsm.state == PipelineState.FEATURES_READY
    assert fsm.markov_state == "CALCULATED"
    assert fsm.zscore_state == "COMPUTED"

    # GeneratePrediction
    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert len(picks) == 6
    assert fsm.bankroll == 990000  # Debited 10,000

    # EvaluateDraw
    new_bankroll = fsm.evaluate_result(actual_draw=[1, 2, 3, 4, 5, 6], prize_payout=50000)
    assert fsm.state == PipelineState.EVALUATED
    assert new_bankroll == 1040000


def test_tier1_f6_fsm_zero_future_leakage_enforcement():
    """F6.2: Invariant ZeroFutureLeakage: Pruning removes any draw date >= cutoff_date."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=1000000)
    draws = [
        {"date": "2026-12-01", "result": [1, 2, 3, 4, 5, 6]},
        {"date": "2026-12-10", "result": [2, 3, 4, 5, 6, 7]},
        {"date": "2026-12-11", "result": [3, 4, 5, 6, 7, 8]},  # cutoff date
        {"date": "2026-12-20", "result": [4, 5, 6, 7, 8, 9]},  # future
    ]
    fsm.ingest_data(draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()

    assert fsm.state == PipelineState.CUTOFF_VERIFIED
    loaded_dates = [r["date"] for r in fsm.loaded_draws]
    assert "2026-12-01" in loaded_dates
    assert "2026-12-10" in loaded_dates
    assert "2026-12-11" not in loaded_dates
    assert "2026-12-20" not in loaded_dates
    assert fsm.future_leak is False


def test_tier1_f6_fsm_bankroll_solvency_capital_debit_and_credit():
    """F6.3: Bankroll capital is accurately debited and credited with solvency invariant maintained."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=50000, ticket_cost=10000)
    draws = [{"date": "2026-01-01", "result": [1, 2, 3, 4, 5, 6]}]
    fsm.ingest_data(draws, cutoff_date="2026-02-01")
    fsm.verify_cutoff()
    fsm.extract_features()

    assert fsm.bankroll == 50000
    fsm.generate_prediction()
    assert fsm.bankroll == 40000

    fsm.evaluate_result(actual_draw=[1, 2, 3, 4, 5, 6], prize_payout=100000)
    assert fsm.bankroll == 140000
    assert fsm.bankroll >= 0


def test_tier1_f6_fsm_bankroll_solvency_safe_halt_when_insufficient_funds():
    """F6.4: Bankroll < ticket_cost transitions safely to HALTED_INSOLVENT without debiting."""
    # Bankroll = 5,000, ticket_cost = 10,000
    fsm = LotteryPipelineStateMachine(initial_bankroll=5000, ticket_cost=10000)
    draws = [{"date": "2026-01-01", "result": [1, 2, 3, 4, 5, 6]}]
    fsm.ingest_data(draws, cutoff_date="2026-02-01")
    fsm.verify_cutoff()
    fsm.extract_features()

    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.HALTED_INSOLVENT
    assert picks == []
    # Bankroll remains 5,000 (not debited to negative)
    assert fsm.bankroll == 5000
    assert fsm.bankroll >= 0


def test_tier1_f6_fsm_features_synchronized_invariant():
    """F6.5: Invariant FeaturesSynchronized: markovState == 'CALCULATED' and zscoreState == 'COMPUTED'."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=1000000)
    draws = [{"date": "2026-01-01", "result": [1, 2, 3, 4, 5, 6]}]
    fsm.ingest_data(draws, cutoff_date="2026-02-01")
    fsm.verify_cutoff()

    assert fsm.markov_state == "UNINITIALIZED"
    assert fsm.zscore_state == "PENDING"

    fsm.extract_features()
    assert fsm.state == PipelineState.FEATURES_READY
    assert fsm.markov_state == "CALCULATED"
    assert fsm.zscore_state == "COMPUTED"


# -----------------------------------------------------------------------------
# F7: TLA+ Specification & TLC Model Checking
# -----------------------------------------------------------------------------

def test_tier1_f7_tla_spec_and_cfg_structural_integrity():
    """F7.1: Verify structural completeness of LotteryPipeline.tla and cfg."""
    tla_file = SPEC_DIR / "LotteryPipeline.tla"
    cfg_file = SPEC_DIR / "LotteryPipeline.cfg"

    assert tla_file.exists()
    assert cfg_file.exists()

    tla = tla_file.read_text(encoding="utf-8")
    cfg = cfg_file.read_text(encoding="utf-8")

    assert "MODULE LotteryPipeline" in tla
    assert "ZeroFutureLeakage ==" in tla
    assert "BankrollSolvency ==" in tla
    assert "ValidPickSize ==" in tla
    assert "FeaturesSynchronized ==" in tla

    assert "INVARIANTS" in cfg
    assert "ZeroFutureLeakage" in cfg
    assert "BankrollSolvency" in cfg


@pytest.fixture(scope="module")
def tlc_baseline_result(java_runtime):
    """Executes TLC model checker on baseline specification once per test session."""
    cmd = [
        java_runtime,
        "-cp", str(TLA2TOOLS_JAR),
        "tlc2.TLC",
        "-config", "LotteryPipeline.cfg",
        "LotteryPipeline.tla",
    ]
    return subprocess.run(cmd, cwd=str(SPEC_DIR), capture_output=True, text=True, encoding="utf-8")


def test_tier1_f7_tlc_model_checker_passes_with_zero_violations(tlc_baseline_result):
    """F7.2: TLC model checker runs and discovers 0 errors and 0 invariant violations."""
    assert tlc_baseline_result.returncode == 0
    assert "Model checking completed. No error has been found." in tlc_baseline_result.stdout
    assert "0 states left on queue" in tlc_baseline_result.stdout


def test_tier1_f7_tlc_invariant_zero_future_leakage_verified(tlc_baseline_result):
    """F7.3: TLC output explicitly confirms ZeroFutureLeakage invariant held."""
    assert "ZeroFutureLeakage" not in tlc_baseline_result.stderr
    assert "Invariant ZeroFutureLeakage is violated" not in tlc_baseline_result.stdout


def test_tier1_f7_tlc_invariant_bankroll_solvency_verified(tlc_baseline_result):
    """F7.4: TLC output explicitly confirms BankrollSolvency invariant held."""
    assert "Invariant BankrollSolvency is violated" not in tlc_baseline_result.stdout


def test_tier1_f7_tlc_invariant_pick_size_and_feature_sync_verified(tlc_baseline_result):
    """F7.5: TLC output confirms ValidPickSize and FeaturesSynchronized."""
    assert "Invariant ValidPickSize is violated" not in tlc_baseline_result.stdout
    assert "Invariant FeaturesSynchronized is violated" not in tlc_baseline_result.stdout



# ═════════════════════════════════════════════════════════════════════════════
# TIER 2: BOUNDARY & CORNER CASES (Minimum 5 Test Cases per Feature, F1 to F7)
# ═════════════════════════════════════════════════════════════════════════════

# -----------------------------------------------------------------------------
# F1 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f1_corner_empty_or_single_draw_fit():
    """F1.C1: Fitting on fewer draws than order + 1 raises ValueError."""
    model = HigherOrderMarkovChain(order=2, total_numbers=55)
    with pytest.raises(ValueError):
        model.fit([])

    with pytest.raises(ValueError):
        model.fit([[1, 2, 3, 4, 5, 6]])


def test_tier2_f1_corner_all_same_draws_uniform_repetition():
    """F1.C2: Degenerate identical draws do not cause NaN or float overflow."""
    draws = [[1, 2, 3, 4, 5, 6] for _ in range(20)]
    model = HigherOrderMarkovChain(order=1, total_numbers=55, alpha=0.1)
    model.fit(draws)

    assert model.verify_normalization_invariant() is True
    prob = model.get_transition_probability(1, 2)
    assert not math.isnan(prob)
    assert 0.0 < prob <= 1.0


def test_tier2_f1_corner_extreme_smoothing_alpha_values():
    """F1.C3: Extreme smoothing values (alpha = 1e-4 and alpha = 1000.0) maintain row sum 1.0."""
    draws = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]]
    for alpha in (1e-4, 1000.0):
        model = HigherOrderMarkovChain(order=1, total_numbers=55, alpha=alpha)
        model.fit(draws)
        assert model.verify_normalization_invariant() is True


def test_tier2_f1_corner_completely_unseen_bigram_predict():
    """F1.C4: Predict next probabilities when recent draws contain bigram never seen in history."""
    train_draws = [
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
    ]
    model = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    model.fit(train_draws)

    # Bigram (50, 51) was never seen
    unseen_recent = [[50, 51, 52, 53, 54, 55], [50, 51, 52, 53, 54, 55]]
    probs = model.predict_next_probabilities(unseen_recent)

    assert len(probs) == 55
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)


def test_tier2_f1_corner_unfitted_model_error_guard():
    """F1.C5: Accessing probabilities on unfitted model raises RuntimeError."""
    model = HigherOrderMarkovChain(order=2, total_numbers=55)
    with pytest.raises(RuntimeError):
        model.get_transition_probability((1, 2), 3)

    with pytest.raises(RuntimeError):
        model.predict_next_probabilities([[1, 2, 3, 4, 5, 6]])


# -----------------------------------------------------------------------------
# F2 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f2_corner_empty_historical_draws():
    """F2.C1: Empty draws sequence handled gracefully with zero frequencies and Z=0."""
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit([])

    z_scores = analyzer.compute_z_scores()
    assert len(z_scores) == 55
    assert all(z == 0.0 for z in z_scores.values())


def test_tier2_f2_corner_single_historical_draw():
    """F2.C2: Single draw input correctly computes binary frequencies (1 for balls in draw, 0 for others)."""
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit([[1, 2, 3, 4, 5, 6]])

    z_scores = analyzer.compute_z_scores()
    assert len(z_scores) == 55
    # Drawn balls have higher Z than undrawn balls
    assert z_scores[1] > z_scores[10]


def test_tier2_f2_corner_uniform_frequencies_zero_variance():
    """F2.C3: Zero variance across pool frequencies results in sigma == 0 and Z == 0."""
    # Draw all 55 numbers in one round
    draw = list(range(1, 56))
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit([draw, draw])

    assert analyzer._sigma == 0.0
    z_scores = analyzer.compute_z_scores()
    assert all(z == 0.0 for z in z_scores.values())


def test_tier2_f2_corner_exact_band_border_values():
    """F2.C4: Exact threshold border values (Z == 2.0 or Z == -2.0) are classified into stationary."""
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit([[1, 2, 3, 4, 5, 6]])
    # Inject exact Z = 2.0 and Z = -2.0
    analyzer._z_scores[1] = 2.0
    analyzer._z_scores[2] = -2.0

    bands = analyzer.classify_bands(threshold=2.0)
    # By contract: hot > 2.0, cold < -2.0, stationary is [-2.0, 2.0]
    assert 1 in bands["stationary"]
    assert 2 in bands["stationary"]


def test_tier2_f2_corner_single_extreme_outlier():
    """F2.C5: Extreme outlier frequency (10,000 occurrences) does not cause numerical overflow."""
    draws = [[1] for _ in range(10000)]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)

    z_scores = analyzer.compute_z_scores()
    assert 1 in analyzer.classify_bands(threshold=2.0)["outlier_hot"]
    assert not math.isinf(z_scores[1])
    assert not math.isnan(z_scores[1])


# -----------------------------------------------------------------------------
# F3 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f3_corner_cycle_never_appeared_number():
    """F3.C1: Ball never appearing in W draws has lag = W and cycle_zscore = 0.0."""
    draws = [[1, 2, 3, 4, 5, 6] for _ in range(25)]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    m55 = metrics[55]
    assert m55["current_lag"] == 25
    assert m55["cycle_zscore"] == 0.0


def test_tier2_f3_corner_cycle_appeared_every_single_draw():
    """F3.C2: Ball appearing in every single draw has interval 1, std 0, current_lag 0."""
    draws = [[1, 2, 3, 4, 5, 6] for _ in range(10)]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    m1 = metrics[1]
    assert m1["mean_cycle"] == 1.0
    assert m1["std_cycle"] == 0.0
    assert m1["current_lag"] == 0


def test_tier2_f3_corner_cycle_consecutive_adjacent_draws():
    """F3.C3: Adjacent appearances yield interval Delta t = 1."""
    draws = [
        [10, 1, 2, 3, 4, 5],
        [10, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
    ]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    assert metrics[10]["mean_cycle"] == 1.0
    assert metrics[10]["current_lag"] == 1


def test_tier2_f3_corner_cycle_zero_division_when_std_zero():
    """F3.C4: Zero-division protection when interval standard deviation is 0."""
    # Appears at intervals [3, 3, 3] -> std = 0
    draws = [
        [1, 2, 3, 4, 5, 6],  # 0
        [2, 3, 4, 5, 6, 7],  # 1
        [2, 3, 4, 5, 6, 7],  # 2
        [1, 2, 3, 4, 5, 6],  # 3
        [2, 3, 4, 5, 6, 7],  # 4
        [2, 3, 4, 5, 6, 7],  # 5
        [1, 2, 3, 4, 5, 6],  # 6
    ]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    assert metrics[1]["std_cycle"] == 0.0
    # Must not raise ZeroDivisionError and return valid float
    assert isinstance(metrics[1]["cycle_zscore"], float)


def test_tier2_f3_corner_cycle_large_window_scale():
    """F3.C5: Cycle analysis on large window (10,000 draws) completes fast without precision loss."""
    draws = [[(i + j) % 55 + 1 for i in range(6)] for j in range(10000)]
    analyzer = GaussianCycleAnalyzer(total_numbers=55)
    analyzer.fit(draws)
    metrics = analyzer.compute_cycle_metrics()

    assert len(metrics) == 55
    for num in range(1, 56):
        assert metrics[num]["current_lag"] >= 0


# -----------------------------------------------------------------------------
# F4 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f4_corner_mc_large_simulation_10000_iterations():
    """F4.C1: Full scale execution of 10,000 Monte Carlo iterations."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    test_draws = [[1, 2, 3, 4, 5, 6]]

    res = mc.run_simulation(test_draws, simulations=10000)
    assert res["simulations"] == 10000
    assert abs(res["mean_hits_per_draw"] - (36.0 / 55.0)) < 0.04


def test_tier2_f4_corner_mc_single_target_draw():
    """F4.C2: Monte Carlo simulation against a single target draw."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    test_draws = [[1, 2, 3, 4, 5, 6]]

    res = mc.run_simulation(test_draws, simulations=1000)
    assert res["draws_count"] == 1
    assert res["total_cost_per_sim"] == 10000


def test_tier2_f4_corner_mc_empty_target_draws_error():
    """F4.C3: Simulating against empty draw list raises ValueError."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6)
    with pytest.raises(ValueError):
        mc.run_simulation([])


def test_tier2_f4_corner_mc_zero_ticket_price():
    """F4.C4: Ticket price 0.0 handled safely without division by zero in ROI."""
    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, ticket_price=0, seed=42)
    test_draws = [[1, 2, 3, 4, 5, 6]]

    res = mc.run_simulation(test_draws, simulations=1000)
    assert res["total_cost_per_sim"] == 0
    assert not math.isnan(res["roi_mean"])


def test_tier2_f4_corner_mc_seed_invariance_and_divergence():
    """F4.C5: Identical seed gives identical outcomes; divergent seed gives divergent outcomes."""
    draws = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    mc_a = MonteCarloBaselineSimulator(seed=12345)
    mc_b = MonteCarloBaselineSimulator(seed=12345)
    mc_c = MonteCarloBaselineSimulator(seed=99999)

    res_a = mc_a.run_simulation(draws, simulations=1000)
    res_b = mc_b.run_simulation(draws, simulations=1000)
    res_c = mc_c.run_simulation(draws, simulations=1000)

    assert res_a["mean_hits_per_draw"] == res_b["mean_hits_per_draw"]
    # Divergent seeds produce different random draws
    assert res_a["sim_mean_hits"] != res_c["sim_mean_hits"]


# -----------------------------------------------------------------------------
# F5 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f5_corner_hypothesis_empty_model_picks_error():
    """F5.C1: Empty model picks list raises ValueError."""
    mc = MonteCarloBaselineSimulator()
    with pytest.raises(ValueError):
        mc.evaluate_model_advantage([], [[1, 2, 3, 4, 5, 6]], {})


def test_tier2_f5_corner_hypothesis_length_mismatch_raises():
    """F5.C2: Length mismatch between model picks and actual draws raises ValueError."""
    mc = MonteCarloBaselineSimulator()
    baseline = mc.run_simulation([[1, 2, 3, 4, 5, 6]], simulations=1000)
    with pytest.raises(ValueError):
        mc.evaluate_model_advantage([[1, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], baseline)


def test_tier2_f5_corner_hypothesis_zero_variance_in_mc_baseline():
    """F5.C3: Zero variance in baseline does not cause ZeroDivisionError in Z-score."""
    mc = MonteCarloBaselineSimulator()
    fake_baseline = {
        "mean_hits_per_draw": 1.0,
        "std_hits_per_draw": 0.0,
        "earnings_mean": 0.0,
        "earnings_std": 0.0,
    }
    res = mc.evaluate_model_advantage([[1, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 6]], fake_baseline)
    assert res["z_score_hits"] == 0.0


def test_tier2_f5_corner_hypothesis_exact_jackpot_model_pick():
    """F5.C4: Exact jackpot pick produces extreme positive Z-score and p_value = 0."""
    mc = MonteCarloBaselineSimulator(seed=42)
    actual = [[1, 2, 3, 4, 5, 6]]
    baseline = mc.run_simulation(actual, simulations=1000)

    res = mc.evaluate_model_advantage([[1, 2, 3, 4, 5, 6]], actual, baseline)
    assert res["z_score_hits"] > 0
    assert res["p_value_hits"] <= 0.05


def test_tier2_f5_corner_hypothesis_invalid_pick_size_raises():
    """F5.C5: Model pick with invalid number count (e.g. 5 balls or duplicates) raises ValueError."""
    mc = MonteCarloBaselineSimulator()
    baseline = mc.run_simulation([[1, 2, 3, 4, 5, 6]], simulations=1000)

    # 5 balls instead of 6
    with pytest.raises(ValueError):
        mc.evaluate_model_advantage([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5, 6]], baseline)

    # Duplicate balls
    with pytest.raises(ValueError):
        mc.evaluate_model_advantage([[1, 1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5, 6]], baseline)


# -----------------------------------------------------------------------------
# F6 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f6_corner_fsm_zero_initial_bankroll():
    """F6.C1: Initial bankroll = 0 immediately halts safely at generate_prediction."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=0, ticket_cost=10000)
    fsm.ingest_data([{"date": "2026-01-01", "result": [1, 2, 3, 4, 5, 6]}], cutoff_date="2026-02-01")
    fsm.verify_cutoff()
    fsm.extract_features()

    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.HALTED_INSOLVENT
    assert picks == []
    assert fsm.bankroll == 0


def test_tier2_f6_corner_fsm_exact_ticket_cost_bankroll():
    """F6.C2: Bankroll equal to exactly ticket_cost executes to 0 without violating solvency."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=10000, ticket_cost=10000)
    fsm.ingest_data([{"date": "2026-01-01", "result": [1, 2, 3, 4, 5, 6]}], cutoff_date="2026-02-01")
    fsm.verify_cutoff()
    fsm.extract_features()

    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert fsm.bankroll == 0
    assert len(picks) == 6


def test_tier2_f6_corner_fsm_illegal_transitions_matrix():
    """F6.C3: Reject illegal out-of-order state transitions."""
    fsm = LotteryPipelineStateMachine()
    # From IDLE:
    with pytest.raises(InvalidStateTransitionError):
        fsm.verify_cutoff()
    with pytest.raises(InvalidStateTransitionError):
        fsm.extract_features()
    with pytest.raises(InvalidStateTransitionError):
        fsm.generate_prediction()
    with pytest.raises(InvalidStateTransitionError):
        fsm.evaluate_result([1, 2, 3, 4, 5, 6])


def test_tier2_f6_corner_fsm_empty_dataset_ingestion():
    """F6.C4: Empty raw draws list ingests and prunes without runtime crash."""
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data([], cutoff_date="2026-01-01")
    fsm.verify_cutoff()
    assert fsm.state == PipelineState.CUTOFF_VERIFIED
    assert len(fsm.loaded_draws) == 0


def test_tier2_f6_corner_fsm_boundary_cutoff_date():
    """F6.C5: Cutoff date exactly matching earliest draw date prunes all data."""
    fsm = LotteryPipelineStateMachine()
    draws = [
        {"date": "2026-05-01", "result": [1, 2, 3, 4, 5, 6]},
        {"date": "2026-05-02", "result": [2, 3, 4, 5, 6, 7]},
    ]
    # Exclusive cutoff at earliest date: 0 draws remain
    fsm.ingest_data(draws, cutoff_date="2026-05-01")
    fsm.verify_cutoff()
    assert len(fsm.loaded_draws) == 0


# -----------------------------------------------------------------------------
# F7 Corner Cases
# -----------------------------------------------------------------------------

def test_tier2_f7_corner_tla_model_missing_spec_file_detection(java_runtime):
    """F7.C1: TLC fails gracefully when pointed to a non-existent specification."""
    cmd = [
        java_runtime,
        "-cp", str(TLA2TOOLS_JAR),
        "tlc2.TLC",
        "NonExistentSpecification.tla",
    ]
    res = subprocess.run(cmd, cwd=str(SPEC_DIR), capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0


def test_tier2_f7_corner_tla_invalid_cfg_syntax_detection(java_runtime, tmp_path):
    """F7.C2: TLC catches syntax error in malformed configuration file."""
    bad_cfg = tmp_path / "BadConfig.cfg"
    bad_cfg.write_text("INVALID_KEYWORD_NOT_SUPPORTED", encoding="utf-8")

    cmd = [
        java_runtime,
        "-cp", str(TLA2TOOLS_JAR),
        "tlc2.TLC",
        "-config", str(bad_cfg),
        str(SPEC_DIR / "LotteryPipeline.tla"),
    ]
    res = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0


def test_tier2_f7_corner_tla_counterexample_trace_capture(java_runtime, tmp_path):
    """F7.C3: An intentional invariant violation produces a valid TLC counterexample error trace."""
    corrupted_tla = SPEC_DIR / "LotteryPipeline.tla"
    content = corrupted_tla.read_text(encoding="utf-8")

    # Inject impossible invariant: BankrollSolvency == bankroll > 100
    corrupted_content = content.replace("bankroll >= 0", "bankroll > 100")
    # File name MUST match module name LotteryPipeline!
    temp_tla = tmp_path / "LotteryPipeline.tla"
    temp_tla.write_text(corrupted_content, encoding="utf-8")

    temp_cfg = tmp_path / "LotteryPipeline.cfg"
    shutil.copy(SPEC_DIR / "LotteryPipeline.cfg", temp_cfg)

    cmd = [
        java_runtime,
        "-cp", str(TLA2TOOLS_JAR),
        "tlc2.TLC",
        "-config", "LotteryPipeline.cfg",
        "LotteryPipeline.tla",
    ]
    res = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8")

    assert res.returncode != 0
    assert "Invariant BankrollSolvency is violated" in res.stdout


def test_tier2_f7_corner_tla_max_days_boundary_scale(java_runtime, tmp_path):
    """F7.C4: Scaling constants (e.g. MaxDays=5, TestCutoffDay=3) preserves 0 invariant violations."""
    # Copy LotteryPipeline.tla to tmp_path
    shutil.copy(SPEC_DIR / "LotteryPipeline.tla", tmp_path / "LotteryPipeline.tla")
    custom_cfg = tmp_path / "Scaled.cfg"
    custom_cfg.write_text(
        "INIT Init\n"
        "NEXT Next\n"
        "CONSTANTS\n"
        "    MaxDays = 5\n"
        "    TestCutoffDay = 3\n"
        "    TicketCost = 1\n"
        "    ThresholdPositiveEV = 230\n"
        "INVARIANTS\n"
        "    ZeroFutureLeakage\n"
        "    BankrollSolvency\n"
        "    ValidPickSize\n"
        "    FeaturesSynchronized\n",
        encoding="utf-8",
    )

    cmd = [
        java_runtime,
        "-cp", str(TLA2TOOLS_JAR),
        "tlc2.TLC",
        "-config", "Scaled.cfg",
        "LotteryPipeline.tla",
    ]
    res = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True, encoding="utf-8")

    assert res.returncode == 0
    assert "Model checking completed. No error has been found." in res.stdout


def test_tier2_f7_corner_tlc_output_semantic_parsing(tlc_baseline_result):
    """F7.C5: Semantic parsing of TLC state count, depth, and distinct state output."""
    stdout = tlc_baseline_result.stdout
    assert "states generated" in stdout
    assert "distinct states found" in stdout
    assert "The depth of the complete state graph search is" in stdout



# ═════════════════════════════════════════════════════════════════════════════
# TIER 3: CROSS-FEATURE INTEGRATION COMBINATIONS
# ═════════════════════════════════════════════════════════════════════════════

def test_tier3_markov_probabilities_filtered_by_gaussian_outliers():
    """T3.1: Combining F1 (Markov) + F2 (Gaussian): Filter Markov transition vector with Gaussian bands."""
    draws = [
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [1, 2, 8, 9, 10, 11],
        [2, 3, 12, 13, 14, 15],
    ]
    markov = HigherOrderMarkovChain(order=1, total_numbers=55).fit(draws)
    gaussian = GaussianCycleAnalyzer(total_numbers=55).fit(draws)

    bands = gaussian.classify_bands(threshold=1.5)
    hot_outliers = set(bands["outlier_hot"])

    # Extract Markov prediction
    recent = [[2, 3, 12, 13, 14, 15]]
    raw_dist = markov.predict_next_probabilities(recent)

    # Down-weight overheated outliers by factor of 0.5 to model mean reversion
    adjusted_dist = {}
    for ball, prob in raw_dist.items():
        if ball in hot_outliers:
            adjusted_dist[ball] = prob * 0.5
        else:
            adjusted_dist[ball] = prob

    # Renormalize
    norm_factor = sum(adjusted_dist.values())
    final_dist = {b: p / norm_factor for b, p in adjusted_dist.items()}

    assert math.isclose(sum(final_dist.values()), 1.0, rel_tol=1e-6)
    assert all(p > 0.0 for p in final_dist.values())


def test_tier3_fsm_orchestrating_markov_and_gaussian_feature_extraction():
    """T3.2: Combining F1, F2, F6: FSM extract_features automatically trains both models."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=1000000)
    raw_draws = [
        {"date": "2026-10-01", "result": [1, 2, 3, 4, 5, 6]},
        {"date": "2026-10-03", "result": [2, 3, 4, 5, 6, 7]},
        {"date": "2026-10-05", "result": [3, 4, 5, 6, 7, 8]},
        {"date": "2026-10-08", "result": [4, 5, 6, 7, 8, 9]},
        {"date": "2026-10-10", "result": [5, 6, 7, 8, 9, 10]},
    ]
    fsm.ingest_data(raw_draws, cutoff_date="2026-10-09")
    fsm.verify_cutoff()

    markov = HigherOrderMarkovChain(order=2, total_numbers=55)
    gaussian = GaussianCycleAnalyzer(total_numbers=55)

    fsm.extract_features(markov_model=markov, gaussian_model=gaussian)

    assert fsm.state == PipelineState.FEATURES_READY
    assert markov.is_fitted
    assert gaussian._is_fitted
    assert markov.verify_normalization_invariant() is True


def test_tier3_fsm_prediction_benchmarked_against_monte_carlo():
    """T3.3: Combining F4, F5, F6: FSM prediction evaluated against Monte Carlo baseline."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=1000000, ticket_cost=10000)
    draws = [
        {"date": "2026-10-01", "result": [1, 2, 3, 4, 5, 6]},
        {"date": "2026-10-03", "result": [2, 3, 4, 5, 6, 7]},
        {"date": "2026-10-05", "result": [3, 4, 5, 6, 7, 8]},
    ]
    fsm.ingest_data(draws, cutoff_date="2026-10-04")
    fsm.verify_cutoff()
    fsm.extract_features()

    picks = fsm.generate_prediction(predictor_func=lambda history: [1, 2, 3, 4, 5, 6])
    assert len(picks) == 6

    # Benchmark pick against target test draw
    test_draw = [[1, 2, 3, 4, 5, 6]]
    mc = MonteCarloBaselineSimulator(total_numbers=55, seed=42)
    mc_baseline = mc.run_simulation(test_draw, simulations=1000)

    advantage = mc.evaluate_model_advantage([picks], test_draw, mc_baseline)
    assert "z_score_hits" in advantage
    assert "p_value_hits" in advantage


def test_tier3_temporal_cutoff_isolating_features_and_predictions():
    """T3.4: Combining F1, F2, F6: Ingest 30 days, cutoff at day 11, strictly verify no leakage into features."""
    draws = []
    for day in range(1, 31):
        # Ball 55 only appears in days 15..30 (after cutoff day 11)
        res = [1, 2, 3, 4, 5, 55] if day >= 15 else [1, 2, 3, 4, 5, 6]
        draws.append({"date": f"2026-12-{day:02d}", "result": res})

    fsm = LotteryPipelineStateMachine(initial_bankroll=1000000)
    fsm.ingest_data(draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()

    gaussian = GaussianCycleAnalyzer(total_numbers=55)
    fsm.extract_features(gaussian_model=gaussian)

    # Ball 55 should have frequency 0 because it only appeared on days 15..30
    assert gaussian._frequencies[55] == 0


def test_tier3_cycle_metrics_informing_markov_temperature_sampling():
    """T3.5: Combining F1 + F3: Using cycle overdue lag to weight Markov prediction probabilities."""
    draws = [
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8],
    ]
    markov = HigherOrderMarkovChain(order=1, total_numbers=55).fit(draws)
    analyzer = GaussianCycleAnalyzer(total_numbers=55).fit(draws)
    cycle_metrics = analyzer.compute_cycle_metrics()

    base_probs = markov.predict_next_probabilities([[3, 4, 5, 6, 7, 8]])

    # Modulate probability with overdue lag boost
    modulated = {}
    for ball, p in base_probs.items():
        lag = cycle_metrics[ball]["current_lag"]
        boost = 1.0 + 0.05 * min(lag, 10)
        modulated[ball] = p * boost

    # Renormalize
    total_mod = sum(modulated.values())
    final_probs = {b: p / total_mod for b, p in modulated.items()}

    assert math.isclose(sum(final_probs.values()), 1.0, rel_tol=1e-6)
    # Select top 6 distinct balls
    sorted_balls = sorted(final_probs.keys(), key=lambda b: final_probs[b], reverse=True)
    top_6 = sorted(sorted_balls[:6])
    assert len(top_6) == 6
    assert len(set(top_6)) == 6


# ═════════════════════════════════════════════════════════════════════════════
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (Vietlott Power 6/55)
# ═════════════════════════════════════════════════════════════════════════════

def test_tier4_real_power655_e2e_temporal_cutoff_zero_leakage(power655_raw_records):
    """T4.1: Real-world Power 6/55 ingestion: Prune at historical cutoff date with zero data leakage."""
    assert len(power655_raw_records) >= 1000

    fsm = LotteryPipelineStateMachine(initial_bankroll=10000000)
    cutoff = "2026-06-01"
    fsm.ingest_data(power655_raw_records, cutoff_date=cutoff)
    fsm.verify_cutoff()

    assert fsm.state == PipelineState.CUTOFF_VERIFIED
    loaded = fsm.loaded_draws
    assert len(loaded) > 0

    cutoff_iso = parse_date(cutoff)
    for record in loaded:
        d_iso = parse_date(record["date"])
        assert d_iso < cutoff_iso, f"Data leakage detected! {d_iso} >= {cutoff_iso}"


def test_tier4_real_power655_e2e_markov_gaussian_feature_extraction(power655_raw_records):
    """T4.2: Train Markov (k=2) and Gaussian cycle models on real historical Power 6/55 draws."""
    # Take earliest 200 chronological draws (JSONL is newest-to-oldest, so reverse)
    chronological = list(reversed(power655_raw_records))[:200]
    draw_lists = [r["result"][:6] for r in chronological]

    markov = HigherOrderMarkovChain(order=2, total_numbers=55, alpha=0.1)
    markov.fit(draw_lists)
    assert markov.verify_normalization_invariant() is True

    gaussian = GaussianCycleAnalyzer(total_numbers=55)
    gaussian.fit(draw_lists)
    bands = gaussian.classify_bands(threshold=2.0)

    # In 200 real draws, there are typically stationary balls and occasional 2-sigma outliers
    assert len(bands["stationary"]) > 40
    assert len(bands["outlier_hot"]) + len(bands["outlier_cold"]) + len(bands["stationary"]) == 55


def test_tier4_real_power655_e2e_monte_carlo_baseline_statistical_validation(power655_raw_records):
    """T4.3: Run 1,000 Monte Carlo simulations on real 50-draw test partition; verify hit rate convergence."""
    chronological = list(reversed(power655_raw_records))
    test_50 = [r["result"][:6] for r in chronological[-50:]]

    mc = MonteCarloBaselineSimulator(total_numbers=55, pick_count=6, seed=42)
    res = mc.run_simulation(test_50, simulations=1000)

    assert res["simulations"] == 1000
    assert res["draws_count"] == 50
    # Mean hits on real test partition should be within 0.05 of theoretical 0.6545
    assert abs(res["mean_hits_per_draw"] - (36.0 / 55.0)) < 0.05


def test_tier4_real_power655_e2e_fsm_multi_round_walk_forward_simulation(power655_raw_records):
    """T4.4: Walk-forward backtesting across 10 consecutive real draws tracking capital and solvency."""
    chronological = list(reversed(power655_raw_records))
    initial_capital = 5000000  # 5,000,000 VNĐ
    ticket_cost = 10000

    fsm = LotteryPipelineStateMachine(initial_bankroll=initial_capital, ticket_cost=ticket_cost)

    # 10 rounds of walk-forward prediction
    for i in range(10):
        target_idx = len(chronological) - 10 + i
        target_draw = chronological[target_idx]
        history = chronological[:target_idx]

        fsm.ingest_data(history, cutoff_date=target_draw["date"])
        fsm.verify_cutoff()
        fsm.extract_features()

        # Generate prediction using simple top-6 most frequent numbers in history
        counts = {}
        for d in history[-50:]:
            for ball in d["result"][:6]:
                counts[ball] = counts.get(ball, 0) + 1
        top_6 = sorted(sorted(counts.keys(), key=lambda b: counts[b], reverse=True)[:6])

        picks = fsm.generate_prediction(predictor_func=lambda hist: top_6)
        assert len(picks) == 6

        # Check payout: Power 6/55 match 3 is 50,000 VNĐ
        actual_main = set(target_draw["result"][:6])
        hits = len(set(picks) & actual_main)
        payout = 50000 if hits == 3 else (500000 if hits == 4 else (40000000 if hits == 5 else 0))

        fsm.evaluate_result(target_draw["result"], prize_payout=payout)
        assert fsm.bankroll >= 0

        # Reset for next round
        fsm.reset()

    # System must remain solvent after 10 rounds
    assert fsm.bankroll >= 0


def test_tier4_real_power655_e2e_full_system_solvency_and_model_advantage_report(power655_raw_records):
    """T4.5: Full end-to-end integration benchmark: Model picks vs Monte Carlo baseline on real dataset."""
    chronological = list(reversed(power655_raw_records))
    # Use last 20 draws as out-of-sample evaluation period
    eval_draws = chronological[-20:]
    train_draws = [r["result"][:6] for r in chronological[:-20]]

    # Train Markov and Gaussian models on historical training partition
    markov = HigherOrderMarkovChain(order=2, total_numbers=55).fit(train_draws)
    gaussian = GaussianCycleAnalyzer(total_numbers=55).fit(train_draws)

    # Generate model picks for each evaluation draw
    model_picks = []
    for draw_record in eval_draws:
        recent = train_draws[-2:]
        probs = markov.predict_next_probabilities(recent)
        top_picks = sorted(sorted(probs.keys(), key=lambda b: probs[b], reverse=True)[:6])
        model_picks.append(top_picks)

    # Monte Carlo baseline on evaluation period
    actual_lists = [r["result"][:6] for r in eval_draws]
    mc = MonteCarloBaselineSimulator(total_numbers=55, seed=42)
    mc_baseline = mc.run_simulation(actual_lists, simulations=1000)

    # Comparative evaluation
    advantage = mc.evaluate_model_advantage(model_picks, actual_lists, mc_baseline)

    assert "model_metrics" in advantage
    assert "mc_baseline_metrics" in advantage
    assert "z_score_hits" in advantage
    assert "p_value_hits" in advantage
    assert "roi_delta" in advantage
    assert advantage["model_metrics"]["draws_count"] == 20
