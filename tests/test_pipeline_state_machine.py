#!/usr/bin/env python3
"""
Test Suite: Formal Pipeline State Machine Verification
──────────────────────────────────────────────────────
Unit, property, and invariant verification tests for LotteryPipelineStateMachine
mirroring TLA+ formal specification in spec/LotteryPipeline.tla.

Key Invariant Guarantees Tested:
  1. ZeroFutureLeakage: Strict temporal filtration preventing look-ahead bias.
  2. BankrollSolvency: Capital never drops below zero; safe trap to HALTED_INSOLVENT.
  3. ValidPickSize: Exactly 6 distinct valid lottery numbers.
  4. FeaturesSynchronized: Markov chain and Gaussian Z-score computed before prediction.
  5. Invalid Transition Rejection: Rejects any illegal transition out of lifecycle order.
"""
import sys
import pathlib
import pytest
from typing import List, Dict, Any

# Ensure scripts directory is on sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_fsm import (
    LotteryPipelineStateMachine,
    PipelineState,
    InvariantViolationError,
    BankrollSolvencyError,
    ZeroFutureLeakageError,
    InvalidStateTransitionError,
)


@pytest.fixture
def sample_30_days_draws() -> List[Dict[str, Any]]:
    """Fixture providing 30 consecutive draws for December 2026."""
    rows = []
    for day in range(1, 31):
        d_str = f"2026-12-{day:02d}"
        rows.append({
            "id": f"D{day:04d}",
            "date": d_str,
            "result": [
                ((day * 3 + 1) % 55) + 1,
                ((day * 5 + 7) % 55) + 1,
                ((day * 7 + 13) % 55) + 1,
                ((day * 11 + 17) % 55) + 1,
                ((day * 13 + 19) % 55) + 1,
                ((day * 17 + 23) % 55) + 1,
            ],
        })
    return rows


# ==============================================================================
# 1. INITIALIZATION & PARAMETER VALIDATION
# ==============================================================================

def test_fsm_initial_state():
    """Verify fresh state machine starts in canonical IDLE state with default parameters."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000, ticket_cost=10000, pick_size=6)
    assert fsm.state == PipelineState.IDLE
    assert fsm.phase == PipelineState.IDLE
    assert fsm.bankroll == 100000
    assert fsm.ticket_cost == 10000
    assert fsm.pick_size == 6
    assert fsm.markov_state == "UNINITIALIZED"
    assert fsm.zscore_state == "PENDING"
    assert fsm.picks == []
    assert fsm.picks_count == 0
    assert fsm.future_leak is False
    assert fsm.current_round == 1


def test_fsm_initialization_negative_bankroll():
    """Verify initializing with negative bankroll violates BankrollSolvency immediately."""
    with pytest.raises(BankrollSolvencyError) as exc_info:
        LotteryPipelineStateMachine(initial_bankroll=-500)
    assert "BankrollSolvency" in str(exc_info.value)


def test_fsm_initialization_invalid_ticket_or_pick():
    """Verify invalid ticket cost or pick size triggers ValueError."""
    with pytest.raises(ValueError):
        LotteryPipelineStateMachine(ticket_cost=0)
    with pytest.raises(ValueError):
        LotteryPipelineStateMachine(ticket_cost=-1000)
    with pytest.raises(ValueError):
        LotteryPipelineStateMachine(pick_size=0)


# ==============================================================================
# 2. CANONICAL HAPPY PATH LIFECYCLE (IDLE -> EVALUATED)
# ==============================================================================

def test_canonical_lifecycle_execution(sample_30_days_draws):
    """
    Test complete canonical pipeline execution:
    IDLE -> INGESTED -> CUTOFF_VERIFIED -> FEATURES_READY -> PREDICTED -> EVALUATED
    """
    fsm = LotteryPipelineStateMachine(initial_bankroll=50000, ticket_cost=10000)
    assert fsm.state == PipelineState.IDLE

    # 1. Ingest Data
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    assert fsm.state == PipelineState.INGESTED
    assert len(fsm.loaded_draws) == 30

    # 2. Verify Cutoff (exclusive: keeps 01/12 -> 10/12, exactly 10 records)
    fsm.verify_cutoff()
    assert fsm.state == PipelineState.CUTOFF_VERIFIED
    assert len(fsm.loaded_draws) == 10
    assert all(r["date"] < "2026-12-11" for r in fsm.loaded_draws)

    # 3. Extract Features
    fsm.extract_features()
    assert fsm.state == PipelineState.FEATURES_READY
    assert fsm.markov_state == "CALCULATED"
    assert fsm.zscore_state == "COMPUTED"

    # 4. Generate Prediction (debits 10,000, bankroll: 50,000 -> 40,000)
    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert len(picks) == 6
    assert fsm.bankroll == 40000
    assert fsm.picks_count == 6

    # 5. Evaluate Result (prize payout 100,000, bankroll: 40,000 -> 140,000)
    actual_draw = {"result": [1, 2, 3, 10, 20, 30]}
    new_bankroll = fsm.evaluate_result(actual_draw, prize_payout=100000)
    assert fsm.state == PipelineState.EVALUATED
    assert new_bankroll == 140000
    assert fsm.bankroll == 140000
    assert fsm.evaluation_result is not None
    assert fsm.evaluation_result["prize_payout"] == 100000
    assert fsm.evaluation_result["bankroll"] == 140000


# ==============================================================================
# 3. MULTI-ROUND BACKTEST TRANSITION (EVALUATED -> IDLE)
# ==============================================================================

def test_multi_round_backtesting_transition(sample_30_days_draws):
    """Verify multiple rounds preserve cumulative bankroll across iterations."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=30000, ticket_cost=10000)

    # Round 1: Buy ticket (30k -> 20k), win 0 -> bankroll 20k
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.generate_prediction()
    fsm.evaluate_result([50, 51, 52, 53, 54, 55], prize_payout=0)
    assert fsm.state == PipelineState.EVALUATED
    assert fsm.bankroll == 20000
    assert fsm.current_round == 1

    # Transition to Round 2
    fsm.next_round()
    assert fsm.state == PipelineState.IDLE
    assert fsm.current_round == 2
    assert fsm.bankroll == 20000  # Preserved
    assert fsm.picks == []
    assert fsm.markov_state == "UNINITIALIZED"
    assert fsm.zscore_state == "PENDING"

    # Round 2: Buy ticket (20k -> 10k), win 50k -> bankroll 60k
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-12")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.generate_prediction()
    fsm.evaluate_result([1, 2, 3, 4, 5, 6], prize_payout=50000)
    assert fsm.state == PipelineState.EVALUATED
    assert fsm.bankroll == 60000
    assert fsm.current_round == 2


# ==============================================================================
# 4. ILLEGAL TRANSITIONS REJECTION
# ==============================================================================

@pytest.mark.parametrize("invalid_call", [
    lambda fsm: fsm.verify_cutoff(),
    lambda fsm: fsm.extract_features(),
    lambda fsm: fsm.generate_prediction(),
    lambda fsm: fsm.evaluate_result([1, 2, 3, 4, 5, 6]),
    lambda fsm: fsm.next_round(),
])
def test_illegal_transitions_from_idle(invalid_call):
    """Verify invoking out-of-order actions from IDLE raises InvalidStateTransitionError."""
    fsm = LotteryPipelineStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        invalid_call(fsm)


def test_illegal_transitions_from_ingested(sample_30_days_draws):
    """Verify illegal jumps from INGESTED directly to FEATURES_READY or PREDICTED."""
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")

    # Cannot skip cutoff verification to extract features
    with pytest.raises(InvalidStateTransitionError):
        fsm.extract_features()

    # Cannot skip directly to prediction
    with pytest.raises(InvalidStateTransitionError):
        fsm.generate_prediction()


def test_illegal_transitions_from_features_ready(sample_30_days_draws):
    """Verify illegal transitions from FEATURES_READY."""
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    # Cannot evaluate without predicting first
    with pytest.raises(InvalidStateTransitionError):
        fsm.evaluate_result([1, 2, 3, 4, 5, 6])


# ==============================================================================
# 5. CORE INVARIANT 1: ZeroFutureLeakage
# ==============================================================================

def test_zero_future_leakage_pruning(sample_30_days_draws):
    """Verify strict cutoff date enforcement completely removes future draws."""
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()

    assert len(fsm.loaded_draws) == 10
    dates = [r["date"] for r in fsm.loaded_draws]
    assert all(d < "2026-12-11" for d in dates)
    assert "2026-12-11" not in dates
    assert "2026-12-30" not in dates


def test_zero_future_leakage_invariant_detection(sample_30_days_draws):
    """
    Verify that if future data is maliciously or erroneously injected
    after cutoff verification, the invariant immediately triggers ZeroFutureLeakageError.
    """
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()

    # Maliciously contaminate internal buffer with future record
    fsm._loaded_draws.append({"id": "LEAK01", "date": "2026-12-25", "result": [1, 2, 3, 4, 5, 6]})

    # Invariant assertion must detect and reject the contamination
    with pytest.raises(ZeroFutureLeakageError) as exc_info:
        fsm.assert_invariants()
    assert "ZeroFutureLeakage" in str(exc_info.value)
    assert fsm.future_leak is True


# ==============================================================================
# 6. CORE INVARIANT 2: BankrollSolvency & HALTED_INSOLVENT
# ==============================================================================

def test_bankroll_solvency_safe_halt_when_insufficient_capital(sample_30_days_draws):
    """
    Verify that when capital is strictly less than ticket cost:
    - FSM transitions safely to HALTED_INSOLVENT.
    - Bankroll is NOT debited (never negative).
    - Returns empty list of picks.
    - BankrollSolvency invariant strictly holds.
    """
    # Capital: 5,000, Ticket cost: 10,000
    fsm = LotteryPipelineStateMachine(initial_bankroll=5000, ticket_cost=10000)
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.HALTED_INSOLVENT
    assert picks == []
    assert fsm.bankroll == 5000  # Capital preserved, zero debit!
    assert fsm.bankroll >= 0


def test_bankroll_solvency_exact_ticket_cost(sample_30_days_draws):
    """
    Boundary test: bankroll exactly equals ticket cost.
    Prediction debits full amount leaving bankroll == 0 (valid: 0 >= 0).
    Subsequent round halts insolvent.
    """
    fsm = LotteryPipelineStateMachine(initial_bankroll=10000, ticket_cost=10000)
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert len(picks) == 6
    assert fsm.bankroll == 0  # 0 >= 0, invariant preserved!
    fsm.assert_invariants()

    # Next round with 0 bankroll
    fsm.evaluate_result([9, 10, 11, 12, 13, 14], prize_payout=0)
    fsm.next_round()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-12")
    fsm.verify_cutoff()
    fsm.extract_features()

    next_picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.HALTED_INSOLVENT
    assert next_picks == []
    assert fsm.bankroll == 0


def test_bankroll_solvency_negative_corruption_fails_invariant():
    """Verify manual bankroll corruption (< 0) raises BankrollSolvencyError."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=10000)
    fsm._bankroll = -1
    with pytest.raises(BankrollSolvencyError) as exc_info:
        fsm.assert_invariants()
    assert "BankrollSolvency violated" in str(exc_info.value)


# ==============================================================================
# 7. FEATURE SYNCHRONIZATION & PICK SIZE INVARIANTS
# ==============================================================================

def test_feature_synchronization_invariant(sample_30_days_draws):
    """
    Verify FeaturesSynchronized: markovState == CALCULATED and zscoreState == COMPUTED
    in FEATURES_READY and subsequent states.
    """
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    assert fsm.markov_state == "CALCULATED"
    assert fsm.zscore_state == "COMPUTED"

    # Corrupt feature state
    fsm._markov_state = "CORRUPTED"
    with pytest.raises(InvariantViolationError) as exc_info:
        fsm.assert_invariants()
    assert "FeaturesSynchronized" in str(exc_info.value)


def test_valid_pick_size_and_duplicate_rejection(sample_30_days_draws):
    """Verify custom predictor returning wrong size or duplicates is rejected."""
    fsm = LotteryPipelineStateMachine()
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    # Predictor returns only 5 numbers
    with pytest.raises(InvariantViolationError):
        fsm.generate_prediction(predictor_func=lambda h: [1, 2, 3, 4, 5])

    # Predictor returns duplicate numbers
    with pytest.raises(InvariantViolationError):
        fsm.generate_prediction(predictor_func=lambda h: [1, 2, 3, 4, 5, 5])


# ==============================================================================
# 8. RESET FUNCTIONALITY
# ==============================================================================

def test_fsm_reset(sample_30_days_draws):
    """Verify reset returns state machine to IDLE with specified or existing bankroll."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000)
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.generate_prediction()
    fsm.evaluate_result([1, 2, 3, 4, 5, 6], prize_payout=20000)

    fsm.reset(initial_bankroll=50000)
    assert fsm.state == PipelineState.IDLE
    assert fsm.bankroll == 50000
    assert fsm.current_round == 1
    assert fsm.picks == []
    assert fsm.loaded_draws == []
    assert fsm.markov_state == "UNINITIALIZED"
    assert fsm.zscore_state == "PENDING"


# ==============================================================================
# 9. FRACTIONAL KELLY CRITERION (compute_fractional_kelly)
# ==============================================================================

import math
from pipeline_fsm import compute_fractional_kelly


def test_fractional_kelly_canonical_calculation():
    """Verify standard Quarter-Kelly stake calculation for positive edge."""
    # bankroll = 100,000, edge = 0.10, odds = 2.0, fraction = 0.25
    # f* = 0.10 / 2.0 = 0.05 -> f = 0.25 * 0.05 = 0.0125 -> stake = 1,250
    stake = compute_fractional_kelly(bankroll=100000.0, edge=0.10, odds=2.0, fraction=0.25)
    assert stake == pytest.approx(1250.0, rel=1e-5)


def test_fractional_kelly_zero_and_negative_edge():
    """Verify zero or negative edge strictly results in 0 allocation (capital preservation)."""
    assert compute_fractional_kelly(bankroll=100000.0, edge=0.0, odds=2.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=-0.05, odds=2.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=-1.0, odds=5.0) == 0.0


def test_fractional_kelly_huge_odds_lottery_scale():
    """Verify huge lottery net odds (e.g. 28,989,675) returns valid small non-negative float."""
    stake = compute_fractional_kelly(
        bankroll=1_000_000_000.0,
        edge=0.05,
        odds=28_989_675.0,
        fraction=0.25,
    )
    assert stake > 0.0
    assert stake <= 1_000_000_000.0 * 0.25
    assert isinstance(stake, float)


def test_fractional_kelly_boundary_conditions():
    """Verify boundary conditions: zero/negative bankroll, zero/negative odds, fraction, or non-finite inputs."""
    assert compute_fractional_kelly(bankroll=0.0, edge=0.10, odds=2.0) == 0.0
    assert compute_fractional_kelly(bankroll=-500.0, edge=0.10, odds=2.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=0.10, odds=0.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=0.10, odds=-1.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=0.10, odds=2.0, fraction=0.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=0.10, odds=2.0, fraction=-0.5) == 0.0
    assert compute_fractional_kelly(bankroll=float("nan"), edge=0.10, odds=2.0) == 0.0
    assert compute_fractional_kelly(bankroll=100000.0, edge=float("inf"), odds=2.0) == 0.0


def test_fractional_kelly_solvency_capping():
    """Verify extreme edge (e.g. 5.0 on odds 1.0) is capped at fraction * bankroll <= bankroll."""
    # f* = 5.0 -> scaled = 0.25 * 5.0 = 1.25 -> capped at fraction (0.25)
    stake = compute_fractional_kelly(bankroll=100000.0, edge=5.0, odds=1.0, fraction=0.25)
    assert stake == pytest.approx(25000.0, rel=1e-5)
    assert stake <= 100000.0


# ==============================================================================
# 10. ARBITRAGE EVALUATION & DORMANT_SKIP LIFECYCLE
# ==============================================================================

def test_fsm_arbitrage_evaluation_transition_to_dormant_skip(sample_30_days_draws):
    """
    Verify FSM transitions FEATURES_READY -> DORMANT_SKIP when arbitrage_mode=True
    and jackpot < threshold_ev. Bankroll preserved, 0 picks emitted.
    """
    fsm = LotteryPipelineStateMachine(
        initial_bankroll=100000,
        ticket_cost=10000,
        threshold_ev=230_000_000_000,
        arbitrage_mode=True,
    )
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()
    assert fsm.state == PipelineState.FEATURES_READY

    # Evaluate arbitrage with sub-threshold jackpot (100 billion < 230 billion)
    res_state = fsm.evaluate_arbitrage(jackpot_value=100_000_000_000)
    assert res_state == PipelineState.DORMANT_SKIP
    assert fsm.state == PipelineState.DORMANT_SKIP
    assert fsm.bankroll == 100000  # Zero debit!
    assert fsm.picks == []
    assert fsm.picks_count == 0


def test_fsm_arbitrage_evaluation_super_threshold_proceeds(sample_30_days_draws):
    """
    Verify FSM remains in FEATURES_READY when jackpot >= threshold_ev,
    and subsequent prediction succeeds with standard debit.
    """
    fsm = LotteryPipelineStateMachine(
        initial_bankroll=100000,
        ticket_cost=10000,
        threshold_ev=230_000_000_000,
        arbitrage_mode=True,
    )
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    # Super-threshold jackpot (250 billion >= 230 billion)
    res_state = fsm.evaluate_arbitrage(jackpot_value=250_000_000_000)
    assert res_state == PipelineState.FEATURES_READY
    assert fsm.state == PipelineState.FEATURES_READY

    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert len(picks) == 6
    assert fsm.bankroll == 90000


def test_fsm_arbitrage_disabled_bypasses_hurdle(sample_30_days_draws):
    """Verify that when arbitrage_mode=False, sub-threshold jackpot does not trigger DORMANT_SKIP."""
    fsm = LotteryPipelineStateMachine(
        initial_bankroll=100000,
        ticket_cost=10000,
        threshold_ev=230_000_000_000,
        arbitrage_mode=False,
    )
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()

    res_state = fsm.evaluate_arbitrage(jackpot_value=50_000_000_000)
    assert res_state == PipelineState.FEATURES_READY
    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert len(picks) == 6


def test_fsm_generate_prediction_behavior_in_dormant_vs_active(sample_30_days_draws):
    """
    Verify generate_prediction() behavior:
    1. Returns [] when already in DORMANT_SKIP without debiting.
    2. Auto-transitions to DORMANT_SKIP if called directly from FEATURES_READY with sub-threshold jackpot.
    """
    # Case 1: Already dormant
    fsm = LotteryPipelineStateMachine(initial_bankroll=50000, arbitrage_mode=True)
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.evaluate_arbitrage(jackpot_value=100_000_000_000)
    assert fsm.state == PipelineState.DORMANT_SKIP

    dormant_picks = fsm.generate_prediction()
    assert dormant_picks == []
    assert fsm.bankroll == 50000
    assert fsm.state == PipelineState.DORMANT_SKIP

    # Case 2: Auto-guard from FEATURES_READY
    fsm2 = LotteryPipelineStateMachine(initial_bankroll=50000, arbitrage_mode=True)
    fsm2.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm2.verify_cutoff()
    fsm2.extract_features()
    fsm2._jackpot_value = 100_000_000_000
    auto_picks = fsm2.generate_prediction()
    assert auto_picks == []
    assert fsm2.bankroll == 50000
    assert fsm2.state == PipelineState.DORMANT_SKIP


def test_fsm_multi_round_dormant_to_active_cycle(sample_30_days_draws):
    """Verify multi-round cycling from DORMANT_SKIP to next round IDLE, then qualified prediction."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=50000, arbitrage_mode=True)

    # Round 1: Dormant Skip
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.evaluate_arbitrage(jackpot_value=100_000_000_000)
    assert fsm.state == PipelineState.DORMANT_SKIP

    # Advance to Round 2
    fsm.next_round()
    assert fsm.state == PipelineState.IDLE
    assert fsm.current_round == 2
    assert fsm.bankroll == 50000

    # Round 2: Qualified prediction
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-12")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.evaluate_arbitrage(jackpot_value=250_000_000_000)
    assert fsm.state == PipelineState.FEATURES_READY
    picks = fsm.generate_prediction()
    assert fsm.state == PipelineState.PREDICTED
    assert fsm.bankroll == 40000


def test_fsm_illegal_transitions_relating_to_dormant_skip(sample_30_days_draws):
    """Verify illegal transitions: evaluate_arbitrage out of order, evaluate_result from DORMANT_SKIP."""
    fsm = LotteryPipelineStateMachine()
    with pytest.raises(InvalidStateTransitionError):
        fsm.evaluate_arbitrage()

    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    with pytest.raises(InvalidStateTransitionError):
        fsm.evaluate_arbitrage()

    fsm.verify_cutoff()
    fsm.extract_features()
    fsm._arbitrage_mode = True
    fsm.evaluate_arbitrage(jackpot_value=100_000_000_000)
    assert fsm.state == PipelineState.DORMANT_SKIP

    with pytest.raises(InvalidStateTransitionError):
        fsm.evaluate_result([1, 2, 3, 4, 5, 6])


# ==============================================================================
# 11. RUNTIME INVARIANT VERIFICATION (assert_invariants)
# ==============================================================================

def test_arbitrage_positive_ev_invariant_violation_detection():
    """Verify assert_invariants() detects ArbitragePositiveEV violation in PREDICTED state."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000, arbitrage_mode=True)
    fsm._state = PipelineState.PREDICTED
    fsm._markov_state = "CALCULATED"
    fsm._zscore_state = "COMPUTED"
    fsm._picks = [1, 2, 3, 4, 5, 6]
    fsm._jackpot_value = 100_000_000_000  # Sub-threshold!
    fsm._threshold_ev = 230_000_000_000

    with pytest.raises(InvariantViolationError) as exc_info:
        fsm.assert_invariants()
    assert "ArbitragePositiveEV violated" in str(exc_info.value)


def test_arbitrage_positive_ev_invariant_violation_in_dormant_skip():
    """Verify assert_invariants() rejects DORMANT_SKIP if arbitrage_mode=False or jackpot >= threshold."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000, arbitrage_mode=False)
    fsm._state = PipelineState.DORMANT_SKIP
    fsm._markov_state = "CALCULATED"
    fsm._zscore_state = "COMPUTED"
    fsm._picks = []
    with pytest.raises(InvariantViolationError) as exc_info:
        fsm.assert_invariants()
    assert "ArbitragePositiveEV violated" in str(exc_info.value)

    fsm._arbitrage_mode = True
    fsm._jackpot_value = 250_000_000_000  # Super-threshold in DORMANT_SKIP
    with pytest.raises(InvariantViolationError) as exc_info:
        fsm.assert_invariants()
    assert "ArbitragePositiveEV violated" in str(exc_info.value)


def test_kelly_solvency_bound_invariant_violation_in_dormant_skip():
    """Verify assert_invariants() rejects non-empty picks, negative bankroll, or out-of-bounds stake in DORMANT_SKIP."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000, arbitrage_mode=True)
    fsm._state = PipelineState.DORMANT_SKIP
    fsm._markov_state = "CALCULATED"
    fsm._zscore_state = "COMPUTED"
    fsm._jackpot_value = 100_000_000_000
    fsm._picks = [1, 2, 3, 4, 5, 6]  # Non-empty in DORMANT_SKIP

    with pytest.raises(InvariantViolationError) as exc_info:
        fsm.assert_invariants()
    assert "ValidPickSize violated" in str(exc_info.value) or "KellySolvencyBound" in str(exc_info.value)

    fsm._picks = []
    fsm._bankroll = -100
    with pytest.raises(BankrollSolvencyError) as exc_info:
        fsm.assert_invariants()
    assert "BankrollSolvency" in str(exc_info.value)

    fsm._bankroll = 100000
    fsm._kelly_stake = 200000.0  # Exceeds bankroll!
    with pytest.raises(InvariantViolationError) as exc_info:
        fsm.assert_invariants()
    assert "KellySolvencyBound violated" in str(exc_info.value)


def test_dormant_skip_preserves_temporal_and_feature_invariants(sample_30_days_draws):
    """Verify DORMANT_SKIP state strictly upholds ZeroFutureLeakage and FeaturesSynchronized."""
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000, arbitrage_mode=True)
    fsm.ingest_data(sample_30_days_draws, cutoff_date="2026-12-11")
    fsm.verify_cutoff()
    fsm.extract_features()
    fsm.evaluate_arbitrage(jackpot_value=100_000_000_000)
    assert fsm.state == PipelineState.DORMANT_SKIP

    # 1. Contaminate with future record
    fsm._loaded_draws.append({"id": "LEAK02", "date": "2026-12-25", "result": [1, 2, 3, 4, 5, 6]})
    with pytest.raises(ZeroFutureLeakageError):
        fsm.assert_invariants()

    # Reset loaded_draws
    fsm._loaded_draws.pop()
    fsm.assert_invariants()

    # 2. Corrupt markov state
    fsm._markov_state = "CORRUPTED"
    with pytest.raises(InvariantViolationError):
        fsm.assert_invariants()

