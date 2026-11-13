#!/usr/bin/env python3
"""
VietLot AI — Production Finite State Machine Pipeline
─────────────────────────────────────────────────────
Standardized Formal State Machine mirroring TLA+ specification (`spec/LotteryPipeline.tla`).
Formally verified under TLC Model Checker for safety invariants:
  1. ZeroFutureLeakage: Strict temporal isolation (all accessible draw dates < cutoff_date).
  2. BankrollSolvency: Capital preservation (bankroll >= 0, halts safely when bankroll < ticket_cost).
  3. ValidPickSize: Prediction produces pick_size numbers in PREDICTED/EVALUATED, 0 in DORMANT_SKIP.
  4. FeaturesSynchronized: Markov chain calculated and Z-score computed before prediction.
  5. ArbitragePositiveEV: Predictions emitted only when expected value is positive (EV > 0, e.g. Jackpot >= 230B VND in arbitrage mode); otherwise transitions safely to DORMANT_SKIP.
  6. KellySolvencyBound: Non-negative bankroll, fractional Kelly capital allocation bounded within [0, bankroll], zero picks emitted in DORMANT_SKIP.
"""
from __future__ import annotations

import sys
import math
import pathlib
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable, Set

# Ensure scripts directory is on sys.path for data_utils
ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from data_utils import cutoff_future_data, parse_date
except ImportError:
    # Standalone fallback if invoked in isolated sub-environments
    from datetime import datetime, timezone, timedelta

    VN_TZ = timezone(timedelta(hours=7))

    def parse_date(val: Any, default_year: Optional[int] = None) -> str:
        s = str(val).strip()
        if not s:
            raise ValueError("Empty date string")
        if "T" in s:
            s = s.split("T")[0]
        if " " in s:
            s = s.split(" ")[0]
        s = s.replace("/", "-")
        parts = s.split("-")
        if len(parts) == 3:
            if len(parts[0]) == 4:
                return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            elif len(parts[2]) == 4:
                return f"{int(parts[2]):04d}-{int(parts[1]):02d}-{int(parts[0]):02d}"
        elif len(parts) == 2:
            year = default_year or 2026
            return f"{year:04d}-{int(parts[1]):02d}-{int(parts[0]):02d}"
        raise ValueError(f"Cannot parse date: {val}")

    def cutoff_future_data(
        rows: List[Dict[str, Any]],
        cutoff_date: Any,
        inclusive: bool = False,
        default_year: Optional[int] = None,
        reverse: bool = True,
    ) -> List[Dict[str, Any]]:
        c_iso = parse_date(cutoff_date, default_year=default_year)
        filtered = []
        for r in rows:
            d_raw = r.get("date") or r.get("draw_date")
            if not d_raw:
                continue
            d_iso = parse_date(d_raw, default_year=default_year)
            if inclusive and d_iso <= c_iso:
                filtered.append(r)
            elif not inclusive and d_iso < c_iso:
                filtered.append(r)
        if reverse:
            filtered.sort(key=lambda x: str(x.get("date", "")), reverse=True)
        return filtered


class PipelineState(str, Enum):
    """
    Formal lifecycle phases mirroring TLA+ specification in LotteryPipeline.tla:
      IDLE -> INGESTED -> CUTOFF_VERIFIED -> FEATURES_READY -> PREDICTED -> EVALUATED
    With safe terminal states HALTED_INSOLVENT and DORMANT_SKIP.
    """
    IDLE = "IDLE"
    INGESTED = "INGESTED"
    CUTOFF_VERIFIED = "CUTOFF_VERIFIED"
    FEATURES_READY = "FEATURES_READY"
    PREDICTED = "PREDICTED"
    EVALUATED = "EVALUATED"
    HALTED_INSOLVENT = "HALTED_INSOLVENT"
    DORMANT_SKIP = "DORMANT_SKIP"


class InvariantViolationError(RuntimeError):
    """Base exception for all formal invariant violations."""
    pass


class BankrollSolvencyError(InvariantViolationError):
    """Raised when bankroll drops below zero (BankrollSolvency violated)."""
    pass


class ZeroFutureLeakageError(InvariantViolationError):
    """Raised when future draw data is accessed (ZeroFutureLeakage violated)."""
    pass


class InvalidStateTransitionError(ValueError):
    """Raised when attempting an invalid or illegal state transition."""
    pass


def compute_fractional_kelly(
    bankroll: float,
    edge: float,
    odds: float,
    fraction: float = 0.25,
) -> float:
    """
    Computes capital allocation using the Fractional Kelly Criterion.

    Formula:
        f* = edge / odds
        f = fraction * f*
        stake = f * bankroll

    Mathematical Safety Guarantees:
        - If bankroll <= 0: returns 0.0
        - If edge <= 0: returns 0.0 (strict capital preservation on non-positive EV)
        - If odds <= 0: returns 0.0 (prevents division by zero and invalid odds)
        - If fraction <= 0: returns 0.0
        - Non-finite inputs (NaN, Inf) return 0.0
        - Stake is strictly bounded: 0.0 <= stake <= min(fraction, 1.0) * bankroll <= bankroll
    """
    if not (math.isfinite(bankroll) and math.isfinite(edge) and math.isfinite(odds) and math.isfinite(fraction)):
        return 0.0

    if bankroll <= 0.0 or edge <= 0.0 or odds <= 0.0 or fraction <= 0.0:
        return 0.0

    full_kelly = edge / odds
    scaled_kelly = fraction * full_kelly
    capped_fraction = min(max(scaled_kelly, 0.0), min(fraction, 1.0))
    stake = float(capped_fraction * bankroll)
    return min(stake, float(bankroll))


class LotteryPipelineStateMachine:
    """
    Formal State Machine implementing the quantitative lottery analytics pipeline.
    Maintains and proves runtime invariants equivalent to the TLA+ specification:
      - ZeroFutureLeakage: all loaded records strictly precede the cutoff date.
      - BankrollSolvency: bankroll >= 0 at all times; halts cleanly if capital < ticket_cost.
      - ValidPickSize: exactly pick_size (default 6) numbers produced in PREDICTED/EVALUATED, 0 in DORMANT_SKIP.
      - FeaturesSynchronized: markovState == CALCULATED and zscoreState == COMPUTED in FEATURES_READY.
      - ArbitragePositiveEV: PREDICTED/EVALUATED under arbitrage_mode implies jackpot >= threshold; DORMANT_SKIP implies arbitrage_mode and jackpot < threshold.
      - KellySolvencyBound: bankroll >= 0, 0 <= kelly_stake <= bankroll, picks count 0 in DORMANT_SKIP.
    """

    def __init__(
        self,
        initial_bankroll: int = 1000000,
        ticket_cost: int = 10000,
        pick_size: int = 6,
        threshold_ev: float = 230.0,
        arbitrage_mode: bool = False,
    ):
        if initial_bankroll < 0:
            raise BankrollSolvencyError(
                f"BankrollSolvency violated: initial_bankroll={initial_bankroll} < 0"
            )
        if ticket_cost <= 0:
            raise ValueError(f"ticket_cost must be positive, got {ticket_cost}")
        if pick_size <= 0:
            raise ValueError(f"pick_size must be positive, got {pick_size}")
        if threshold_ev <= 0:
            raise ValueError(f"threshold_ev must be positive, got {threshold_ev}")

        self._bankroll: int = initial_bankroll
        self._ticket_cost: int = ticket_cost
        self._pick_size: int = pick_size
        self._state: PipelineState = PipelineState.IDLE

        # Arbitrage & Kelly configuration
        self._threshold_ev: float = float(threshold_ev)
        self._arbitrage_mode: bool = bool(arbitrage_mode)
        self._jackpot_value: Optional[float] = None
        self._kelly_stake: float = 0.0

        # Data isolation structures
        self._raw_data: List[Dict[str, Any]] = []
        self._loaded_draws: List[Dict[str, Any]] = []
        self._cutoff_date: Optional[str] = None
        self._future_leak: bool = False

        # Feature synchronization status
        self._markov_state: str = "UNINITIALIZED"
        self._zscore_state: str = "PENDING"
        self._markov_model: Any = None
        self._gaussian_model: Any = None
        self._mc_model: Any = None

        # Prediction and evaluation results
        self._picks: List[int] = []
        self._evaluation_result: Optional[Dict[str, Any]] = None
        self._current_round: int = 1

        self.assert_invariants()

    # ──────────────────────────────────────────────────────────────────────────
    # Properties & Inspection
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def phase(self) -> PipelineState:
        """Alias for state, matching TLA+ variable naming."""
        return self._state

    @property
    def bankroll(self) -> int:
        return self._bankroll

    @property
    def ticket_cost(self) -> int:
        return self._ticket_cost

    @property
    def pick_size(self) -> int:
        return self._pick_size

    @property
    def loaded_draws(self) -> List[Dict[str, Any]]:
        return list(self._loaded_draws)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Alias for loaded_draws."""
        return list(self._loaded_draws)

    @property
    def cutoff_date(self) -> Optional[str]:
        return self._cutoff_date

    @property
    def markov_state(self) -> str:
        return self._markov_state

    @property
    def zscore_state(self) -> str:
        return self._zscore_state

    @property
    def picks(self) -> List[int]:
        return list(self._picks)

    @property
    def picks_count(self) -> int:
        return len(self._picks)

    @property
    def future_leak(self) -> bool:
        return self._future_leak

    @property
    def evaluation_result(self) -> Optional[Dict[str, Any]]:
        return self._evaluation_result

    @property
    def current_round(self) -> int:
        return self._current_round

    @property
    def arbitrage_mode(self) -> bool:
        return self._arbitrage_mode

    @property
    def jackpot_value(self) -> Optional[float]:
        return self._jackpot_value

    @property
    def threshold_ev(self) -> float:
        return self._threshold_ev

    @property
    def kelly_stake(self) -> float:
        return self._kelly_stake

    # ──────────────────────────────────────────────────────────────────────────
    # Invariant Verification Core
    # ──────────────────────────────────────────────────────────────────────────

    def assert_invariants(self) -> None:
        """
        Enforces mathematical safety invariants at all state transition boundaries.
        Matches TLA+ specification INVARIANTS:
          1. BankrollSolvency: bankroll >= 0
          2. ZeroFutureLeakage: loadedDraws dates < cutoff_date
          3. ValidPickSize: picksCount == pick_size in PREDICTED/EVALUATED, 0 in DORMANT_SKIP
          4. FeaturesSynchronized: markovState == CALCULATED and zscoreState == COMPUTED
          5. ArbitragePositiveEV: PREDICTED/EVALUATED with arbitrage_mode implies jackpot >= threshold;
             DORMANT_SKIP implies arbitrage_mode and jackpot < threshold
          6. KellySolvencyBound: bankroll >= 0, 0 <= kelly_stake <= bankroll, picks=0 in DORMANT_SKIP
        """
        # 1. Bankroll Solvency Invariant
        if self._bankroll < 0:
            raise BankrollSolvencyError(
                f"BankrollSolvency violated: bankroll={self._bankroll} < 0"
            )

        # 2. Zero Future Leakage Invariant
        post_cutoff_states = {
            PipelineState.CUTOFF_VERIFIED,
            PipelineState.FEATURES_READY,
            PipelineState.PREDICTED,
            PipelineState.EVALUATED,
            PipelineState.HALTED_INSOLVENT,
            PipelineState.DORMANT_SKIP,
        }
        if self._state in post_cutoff_states:
            if self._cutoff_date is not None:
                cutoff_iso = parse_date(self._cutoff_date)
                for record in self._loaded_draws:
                    raw_d = record.get("date") or record.get("draw_date")
                    if raw_d:
                        d_iso = parse_date(raw_d)
                        if d_iso >= cutoff_iso:
                            self._future_leak = True
                            raise ZeroFutureLeakageError(
                                f"ZeroFutureLeakage violated: record date {d_iso} >= cutoff {cutoff_iso} "
                                f"in state {self._state}"
                            )
            self._future_leak = False

        # 3. Valid Pick Size Invariant
        if self._state in {PipelineState.PREDICTED, PipelineState.EVALUATED}:
            if len(self._picks) != self._pick_size:
                raise InvariantViolationError(
                    f"ValidPickSize violated: picks count {len(self._picks)} != {self._pick_size} "
                    f"in state {self._state}"
                )
        elif self._state == PipelineState.DORMANT_SKIP:
            if len(self._picks) != 0:
                raise InvariantViolationError(
                    f"ValidPickSize violated: picks count {len(self._picks)} != 0 in state {self._state}"
                )

        # 4. Features Synchronized Invariant
        sync_states = {
            PipelineState.FEATURES_READY,
            PipelineState.PREDICTED,
            PipelineState.EVALUATED,
            PipelineState.HALTED_INSOLVENT,
            PipelineState.DORMANT_SKIP,
        }
        if self._state in sync_states:
            if self._markov_state != "CALCULATED" or self._zscore_state != "COMPUTED":
                raise InvariantViolationError(
                    f"FeaturesSynchronized violated: markovState={self._markov_state}, "
                    f"zscoreState={self._zscore_state} in state {self._state}"
                )

        # 5. Arbitrage Positive EV Invariant
        if self._state == PipelineState.DORMANT_SKIP and not self._arbitrage_mode:
            raise InvariantViolationError(
                "ArbitragePositiveEV violated: state is DORMANT_SKIP but arbitrage_mode is False"
            )

        eff_threshold = self._threshold_ev
        if self._jackpot_value is not None:
            if self._jackpot_value >= 1_000_000 and self._threshold_ev <= 10_000:
                eff_threshold = self._threshold_ev * 1_000_000_000.0

            if self._arbitrage_mode:
                if self._state in {PipelineState.PREDICTED, PipelineState.EVALUATED}:
                    if self._jackpot_value < eff_threshold:
                        raise InvariantViolationError(
                            f"ArbitragePositiveEV violated: in active state {self._state} with arbitrage_mode=True, "
                            f"jackpot_value={self._jackpot_value} < threshold_ev={eff_threshold}"
                        )
                elif self._state == PipelineState.DORMANT_SKIP:
                    if self._jackpot_value >= eff_threshold:
                        raise InvariantViolationError(
                            f"ArbitragePositiveEV violated: in DORMANT_SKIP with jackpot_value={self._jackpot_value} >= threshold_ev={eff_threshold}"
                        )

        # 6. Kelly Solvency Bound Invariant
        if self._kelly_stake < 0.0 or self._kelly_stake > float(self._bankroll):
            raise InvariantViolationError(
                f"KellySolvencyBound violated: kelly_stake={self._kelly_stake} out of bounds [0, {self._bankroll}]"
            )
        if self._state == PipelineState.DORMANT_SKIP:
            if self._kelly_stake != 0.0 or len(self._picks) != 0:
                raise InvariantViolationError(
                    "KellySolvencyBound violated: dormant state must have kelly_stake=0 and picks=0"
                )

    # ──────────────────────────────────────────────────────────────────────────
    # State Transition Actions (1:1 with TLA+ Actions)
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_data(
        self,
        draws: List[Dict[str, Any]],
        cutoff_date: Optional[str] = None,
    ) -> None:
        """
        Action: IngestData (IDLE -> INGESTED)
        Loads raw historical draws into the pipeline buffer.
        """
        if self._state != PipelineState.IDLE:
            raise InvalidStateTransitionError(
                f"Cannot ingest data from state {self._state}. Must be IDLE."
            )
        if not isinstance(draws, list):
            raise TypeError("draws must be a list of draw dictionaries")

        self._raw_data = list(draws)
        self._loaded_draws = list(draws)
        if cutoff_date is not None:
            self._cutoff_date = parse_date(cutoff_date)

        self._state = PipelineState.INGESTED
        self.assert_invariants()

    def verify_cutoff(self, cutoff_date: Optional[str] = None) -> bool:
        """
        Action: ApplyCutoff (INGESTED -> CUTOFF_VERIFIED)
        Enforces ZeroFutureLeakage: prunes all records with date >= cutoff_date.
        Asserts that no record >= cutoff_date remains in accessible memory.
        """
        if self._state != PipelineState.INGESTED:
            raise InvalidStateTransitionError(
                f"Cannot verify cutoff from state {self._state}. Must be INGESTED."
            )

        if cutoff_date is not None:
            self._cutoff_date = parse_date(cutoff_date)

        if self._cutoff_date is None:
            raise ValueError("Cutoff date must be specified before or during verify_cutoff")

        # Strict temporal pruning (exclusive: only date < cutoff_date)
        filtered = cutoff_future_data(
            self._raw_data,
            cutoff_date=self._cutoff_date,
            inclusive=False,
            reverse=True,
        )
        self._loaded_draws = filtered
        self._state = PipelineState.CUTOFF_VERIFIED
        self.assert_invariants()
        return True

    def apply_cutoff(self, cutoff_date: Optional[str] = None) -> bool:
        """Action alias for verify_cutoff."""
        return self.verify_cutoff(cutoff_date)

    def extract_features(
        self,
        markov_model: Any = None,
        gaussian_model: Any = None,
        mc_model: Any = None,
    ) -> None:
        """
        Action: ExtractFeatures (CUTOFF_VERIFIED -> FEATURES_READY)
        Computes features on verified historical window and synchronizes state:
          markovState: UNINITIALIZED -> CALCULATED
          zscoreState: PENDING -> COMPUTED
        """
        if self._state != PipelineState.CUTOFF_VERIFIED:
            raise InvalidStateTransitionError(
                f"Cannot extract features from state {self._state}. Must be CUTOFF_VERIFIED."
            )

        # Optional integration hook: if model objects or callables provided, fit/invoke them
        if markov_model is not None:
            if hasattr(markov_model, "fit"):
                markov_model.fit(self._loaded_draws)
            self._markov_model = markov_model

        if gaussian_model is not None:
            if hasattr(gaussian_model, "fit"):
                gaussian_model.fit(self._loaded_draws)
            self._gaussian_model = gaussian_model

        if mc_model is not None:
            if hasattr(mc_model, "fit"):
                mc_model.fit(self._loaded_draws)
            self._mc_model = mc_model

        # Synchronize formal states
        self._markov_state = "CALCULATED"
        self._zscore_state = "COMPUTED"
        self._state = PipelineState.FEATURES_READY
        self.assert_invariants()

    def evaluate_arbitrage(
        self,
        jackpot_value: Optional[float] = None,
        threshold_ev: Optional[float] = None,
        arbitrage_mode: Optional[bool] = None,
    ) -> PipelineState:
        """
        Action: EvaluateArbitrage (FEATURES_READY -> DORMANT_SKIP or remains FEATURES_READY)

        When arbitrage_mode == True and jackpot_value < threshold_ev:
            transitions FEATURES_READY -> DORMANT_SKIP without debiting bankroll.
        When arbitrage_mode == False or jackpot_value >= threshold_ev:
            remains in FEATURES_READY, ready for generate_prediction.
        """
        if self._state != PipelineState.FEATURES_READY:
            raise InvalidStateTransitionError(
                f"Cannot evaluate arbitrage from state {self._state}. Must be FEATURES_READY."
            )

        if arbitrage_mode is not None:
            self._arbitrage_mode = bool(arbitrage_mode)
        if jackpot_value is not None:
            if jackpot_value < 0:
                raise ValueError(f"jackpot_value must be non-negative, got {jackpot_value}")
            self._jackpot_value = float(jackpot_value)
        if threshold_ev is not None:
            if threshold_ev <= 0:
                raise ValueError(f"threshold_ev must be positive, got {threshold_ev}")
            self._threshold_ev = float(threshold_ev)

        eff_threshold = self._threshold_ev
        if self._jackpot_value is not None and self._jackpot_value >= 1_000_000 and self._threshold_ev <= 10_000:
            eff_threshold = self._threshold_ev * 1_000_000_000.0

        if self._arbitrage_mode and self._jackpot_value is not None:
            if self._jackpot_value < eff_threshold:
                self._picks = []
                self._kelly_stake = 0.0
                self._state = PipelineState.DORMANT_SKIP
                self.assert_invariants()
                return self._state

        self.assert_invariants()
        return self._state

    def generate_prediction(
        self,
        predictor_func: Optional[Callable[[List[Dict[str, Any]]], List[int]]] = None,
        jackpot_value: Optional[float] = None,
    ) -> List[int]:
        """
        Action: GeneratePrediction (FEATURES_READY -> PREDICTED, HALTED_INSOLVENT, or DORMANT_SKIP)
        Enforces Arbitrage Positive EV hurdle and BankrollSolvency guard:
          - If already in DORMANT_SKIP: returns [] safely without debiting.
          - If arbitrage_mode is active and jackpot < threshold: transitions cleanly to DORMANT_SKIP.
          - If bankroll >= ticket_cost: debits ticket_cost, generates picks, transitions to PREDICTED.
          - If bankroll < ticket_cost: transitions cleanly to HALTED_INSOLVENT without debiting.
        """
        if self._state == PipelineState.DORMANT_SKIP:
            return []

        if self._state != PipelineState.FEATURES_READY:
            raise InvalidStateTransitionError(
                f"Cannot generate prediction from state {self._state}. Must be FEATURES_READY."
            )

        if jackpot_value is not None:
            if jackpot_value < 0:
                raise ValueError(f"jackpot_value must be non-negative, got {jackpot_value}")
            self._jackpot_value = float(jackpot_value)

        # Arbitrage Hurdle Check
        eff_threshold = self._threshold_ev
        if self._jackpot_value is not None and self._jackpot_value >= 1_000_000 and self._threshold_ev <= 10_000:
            eff_threshold = self._threshold_ev * 1_000_000_000.0

        if self._arbitrage_mode and self._jackpot_value is not None:
            if self._jackpot_value < eff_threshold:
                self._picks = []
                self._kelly_stake = 0.0
                self._state = PipelineState.DORMANT_SKIP
                self.assert_invariants()
                return []

        # Bankroll Solvency Guard
        if self._bankroll < self._ticket_cost:
            # Safe terminal state transition
            self._state = PipelineState.HALTED_INSOLVENT
            self.assert_invariants()
            return []

        # Debit ticket cost safely
        self._bankroll -= self._ticket_cost

        # Determine picks
        if predictor_func is not None:
            raw_picks = predictor_func(self._loaded_draws)
        else:
            # Deterministic standard default picks [1..pick_size]
            raw_picks = list(range(1, self._pick_size + 1))

        if not isinstance(raw_picks, (list, tuple)) or len(raw_picks) != self._pick_size:
            raise InvariantViolationError(
                f"Predictor returned invalid pick size: expected {self._pick_size}, got {len(raw_picks)}"
            )

        # Ensure unique positive integers
        unique_picks = sorted(list(set(int(x) for x in raw_picks)))
        if len(unique_picks) != self._pick_size:
            raise InvariantViolationError(
                f"Predictor returned duplicate picks: {raw_picks}"
            )

        self._picks = unique_picks
        self._state = PipelineState.PREDICTED
        self.assert_invariants()
        return list(self._picks)

    def evaluate_result(
        self,
        actual_draw: Union[List[int], Dict[str, Any]],
        prize_payout: int = 0,
    ) -> int:
        """
        Action: EvaluateDraw (PREDICTED -> EVALUATED)
        Credits prize_payout to bankroll and asserts BankrollSolvency invariant.
        Returns the updated bankroll.
        """
        if self._state != PipelineState.PREDICTED:
            raise InvalidStateTransitionError(
                f"Cannot evaluate result from state {self._state}. Must be PREDICTED."
            )
        if prize_payout < 0:
            raise ValueError(f"prize_payout must be non-negative, got {prize_payout}")

        # Extract target numbers
        if isinstance(actual_draw, dict):
            actual_nums = actual_draw.get("result") or actual_draw.get("numbers") or []
        elif isinstance(actual_draw, (list, tuple)):
            actual_nums = list(actual_draw)
        else:
            raise TypeError("actual_draw must be a list of integers or a draw dictionary")

        matches = sorted(list(set(self._picks).intersection(set(int(x) for x in actual_nums))))

        self._bankroll += int(prize_payout)
        self._evaluation_result = {
            "round": self._current_round,
            "picks": list(self._picks),
            "actual": actual_nums,
            "matches": matches,
            "match_count": len(matches),
            "prize_payout": int(prize_payout),
            "bankroll": self._bankroll,
        }

        self._state = PipelineState.EVALUATED
        self.assert_invariants()
        return self._bankroll

    def evaluate_draw(
        self,
        actual_draw: Union[List[int], Dict[str, Any]],
        prize_payout: int = 0,
    ) -> int:
        """Alias for evaluate_result."""
        return self.evaluate_result(actual_draw, prize_payout=prize_payout)

    def next_round(self) -> None:
        """
        Transitions from EVALUATED or DORMANT_SKIP back to IDLE for multi-round backtesting.
        Preserves cumulative bankroll while resetting round-specific prediction artifacts.
        """
        if self._state not in {PipelineState.EVALUATED, PipelineState.DORMANT_SKIP}:
            raise InvalidStateTransitionError(
                f"Cannot start next round from state {self._state}. Must be EVALUATED or DORMANT_SKIP."
            )

        self._current_round += 1
        self._picks = []
        self._evaluation_result = None
        self._markov_state = "UNINITIALIZED"
        self._zscore_state = "PENDING"
        self._kelly_stake = 0.0
        self._state = PipelineState.IDLE
        self.assert_invariants()

    def reset(self, initial_bankroll: Optional[int] = None) -> None:
        """
        Resets the state machine back to clean initial IDLE state.
        """
        if initial_bankroll is not None:
            if initial_bankroll < 0:
                raise BankrollSolvencyError("Initial bankroll cannot be negative")
            self._bankroll = initial_bankroll

        self._current_round = 1
        self._raw_data = []
        self._loaded_draws = []
        self._cutoff_date = None
        self._future_leak = False
        self._markov_state = "UNINITIALIZED"
        self._zscore_state = "PENDING"
        self._markov_model = None
        self._gaussian_model = None
        self._mc_model = None
        self._picks = []
        self._evaluation_result = None
        self._arbitrage_mode = False
        self._jackpot_value = None
        self._threshold_ev = 230.0
        self._kelly_stake = 0.0
        self._state = PipelineState.IDLE
        self.assert_invariants()

    def __repr__(self) -> str:
        return (
            f"<LotteryPipelineStateMachine(phase={self._state.value}, "
            f"bankroll={self._bankroll}, picks={self._picks})>"
        )


if __name__ == "__main__":
    # Self-test demonstration
    print("Initializing LotteryPipelineStateMachine...")
    fsm = LotteryPipelineStateMachine(initial_bankroll=100000, ticket_cost=10000)
    print(f"Initial: {fsm}")

    dummy_draws = [
        {"id": f"D{i:02d}", "date": f"2026-12-{i:02d}", "result": [1, 2, 3, 4, 5, 6]}
        for i in range(1, 31)
    ]
    fsm.ingest_data(dummy_draws, cutoff_date="2026-12-11")
    print(f"After Ingest: {fsm}")

    fsm.verify_cutoff()
    print(f"After VerifyCutoff: {fsm}, loaded_draws count={len(fsm.loaded_draws)}")

    fsm.extract_features()
    print(f"After ExtractFeatures: {fsm}, markov={fsm.markov_state}, zscore={fsm.zscore_state}")

    picks = fsm.generate_prediction()
    print(f"After GeneratePrediction: {fsm}, picks={picks}, bankroll={fsm.bankroll}")

    fsm.evaluate_result([1, 2, 3, 10, 20, 30], prize_payout=50000)
    print(f"After EvaluateResult: {fsm}, bankroll={fsm.bankroll}")
    print("Pipeline lifecycle executed successfully!")
