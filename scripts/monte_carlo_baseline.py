#!/usr/bin/env python3
"""
VietLot AI — Monte Carlo Baseline Simulator
─────────────────────────────────────────────
Formal null-hypothesis baseline engine inspired by kenchudigital/ML-SixMark-Lab.
Provides scientific benchmarking of predictive lottery models against uniform
random ticket selection, calculating empirical null distributions, prize yields,
bankroll returns, empirical Z-scores, and p-values.

Strict Invariants:
  - Zero Look-Ahead Bias: Only evaluates against supplied test draws.
  - Complete Reproducibility: Deterministic seeding produces 100% reproducible trials.
  - Zero-Dependency Core: Uses Python standard library (random, math, statistics, typing).
"""
import math
import random
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Standard Vietlott prize schedules
POWER_655_PRIZES: Dict[str, int] = {
    "jackpot_1": 30_000_000_000,
    "jackpot_2": 3_000_000_000,
    "match_5": 40_000_000,
    "match_4": 500_000,
    "match_3": 50_000,
    "no_prize": 0,
}

POWER_645_PRIZES: Dict[str, int] = {
    "jackpot": 12_000_000_000,
    "match_5": 10_000_000,
    "match_4": 300_000,
    "match_3": 30_000,
    "no_prize": 0,
}


class MonteCarloBaselineSimulator:
    """
    Monte Carlo Baseline Simulator establishing an empirical null hypothesis (H0):
    Lottery selections cannot systematically outperform independent random ticket selection.

    Adheres to the interface contract specified in PROJECT.md and R3 requirements.
    """

    def __init__(
        self,
        total_numbers: int = 55,
        pick_count: int = 6,
        ticket_price: int = 10000,
        seed: int = 42,
        game_type: Optional[str] = None,
        payout_table: Optional[Dict[str, int]] = None,
    ):
        """
        Initializes the Monte Carlo Baseline Simulator.

        Args:
            total_numbers: Size of the number universe N (55 for Power 6/55, 45 for Mega 6/45).
            pick_count: Number of distinct balls selected per ticket k (default 6).
            ticket_price: Cost per ticket in VNĐ (default 10,000 VNĐ).
            seed: Base random seed for deterministic reproducibility.
            game_type: Explicit lottery game identifier ('power655', 'power645').
                       Inferred from total_numbers if omitted.
            payout_table: Optional custom payout mapping by prize tier name.
        """
        if total_numbers < pick_count or pick_count <= 0:
            raise ValueError(
                f"total_numbers ({total_numbers}) must be greater than or equal to pick_count ({pick_count}) > 0"
            )
        if ticket_price < 0:
            raise ValueError(f"ticket_price cannot be negative, got {ticket_price}")

        self.total_numbers = total_numbers
        self.pick_count = pick_count
        self.ticket_price = ticket_price
        self.seed = seed

        if game_type is not None:
            self.game_type = game_type.lower()
        elif total_numbers == 45:
            self.game_type = "power645"
        else:
            self.game_type = "power655"

        self.payout_table = payout_table
        self._number_pool = list(range(1, self.total_numbers + 1))

    @property
    def theoretical_expected_hits_per_draw(self) -> float:
        """
        Theoretical expected number of matches per ticket under uniform random selection:
        E[H] = k * (k / N) = k^2 / N (for standard lottery where draw count equals pick count).
        For Power 6/55: 6 * 6 / 55 = 36 / 55 ≈ 0.654545...
        For Mega 6/45: 6 * 6 / 45 = 36 / 45 = 0.800000...
        """
        return float(self.pick_count * self.pick_count) / float(self.total_numbers)

    def calculate_ticket_prize(
        self,
        pick: Union[List[int], Set[int]],
        draw_main: Set[int],
        bonus_number: Optional[int] = None,
    ) -> Tuple[int, str]:
        """
        Calculates prize payout and tier classification for a single ticket.

        Args:
            pick: Set or list of selected numbers.
            draw_main: Set of winning main numbers.
            bonus_number: Optional special/bonus ball number (for Power 6/55 Jackpot 2).

        Returns:
            Tuple[int, str]: (payout in VNĐ, tier_name)
        """
        pick_set = set(pick) if not isinstance(pick, set) else pick
        hits = len(pick_set & draw_main)
        bonus_match = (bonus_number in pick_set) if bonus_number is not None else False

        # Custom payout table override
        if self.payout_table is not None:
            tier_name = f"match_{hits}"
            if hits == 5 and bonus_match and "jackpot_2" in self.payout_table:
                tier_name = "jackpot_2"
            elif hits == 6 and "jackpot_1" in self.payout_table:
                tier_name = "jackpot_1"
            elif hits == 6 and "jackpot" in self.payout_table:
                tier_name = "jackpot"
            payout = self.payout_table.get(tier_name, 0)
            return payout, tier_name

        # Built-in Power 6/55 rules
        if self.game_type == "power655":
            if hits == 6:
                return POWER_655_PRIZES["jackpot_1"], "jackpot_1"
            elif hits == 5 and bonus_match:
                return POWER_655_PRIZES["jackpot_2"], "jackpot_2"
            elif hits == 5:
                return POWER_655_PRIZES["match_5"], "match_5"
            elif hits == 4:
                return POWER_655_PRIZES["match_4"], "match_4"
            elif hits == 3:
                return POWER_655_PRIZES["match_3"], "match_3"
            else:
                return 0, "no_prize"

        # Built-in Mega 6/45 rules
        if hits == 6:
            return POWER_645_PRIZES["jackpot"], "jackpot"
        elif hits == 5:
            return POWER_645_PRIZES["match_5"], "match_5"
        elif hits == 4:
            return POWER_645_PRIZES["match_4"], "match_4"
        elif hits == 3:
            return POWER_645_PRIZES["match_3"], "match_3"
        else:
            return 0, "no_prize"

    def _standardize_draws(
        self, actual_draws: List[Any]
    ) -> List[Tuple[Set[int], Optional[int]]]:
        """
        Converts heterogeneous draw representations into standardized (main_set, bonus_num) tuples.
        Supports:
          - List of integers e.g. [1, 2, 3, 4, 5, 6, 7]
          - Dict with 'result', 'draw', or 'numbers' key.
        """
        standardized: List[Tuple[Set[int], Optional[int]]] = []
        for idx, item in enumerate(actual_draws):
            if isinstance(item, dict):
                raw = item.get("result") or item.get("draw") or item.get("numbers")
                if raw is None:
                    raise ValueError(f"Draw dict at index {idx} missing 'result' or 'draw' key: {item}")
            elif isinstance(item, (list, tuple)):
                raw = item
            else:
                raise TypeError(f"Unsupported draw format at index {idx}: {type(item)}")

            if len(raw) < self.pick_count:
                raise ValueError(
                    f"Draw at index {idx} contains {len(raw)} numbers, expected at least {self.pick_count}"
                )

            main_set = set(raw[: self.pick_count])
            bonus = raw[self.pick_count] if len(raw) > self.pick_count else None
            standardized.append((main_set, bonus))

        return standardized

    def run_simulation(
        self,
        actual_draws: List[Any],
        simulations: int = 1000,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes Monte Carlo simulation of independent random tickets against actual lottery draws.

        Args:
            actual_draws: Test sequence of lottery draws (list of lists or dicts).
            simulations: Number of Monte Carlo iterations (supports 1,000 to 10,000+).
            seed: Optional seed override; defaults to self.seed.

        Returns:
            Dict containing empirical null distribution metrics:
              - mean_hits_per_draw, std_hits_per_draw
              - hit_distribution_mean, hit_rate_distribution
              - prize_tier_counts_mean, prize_tier_totals
              - total_cost_per_sim, earnings_mean, earnings_std, earnings_min, earnings_max
              - net_earnings_mean, net_earnings_std, net_earnings_min, net_earnings_max
              - roi_mean, roi_mean_percent, payout_ratio_mean
              - raw simulation vectors: sim_mean_hits, sim_total_hits, sim_earnings, sim_net_earnings
        """
        if not actual_draws:
            raise ValueError("actual_draws list cannot be empty")
        if simulations < 1:
            raise ValueError(f"simulations must be >= 1, got {simulations}")

        base_seed = seed if seed is not None else self.seed
        parsed_draws = self._standardize_draws(actual_draws)
        draws_count = len(parsed_draws)
        cost_per_sim = draws_count * self.ticket_price

        # Vectors for storing per-simulation aggregate metrics
        sim_mean_hits: List[float] = [0.0] * simulations
        sim_total_hits: List[int] = [0] * simulations
        sim_earnings: List[int] = [0] * simulations
        sim_net_earnings: List[int] = [0] * simulations
        sim_roi: List[float] = [0.0] * simulations

        hit_counts_totals: Dict[int, int] = {k: 0 for k in range(self.pick_count + 1)}
        prize_tier_totals: Dict[str, int] = {}

        pool = self._number_pool
        k = self.pick_count

        for s in range(simulations):
            rng = random.Random(base_seed + s)
            s_hits = 0
            s_earnings = 0

            for main_set, bonus in parsed_draws:
                pick = set(rng.sample(pool, k))
                hits = len(pick & main_set)
                s_hits += hits
                hit_counts_totals[hits] = hit_counts_totals.get(hits, 0) + 1

                payout, tier = self.calculate_ticket_prize(pick, main_set, bonus)
                s_earnings += payout
                prize_tier_totals[tier] = prize_tier_totals.get(tier, 0) + 1

            sim_total_hits[s] = s_hits
            mean_hits = s_hits / draws_count
            sim_mean_hits[s] = mean_hits
            sim_earnings[s] = s_earnings
            net_earn = s_earnings - cost_per_sim
            sim_net_earnings[s] = net_earn
            sim_roi[s] = (net_earn / cost_per_sim) if cost_per_sim > 0 else 0.0

        # Empirical Distribution Statistics
        mean_hits_per_draw = float(statistics.fmean(sim_mean_hits))
        std_hits_per_draw = float(statistics.stdev(sim_mean_hits)) if simulations > 1 else 0.0
        hits_mean = float(statistics.fmean(sim_total_hits))
        hits_std = float(statistics.stdev(sim_total_hits)) if simulations > 1 else 0.0

        earnings_mean = float(statistics.fmean(sim_earnings))
        earnings_std = float(statistics.stdev(sim_earnings)) if simulations > 1 else 0.0
        earnings_min = int(min(sim_earnings))
        earnings_max = int(max(sim_earnings))

        net_earnings_mean = float(statistics.fmean(sim_net_earnings))
        net_earnings_std = float(statistics.stdev(sim_net_earnings)) if simulations > 1 else 0.0
        net_earnings_min = int(min(sim_net_earnings))
        net_earnings_max = int(max(sim_net_earnings))

        roi_mean = float(statistics.fmean(sim_roi))
        roi_mean_percent = roi_mean * 100.0
        payout_ratio_mean = (earnings_mean / cost_per_sim) if cost_per_sim > 0 else 0.0

        # Mean tier distributions per simulation
        prize_tier_counts_mean = {
            tier: count / simulations for tier, count in prize_tier_totals.items()
        }
        hit_distribution_mean = {
            num_hits: count / simulations for num_hits, count in hit_counts_totals.items()
        }
        total_tickets_played = simulations * draws_count
        hit_rate_distribution = {
            num_hits: count / total_tickets_played for num_hits, count in hit_counts_totals.items()
        }

        return {
            "game_type": self.game_type,
            "total_numbers": self.total_numbers,
            "pick_count": self.pick_count,
            "simulations": simulations,
            "draws_count": draws_count,
            "ticket_price": self.ticket_price,
            "total_cost_per_sim": cost_per_sim,
            "total_spend_per_sim": cost_per_sim,  # Compatibility alias
            "theoretical_expected_hits_per_draw": round(self.theoretical_expected_hits_per_draw, 6),
            "mean_hits_per_draw": round(mean_hits_per_draw, 6),
            "std_hits_per_draw": round(std_hits_per_draw, 6),
            "hits_mean": round(hits_mean, 4),
            "hits_std": round(hits_std, 4),
            "hit_distribution_mean": hit_distribution_mean,
            "hit_rate_distribution": hit_rate_distribution,
            "prize_tier_counts_mean": prize_tier_counts_mean,
            "prize_tier_totals": prize_tier_totals,
            "earnings_mean": round(earnings_mean, 2),
            "earnings_std": round(earnings_std, 2),
            "earnings_min": earnings_min,
            "earnings_max": earnings_max,
            "reward_mean": round(earnings_mean, 2),  # Compatibility alias
            "reward_std": round(earnings_std, 2),
            "reward_min": earnings_min,
            "reward_max": earnings_max,
            "net_earnings_mean": round(net_earnings_mean, 2),
            "net_earnings_std": round(net_earnings_std, 2),
            "net_earnings_min": net_earnings_min,
            "net_earnings_max": net_earnings_max,
            "net_earn_mean": round(net_earnings_mean, 2),  # Compatibility alias
            "net_earn_std": round(net_earnings_std, 2),
            "roi_mean": round(roi_mean, 6),
            "roi_mean_percent": round(roi_mean_percent, 4),
            "payout_ratio_mean": round(payout_ratio_mean, 6),
            "sim_mean_hits": sim_mean_hits,
            "sim_total_hits": sim_total_hits,
            "sim_earnings": sim_earnings,
            "sim_net_earnings": sim_net_earnings,
        }

    def evaluate_model_advantage(
        self,
        model_picks: List[Any],
        actual_draws: List[Any],
        mc_baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluates actual model predictions against the Monte Carlo null hypothesis baseline.

        Computes:
          - Empirical Z-Scores for hits and earnings: Z = (Metric_model - mu_MC) / sigma_MC
          - Empirical p-values: Proportion of MC simulations matching or exceeding model
          - Hit rate comparison and ROI delta
          - Two-tailed and one-tailed statistical significance flags (95% and 99% levels)

        Args:
            model_picks: List of predicted tickets (one pick of k numbers per draw).
            actual_draws: Matching list of actual lottery draws.
            mc_baseline: Output dictionary from run_simulation().

        Returns:
            Dict containing comparative evaluation results.
        """
        if len(model_picks) != len(actual_draws):
            raise ValueError(
                f"Length mismatch: model_picks has {len(model_picks)} entries, "
                f"actual_draws has {len(actual_draws)}"
            )
        if not model_picks:
            raise ValueError("model_picks cannot be empty")

        parsed_draws = self._standardize_draws(actual_draws)
        draws_count = len(parsed_draws)
        total_cost = draws_count * self.ticket_price

        model_hits_total = 0
        model_earnings = 0
        model_tier_counts: Dict[str, int] = {}
        model_hit_counts: Dict[int, int] = {k: 0 for k in range(self.pick_count + 1)}

        for idx, (pick_item, (main_set, bonus)) in enumerate(zip(model_picks, parsed_draws)):
            if isinstance(pick_item, dict):
                raw_pick = (
                    pick_item.get("picks")
                    or pick_item.get("numbers")
                    or pick_item.get("result")
                )
                if raw_pick is None:
                    raise ValueError(f"Model pick dict at index {idx} missing numbers: {pick_item}")
            elif isinstance(pick_item, (list, tuple, set)):
                raw_pick = pick_item
            else:
                raise TypeError(f"Unsupported model pick format at index {idx}: {type(pick_item)}")

            pick_set = set(raw_pick)
            if len(pick_set) != self.pick_count:
                raise ValueError(
                    f"Model pick at index {idx} has {len(pick_set)} distinct numbers, "
                    f"expected {self.pick_count}"
                )

            hits = len(pick_set & main_set)
            model_hits_total += hits
            model_hit_counts[hits] = model_hit_counts.get(hits, 0) + 1

            payout, tier = self.calculate_ticket_prize(pick_set, main_set, bonus)
            model_earnings += payout
            model_tier_counts[tier] = model_tier_counts.get(tier, 0) + 1

        model_mean_hits = model_hits_total / draws_count
        model_net_earnings = model_earnings - total_cost
        model_roi = (model_net_earnings / total_cost) if total_cost > 0 else 0.0
        model_roi_percent = model_roi * 100.0

        # Baseline Parameters
        mc_mean_hits = float(mc_baseline["mean_hits_per_draw"])
        mc_std_hits = float(mc_baseline["std_hits_per_draw"])
        mc_earnings_mean = float(mc_baseline["earnings_mean"])
        mc_earnings_std = float(mc_baseline["earnings_std"])
        mc_net_earnings_mean = float(mc_baseline.get("net_earnings_mean", mc_earnings_mean - total_cost))
        mc_roi_mean = float(mc_baseline.get("roi_mean", 0.0))

        # Empirical Z-Scores
        z_hits = (
            (model_mean_hits - mc_mean_hits) / mc_std_hits
            if mc_std_hits > 1e-9
            else 0.0
        )
        z_earnings = (
            (model_earnings - mc_earnings_mean) / mc_earnings_std
            if mc_earnings_std > 1e-9
            else 0.0
        )

        # Empirical p-values (proportion of MC simulations >= model metric)
        sim_mean_hits = mc_baseline.get("sim_mean_hits")
        if sim_mean_hits and len(sim_mean_hits) > 0:
            sim_hits_ge = sum(1 for h in sim_mean_hits if h >= model_mean_hits)
            p_value_hits = sim_hits_ge / len(sim_mean_hits)
        else:
            # Fallback to Gaussian tail survival function: 0.5 * erfc(z / sqrt(2))
            p_value_hits = 0.5 * math.erfc(z_hits / math.sqrt(2.0))

        sim_earnings = mc_baseline.get("sim_earnings")
        if sim_earnings and len(sim_earnings) > 0:
            sim_earn_ge = sum(1 for e in sim_earnings if e >= model_earnings)
            p_value_earnings = sim_earn_ge / len(sim_earnings)
        else:
            p_value_earnings = 0.5 * math.erfc(z_earnings / math.sqrt(2.0))

        # Comparative deltas
        hit_rate_delta = model_mean_hits - mc_mean_hits
        earnings_delta = model_earnings - mc_earnings_mean
        net_earnings_delta = model_net_earnings - mc_net_earnings_mean
        roi_delta = model_roi - mc_roi_mean

        better_than_mc_mean = bool(model_mean_hits > mc_mean_hits)
        significant_95 = bool(p_value_hits <= 0.05 or z_hits >= 1.645)
        significant_99 = bool(p_value_hits <= 0.01 or z_hits >= 2.326)

        return {
            "model_metrics": {
                "draws_count": draws_count,
                "total_cost": total_cost,
                "total_hits": model_hits_total,
                "mean_hits_per_draw": round(model_mean_hits, 6),
                "total_earnings": model_earnings,
                "net_earnings": model_net_earnings,
                "roi": round(model_roi, 6),
                "roi_percent": round(model_roi_percent, 4),
                "prize_tier_counts": model_tier_counts,
                "hit_distribution": model_hit_counts,
            },
            "mc_baseline_metrics": {
                "mean_hits_per_draw": mc_mean_hits,
                "std_hits_per_draw": mc_std_hits,
                "earnings_mean": mc_earnings_mean,
                "earnings_std": mc_earnings_std,
                "net_earnings_mean": mc_net_earnings_mean,
                "roi_mean": mc_roi_mean,
            },
            "z_score_hits": round(float(z_hits), 4),
            "z_score_earnings": round(float(z_earnings), 4),
            "z_score": round(float(z_hits), 4),  # Standard Z-Score alias
            "p_value_hits": round(float(p_value_hits), 6),
            "p_value_earnings": round(float(p_value_earnings), 6),
            "p_value": round(float(p_value_hits), 6),  # Standard p-value alias
            "hit_rate_delta": round(float(hit_rate_delta), 6),
            "earnings_delta": round(float(earnings_delta), 2),
            "net_earnings_delta": round(float(net_earnings_delta), 2),
            "roi_delta": round(float(roi_delta), 6),
            "better_than_mc_mean": better_than_mc_mean,
            "statistically_significant_95": significant_95,
            "statistically_significant_99": significant_99,
        }
