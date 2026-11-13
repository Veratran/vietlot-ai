#!/usr/bin/env python3
"""
VietLot AI — Gaussian Z-Score & Cycle Recurrence Engine
────────────────────────────────────────────────────────
Requirement R2: Statistical Gaussian Distribution Testing & Cycle Analysis.
Reference: khiemdoan/vietnam-lottery-xsmb-analysis (src/analyze.py).
Interface Contract: PROJECT.md § Interface Contracts (GaussianCycleAnalyzer).

Key Capabilities:
  1. Gaussian Frequency Analysis:
     - Sample mean (mu) and unbiased sample standard deviation (sigma, ddof=1).
     - Exact Z-Score computation: Z_i = (x_i - mu) / sigma.
     - 2-Sigma confidence band classification:
         * OUTLIER_HOT  : Z_i > +2.0  (overheated frequencies)
         * OUTLIER_COLD : Z_i < -2.0  (undercooled frequencies / "số gan")
         * STATIONARY   : -2.0 <= Z_i <= +2.0 (within 95.45% confidence band)
  2. Cycle & Lag Analysis:
     - Chronological draw index tracking per lottery number.
     - Inter-arrival interval calculation: Delta t = t_{j+1} - t_j.
     - Mean cycle interval (C_bar) and cycle standard deviation (sigma_C).
     - Current overdue lag: lag_i = W - t_{last} (where W = total draws, t_{last} is 1-indexed draw).
     - Cycle anomaly Z-Score: Z_{cycle, i} = (lag_i - C_bar) / sigma_C with zero-division protection.

Zero-dependency standard library implementation: math, statistics, typing, json, collections.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


class BandClassificationDict(dict):
    """
    Case-insensitive dictionary wrapper for 2-sigma band classifications.
    Allows accessing 'outlier_hot', 'OUTLIER_HOT', 'outlier_cold', 'stationary', etc.
    """

    def __getitem__(self, key: str) -> List[int]:
        if isinstance(key, str):
            lower_k = key.lower()
            if lower_k in self:
                return super().__getitem__(lower_k)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(key, str):
            lower_k = key.lower()
            if lower_k in self:
                return super().__getitem__(lower_k)
        return super().get(key, default)


class MetricDict(dict):
    """
    Dictionary wrapper for cycle metrics supporting case-insensitivity and standard aliases.
    E.g. metric['current_lag'], metric['lag'], metric['cycle_zscore'], metric['z_cycle'].
    """

    _ALIASES = {
        "lag": "current_lag",
        "overdue_lag": "current_lag",
        "cycle_zscore": "cycle_zscore",
        "cycle_z_score": "cycle_zscore",
        "z_cycle": "cycle_zscore",
        "zscore_cycle": "cycle_zscore",
        "mean_cycle": "mean_cycle",
        "mu_cycle": "mean_cycle",
        "std_cycle": "std_cycle",
        "sigma_cycle": "std_cycle",
    }

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            k = key.lower()
            if k in self:
                return super().__getitem__(k)
            alias = self._ALIASES.get(k)
            if alias and alias in self:
                return super().__getitem__(alias)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class GaussianCycleAnalyzer:
    """
    Gaussian Distribution Testing and Recurrence Cycle Analyzer.
    Strictly conforms to PROJECT.md § Interface Contracts.
    """

    def __init__(self, total_numbers: int = 55) -> None:
        """
        Initialize the analyzer for a lottery pool of size total_numbers.

        Args:
            total_numbers: The total pool size (e.g. 55 for Power 6/55, 45 for Mega 6/45).
        """
        if total_numbers < 2:
            raise ValueError(f"total_numbers must be >= 2, got {total_numbers}")

        self.total_numbers: int = total_numbers
        self._is_fitted: bool = False
        self._draws_count: int = 0
        self._frequencies: Dict[int, int] = {i: 0 for i in range(1, total_numbers + 1)}
        self._appearances: Dict[int, List[int]] = {i: [] for i in range(1, total_numbers + 1)}

        # Statistical properties
        self._mu: float = 0.0
        self._sigma: float = 0.0
        self._z_scores: Dict[int, float] = {}
        self._cycle_metrics: Dict[int, MetricDict] = {}

    @property
    def is_fitted(self) -> bool:
        """Indicates whether the analyzer has been fitted with historical draw data."""
        return self._is_fitted

    @property
    def total_draws(self) -> int:
        """The total number of historical draws analyzed."""
        return self._draws_count

    @property
    def mean_frequency(self) -> float:
        """Sample mean frequency mu across all numbers in the pool."""
        self._ensure_fitted()
        return self._mu

    @property
    def std_frequency(self) -> float:
        """Sample standard deviation sigma of frequencies across all numbers in the pool."""
        self._ensure_fitted()
        return self._sigma

    @property
    def frequencies(self) -> Dict[int, int]:
        """Dictionary mapping each number to its observed appearance frequency."""
        self._ensure_fitted()
        return dict(self._frequencies)

    @property
    def appearances(self) -> Dict[int, List[int]]:
        """Dictionary mapping each number to chronological draw indices (0-indexed)."""
        self._ensure_fitted()
        return {num: list(indices) for num, indices in self._appearances.items()}

    def _ensure_fitted(self) -> None:
        """Internal guard ensuring the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "GaussianCycleAnalyzer is not fitted yet. Call fit(draws) before accessing statistics."
            )

    @staticmethod
    def _extract_draw_numbers(draw_item: Union[List[int], Tuple[int, ...], Dict[str, Any]]) -> Set[int]:
        """
        Safely extracts an integer set from various draw representations (list, tuple, dict).
        """
        if isinstance(draw_item, dict):
            raw = draw_item.get("result") or draw_item.get("nums") or []
            if isinstance(raw, (list, tuple)):
                return {int(n) for n in raw if isinstance(n, (int, str)) and str(n).isdigit()}
            return set()
        elif isinstance(draw_item, (list, tuple, set)):
            return {int(n) for n in draw_item if isinstance(n, (int, str)) and str(n).isdigit()}
        return set()

    def fit(self, draws: Sequence[Union[List[int], Tuple[int, ...], Dict[str, Any]]]) -> "GaussianCycleAnalyzer":
        """
        Fit the Gaussian and cycle models on historical draws.

        Args:
            draws: Chronological list of historical draws (index 0 is oldest, index -1 is newest).
                   Each item can be a list/tuple of numbers or a dict containing 'result'/'nums'.

        Returns:
            self: The fitted analyzer instance (fluent interface).
        """
        self._draws_count = len(draws)
        self._frequencies = {i: 0 for i in range(1, self.total_numbers + 1)}
        self._appearances = {i: [] for i in range(1, self.total_numbers + 1)}

        # 1. Accumulate frequencies and appearance timelines
        for t_idx, draw in enumerate(draws):
            nums_in_draw = self._extract_draw_numbers(draw)
            for num in nums_in_draw:
                if 1 <= num <= self.total_numbers:
                    self._frequencies[num] += 1
                    self._appearances[num].append(t_idx)

        # 2. Compute sample mean and standard deviation across all total_numbers
        counts = list(self._frequencies.values())
        self._mu = float(statistics.mean(counts))
        if self.total_numbers > 1:
            self._sigma = float(statistics.stdev(counts))
        else:
            self._sigma = 0.0

        # 3. Compute exact Gaussian Z-Scores
        self._z_scores = {}
        for num in range(1, self.total_numbers + 1):
            count = self._frequencies[num]
            if self._sigma > 1e-9:
                z = (count - self._mu) / self._sigma
            else:
                z = 0.0
            self._z_scores[num] = float(z)

        # 4. Compute Cycle & Lag Metrics
        self._cycle_metrics = {}
        w = self._draws_count

        for num in range(1, self.total_numbers + 1):
            app = self._appearances[num]
            k_i = len(app)

            if w == 0:
                current_lag = 0
                mean_cycle = 0.0
                std_cycle = 0.0
                cycle_zscore = 0.0
                max_lag = 0
            elif k_i == 0:
                # Never appeared in the historical window
                current_lag = w
                mean_cycle = float(w)
                std_cycle = 0.0
                cycle_zscore = 0.0
                max_lag = w
            elif k_i == 1:
                # Appeared once: draw index app[0] (0-indexed). 1-indexed draw is app[0] + 1.
                # Lag is elapsed draws since last appearance: w - (app[0] + 1) == (w - 1) - app[0]
                t_last_1_indexed = app[0] + 1
                current_lag = w - t_last_1_indexed
                mean_cycle = float(w)
                std_cycle = 0.0
                # Zero-division protection: fallback denominator 1.0
                cycle_zscore = float(current_lag - mean_cycle) / 1.0 if current_lag != mean_cycle else 0.0
                max_lag = max(app[0] + 1, current_lag)
            else:
                # Appeared 2 or more times: calculate inter-arrival intervals Delta t
                intervals = [app[j + 1] - app[j] for j in range(k_i - 1)]
                mean_cycle = float(statistics.mean(intervals))
                std_cycle = float(statistics.stdev(intervals)) if len(intervals) > 1 else 0.0
                t_last_1_indexed = app[-1] + 1
                current_lag = w - t_last_1_indexed

                if std_cycle > 1e-9:
                    cycle_zscore = float((current_lag - mean_cycle) / std_cycle)
                else:
                    cycle_zscore = float(current_lag - mean_cycle) / 1.0 if current_lag != mean_cycle else 0.0

                max_lag = int(max(max(intervals), current_lag))

            metric = MetricDict({
                "mean_cycle": mean_cycle,
                "std_cycle": std_cycle,
                "current_lag": int(current_lag),
                "cycle_zscore": cycle_zscore,
                # Aliases for interoperability
                "lag": int(current_lag),
                "cycle_z_score": cycle_zscore,
                "z_cycle": cycle_zscore,
                "max_lag": int(max_lag),
                "appearances": int(k_i),
                "frequency": int(self._frequencies[num]),
            })
            self._cycle_metrics[num] = metric

        self._is_fitted = True
        return self

    def compute_z_scores(self) -> Dict[int, float]:
        """
        Computes exact Z-Score = (x_i - mu) / sigma for each number frequency.

        Returns:
            Dict[int, float]: Mapping from number (1..total_numbers) to exact Z-score.
        """
        self._ensure_fitted()
        return dict(self._z_scores)

    def classify_bands(self, threshold: float = 2.0) -> Dict[str, List[int]]:
        """
        Classifies numbers into 2-sigma confidence bands:
          - 'outlier_hot'  : Z_i > +threshold (significantly above 2-sigma)
          - 'outlier_cold' : Z_i < -threshold (significantly below 2-sigma / "số gan")
          - 'stationary'   : -threshold <= Z_i <= +threshold (within confidence band)

        Args:
            threshold: Standard deviation boundary (default: 2.0).

        Returns:
            BandClassificationDict: Dictionary containing 'outlier_hot', 'outlier_cold',
                                   and 'stationary' lists of numbers, sorted ascending.
        """
        self._ensure_fitted()
        hot: List[int] = []
        cold: List[int] = []
        stat: List[int] = []

        for num, z in self._z_scores.items():
            if z > threshold:
                hot.append(num)
            elif z < -threshold:
                cold.append(num)
            else:
                stat.append(num)

        hot.sort()
        cold.sort()
        stat.sort()

        return BandClassificationDict({
            "outlier_hot": hot,
            "outlier_cold": cold,
            "stationary": stat,
            # Uppercase aliases for strict compliance
            "OUTLIER_HOT": hot,
            "OUTLIER_COLD": cold,
            "STATIONARY": stat,
        })

    def compute_cycle_metrics(self) -> Dict[int, Dict[str, Any]]:
        """
        Returns recurrence cycle and lag metrics for all numbers in the pool.

        Returns:
            Dict[int, Dict[str, Any]]: Mapping num -> {
                'mean_cycle': float,
                'std_cycle': float,
                'current_lag': int,
                'cycle_zscore': float
            }
        """
        self._ensure_fitted()
        return dict(self._cycle_metrics)

    def get_summary(self, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Returns a high-level statistical summary of the fitted lottery pool.
        """
        self._ensure_fitted()
        bands = self.classify_bands(threshold=threshold)
        return {
            "total_numbers": self.total_numbers,
            "total_draws": self._draws_count,
            "mean_frequency": round(self._mu, 4),
            "std_frequency": round(self._sigma, 4),
            "band_2sigma": [
                round(self._mu - threshold * self._sigma, 4),
                round(self._mu + threshold * self._sigma, 4),
            ],
            "counts": {
                "outlier_hot": len(bands["outlier_hot"]),
                "outlier_cold": len(bands["outlier_cold"]),
                "stationary": len(bands["stationary"]),
            },
            "outlier_hot": bands["outlier_hot"],
            "outlier_cold": bands["outlier_cold"],
            "stationary": bands["stationary"],
        }

    def analyze(self, draws: Sequence[Union[List[int], Tuple[int, ...], Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Fits the model on draws and returns a consolidated analysis dictionary.
        Complies with Explorer 2's specification.
        """
        self.fit(draws)
        bands = self.classify_bands(threshold=2.0)
        cycle_metrics = self.compute_cycle_metrics()

        number_details = {}
        for num in range(1, self.total_numbers + 1):
            z_val = self._z_scores[num]
            if z_val > 2.0:
                cat = "OUTLIER_HOT"
            elif z_val < -2.0:
                cat = "OUTLIER_COLD"
            else:
                cat = "STATIONARY"

            m = cycle_metrics[num]
            number_details[num] = {
                "number": num,
                "frequency": self._frequencies[num],
                "z_score": round(z_val, 4),
                "band_category": cat,
                "lag": m["current_lag"],
                "mean_cycle": round(m["mean_cycle"], 2),
                "std_cycle": round(m["std_cycle"], 2),
                "max_lag": m["max_lag"],
                "cycle_zscore": round(m["cycle_zscore"], 4),
            }

        return {
            "total_draws": self._draws_count,
            "pool_size": self.total_numbers,
            "mean_frequency": round(self._mu, 2),
            "std_frequency": round(self._sigma, 2),
            "band_2sigma": [
                round(self._mu - 2.0 * self._sigma, 2),
                round(self._mu + 2.0 * self._sigma, 2),
            ],
            "numbers": number_details,
            "outlier_hot": bands["outlier_hot"],
            "outlier_cold": bands["outlier_cold"],
            "stationary": bands["stationary"],
        }


if __name__ == "__main__":
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "power655.jsonl"

    if data_path.exists():
        import json

        loaded_draws = []
        with data_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded_draws.append(json.loads(line))

        # Reverse so chronological: oldest -> newest
        chronological_draws = list(reversed(loaded_draws))
        analyzer = GaussianCycleAnalyzer(total_numbers=55)
        analyzer.fit(chronological_draws)

        summary = analyzer.get_summary()
        print(f"=== VietLot AI Gaussian & Cycle Analysis (Power 6/55) ===")
        print(f"Analyzed {summary['total_draws']} draws across {summary['total_numbers']} numbers.")
        print(f"Mean frequency: {summary['mean_frequency']:.2f}, Std: {summary['std_frequency']:.2f}")
        print(f"2-Sigma Band: [{summary['band_2sigma'][0]:.2f}, {summary['band_2sigma'][1]:.2f}]")
        print(f"Outlier HOT  (> +2.0): {summary['outlier_hot']}")
        print(f"Outlier COLD (< -2.0): {summary['outlier_cold']}")
        print(f"Stationary count: {summary['counts']['stationary']}")

        # Top 5 most overdue ("số gan")
        metrics = analyzer.compute_cycle_metrics()
        sorted_by_lag = sorted(metrics.items(), key=lambda item: item[1]["current_lag"], reverse=True)
        print("\nTop 5 Most Overdue Numbers (Số Gan):")
        for num, m in sorted_by_lag[:5]:
            print(
                f"  Ball {num:02d}: Lag = {m['current_lag']:2d} draws | "
                f"Mean cycle = {m['mean_cycle']:5.2f} | Std cycle = {m['std_cycle']:5.2f} | "
                f"Cycle Z-score = {m['cycle_zscore']:+5.2f}"
            )
