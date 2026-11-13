"""
Unit Test Suite for Gaussian & Cycle Analyzer (Milestone 2 / Requirement R2)
────────────────────────────────────────────────────────────────────────────
Validates:
  1. Gaussian distribution estimation and exact Z-scores (mean(Z) ~= 0, std(Z) ~= 1).
  2. 2-Sigma confidence band classification (OUTLIER_HOT, OUTLIER_COLD, STATIONARY).
  3. Recurrence cycle metrics: mean cycle, standard deviation, lag, cycle Z-score.
  4. Real dataset verification on data/power655.jsonl.
  5. Mathematical edge cases, zero-division protection, and parameter bounds.
"""

import json
import math
import pathlib
import statistics
import pytest

from scripts.gaussian_cycle import GaussianCycleAnalyzer, BandClassificationDict, MetricDict


@pytest.fixture
def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


class TestGaussianCycleInitialization:
    """Tests for class constructor and parameter validation."""

    def test_default_initialization(self):
        analyzer = GaussianCycleAnalyzer()
        assert analyzer.total_numbers == 55
        assert not analyzer.is_fitted
        assert analyzer.total_draws == 0

    def test_custom_pool_sizes(self):
        a45 = GaussianCycleAnalyzer(total_numbers=45)
        assert a45.total_numbers == 45

        a80 = GaussianCycleAnalyzer(total_numbers=80)
        assert a80.total_numbers == 80

    def test_invalid_pool_size_raises(self):
        with pytest.raises(ValueError):
            GaussianCycleAnalyzer(total_numbers=1)
        with pytest.raises(ValueError):
            GaussianCycleAnalyzer(total_numbers=0)
        with pytest.raises(ValueError):
            GaussianCycleAnalyzer(total_numbers=-10)

    def test_access_before_fit_raises(self):
        analyzer = GaussianCycleAnalyzer(total_numbers=55)
        with pytest.raises(RuntimeError):
            _ = analyzer.mean_frequency
        with pytest.raises(RuntimeError):
            _ = analyzer.std_frequency
        with pytest.raises(RuntimeError):
            _ = analyzer.frequencies
        with pytest.raises(RuntimeError):
            _ = analyzer.compute_z_scores()
        with pytest.raises(RuntimeError):
            _ = analyzer.classify_bands()
        with pytest.raises(RuntimeError):
            _ = analyzer.compute_cycle_metrics()


class TestGaussianZScores:
    """Tests for Gaussian frequency analysis and exact Z-scores."""

    def test_uniform_distribution_zero_std(self):
        """When all numbers appear equally often, std is 0 and all Z-scores must be 0.0."""
        # 10 draws, each draw contains all numbers 1..10
        draws = [[i for i in range(1, 11)] for _ in range(10)]
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit(draws)

        assert analyzer.mean_frequency == 10.0
        assert analyzer.std_frequency == 0.0

        z_scores = analyzer.compute_z_scores()
        assert len(z_scores) == 10
        for num in range(1, 11):
            assert z_scores[num] == 0.0

    def test_synthetic_distribution_zscore_properties(self):
        """
        For any non-degenerate frequency distribution:
        mean(Z) must equal 0.0 (within machine epsilon) and
        std(Z) must equal 1.0 (within machine epsilon).
        """
        total_numbers = 55
        # Construct synthetic draws with distinct frequencies: ball i appears i times
        synthetic_draws = []
        for i in range(1, total_numbers + 1):
            for _ in range(i):
                synthetic_draws.append([i])

        analyzer = GaussianCycleAnalyzer(total_numbers=total_numbers).fit(synthetic_draws)
        z_scores = analyzer.compute_z_scores()
        z_values = list(z_scores.values())

        # Exact mathematical verification
        assert len(z_values) == total_numbers
        assert math.isclose(statistics.mean(z_values), 0.0, abs_tol=1e-10)
        assert math.isclose(statistics.stdev(z_values), 1.0, abs_tol=1e-10)

        # Verify exact Z formula Z_i = (x_i - mu) / sigma
        mu = analyzer.mean_frequency
        sigma = analyzer.std_frequency
        for num in range(1, total_numbers + 1):
            expected_z = (num - mu) / sigma
            assert math.isclose(z_scores[num], expected_z, abs_tol=1e-12)

    def test_known_hand_calculated_distribution(self):
        """Verify hand-computed mean, standard deviation, and Z-scores."""
        # Pool size 4: counts: 10, 20, 30, 40
        # draws: ball 1 x 10, ball 2 x 20, ball 3 x 30, ball 4 x 40
        draws = []
        counts = {1: 10, 2: 20, 3: 30, 4: 40}
        max_c = 40
        for step in range(max_c):
            d = [num for num, c in counts.items() if step < c]
            draws.append(d)

        analyzer = GaussianCycleAnalyzer(total_numbers=4).fit(draws)
        # mu = (10 + 20 + 30 + 40) / 4 = 25.0
        assert math.isclose(analyzer.mean_frequency, 25.0, abs_tol=1e-9)

        # variance = [(10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2] / (4 - 1)
        # = [225 + 25 + 25 + 225] / 3 = 500 / 3 ~= 166.6667
        # sigma = sqrt(500/3) ~= 12.909944
        expected_sigma = math.sqrt(500.0 / 3.0)
        assert math.isclose(analyzer.std_frequency, expected_sigma, abs_tol=1e-9)

        z_scores = analyzer.compute_z_scores()
        assert math.isclose(z_scores[1], (10 - 25.0) / expected_sigma, abs_tol=1e-9)
        assert math.isclose(z_scores[2], (20 - 25.0) / expected_sigma, abs_tol=1e-9)
        assert math.isclose(z_scores[3], (30 - 25.0) / expected_sigma, abs_tol=1e-9)
        assert math.isclose(z_scores[4], (40 - 25.0) / expected_sigma, abs_tol=1e-9)


class TestBandClassification:
    """Tests for 2-sigma confidence band classification and partition invariants."""

    def test_classification_bands_and_partition_invariant(self):
        """
        Creates known outliers:
          - Ball 1: count = 200 -> OUTLIER_HOT (Z > +2.0)
          - Balls 2..54: count = 100 -> STATIONARY (-2.0 <= Z <= +2.0)
          - Ball 55: count = 0 -> OUTLIER_COLD (Z < -2.0)
        """
        total = 55
        draws = []
        # Ball 1 appears 200 times
        for _ in range(200):
            draws.append([1])
        # Balls 2..54 appear 100 times
        for _ in range(100):
            draws.append(list(range(2, total)))
        # Ball 55 appears 0 times

        analyzer = GaussianCycleAnalyzer(total_numbers=total).fit(draws)
        bands = analyzer.classify_bands(threshold=2.0)

        assert 1 in bands["outlier_hot"]
        assert 55 in bands["outlier_cold"]
        assert 2 in bands["stationary"]
        assert 54 in bands["stationary"]

        # Invariant 1: Mutually exclusive (disjoint)
        hot_set = set(bands["outlier_hot"])
        cold_set = set(bands["outlier_cold"])
        stat_set = set(bands["stationary"])

        assert len(hot_set & cold_set) == 0
        assert len(hot_set & stat_set) == 0
        assert len(cold_set & stat_set) == 0

        # Invariant 2: Collectively exhaustive (partitions 1..total)
        assert hot_set | cold_set | stat_set == set(range(1, total + 1))

    def test_case_insensitive_and_uppercase_access(self):
        """Verify dictionary wrapper supports both lowercase and uppercase keys."""
        draws = [[1, 2, 3], [1, 2, 4], [1, 5, 6]]
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit(draws)
        bands = analyzer.classify_bands(threshold=1.5)

        assert bands["outlier_hot"] == bands["OUTLIER_HOT"]
        assert bands["outlier_cold"] == bands["OUTLIER_COLD"]
        assert bands["stationary"] == bands["STATIONARY"]
        assert bands.get("OUTLIER_HOT") is not None

    def test_custom_threshold_scaling(self):
        """Tighter threshold (1.0) captures more outliers than 2.0 or 3.0."""
        # Synthetic draws
        synthetic_draws = []
        for i in range(1, 56):
            for _ in range(i):
                synthetic_draws.append([i])

        analyzer = GaussianCycleAnalyzer(total_numbers=55).fit(synthetic_draws)
        bands_tight = analyzer.classify_bands(threshold=1.0)
        bands_standard = analyzer.classify_bands(threshold=2.0)
        bands_wide = analyzer.classify_bands(threshold=3.0)

        # With standard normal-like distribution, threshold 1.0 has more outliers than 2.0
        outliers_tight = len(bands_tight["outlier_hot"]) + len(bands_tight["outlier_cold"])
        outliers_standard = len(bands_standard["outlier_hot"]) + len(bands_standard["outlier_cold"])
        outliers_wide = len(bands_wide["outlier_hot"]) + len(bands_wide["outlier_cold"])

        assert outliers_tight >= outliers_standard
        assert outliers_standard >= outliers_wide


class TestCycleAndLagMetrics:
    """Tests for cycle analysis, interval calculation, lag, and cycle Z-score."""

    def test_hand_calculated_cycle_metrics(self):
        """
        Hand-crafted timeline of 5 draws (W = 5, indices 0..4):
          - Draw 0: [1, 2]
          - Draw 1: [3, 4]
          - Draw 2: [1, 5]
          - Draw 3: [6, 7]
          - Draw 4: [1, 2]
        """
        draws = [
            [1, 2],
            [3, 4],
            [1, 5],
            [6, 7],
            [1, 2],
        ]
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit(draws)
        metrics = analyzer.compute_cycle_metrics()

        # Ball 1: appeared at [0, 2, 4]
        # Intervals: 2 - 0 = 2, 4 - 2 = 2.
        # Mean cycle = 2.0. Std cycle = 0.0.
        # Last draw index = 4. 1-indexed = 5. Current lag = 5 - 5 = 0.
        # Cycle Z-score: (lag - mean) / 1.0 (std=0 fallback) = (0 - 2.0) / 1.0 = -2.0
        m1 = metrics[1]
        assert m1["current_lag"] == 0
        assert math.isclose(m1["mean_cycle"], 2.0, abs_tol=1e-9)
        assert math.isclose(m1["std_cycle"], 0.0, abs_tol=1e-9)
        assert math.isclose(m1["cycle_zscore"], -2.0, abs_tol=1e-9)

        # Ball 2: appeared at [0, 4]
        # Interval: 4 - 0 = 4.
        # Mean cycle = 4.0. Std cycle = 0.0 (1 interval).
        # Last draw = 4 (1-indexed = 5). Current lag = 0.
        # Cycle Z-score = (0 - 4.0) / 1.0 = -4.0.
        m2 = metrics[2]
        assert m2["current_lag"] == 0
        assert math.isclose(m2["mean_cycle"], 4.0, abs_tol=1e-9)
        assert math.isclose(m2["std_cycle"], 0.0, abs_tol=1e-9)
        assert math.isclose(m2["cycle_zscore"], -4.0, abs_tol=1e-9)

        # Ball 3: appeared at [1]
        # Appeared once at index 1 (1-indexed = 2).
        # Current lag: 5 - 2 = 3.
        # Mean cycle = 5.0 (total draws). Std cycle = 0.0.
        # Cycle Z-score = (3 - 5.0) / 1.0 = -2.0.
        m3 = metrics[3]
        assert m3["current_lag"] == 3
        assert math.isclose(m3["mean_cycle"], 5.0, abs_tol=1e-9)
        assert math.isclose(m3["std_cycle"], 0.0, abs_tol=1e-9)

        # Ball 10: never appeared
        # Current lag: 5. Mean cycle: 5.0. Std cycle = 0.0. Cycle Z-score = 0.0.
        m10 = metrics[10]
        assert m10["current_lag"] == 5
        assert math.isclose(m10["mean_cycle"], 5.0, abs_tol=1e-9)
        assert math.isclose(m10["std_cycle"], 0.0, abs_tol=1e-9)
        assert math.isclose(m10["cycle_zscore"], 0.0, abs_tol=1e-9)

    def test_cycle_metrics_variable_intervals(self):
        """
        Test ball with varying intervals to check standard deviation calculation:
        W = 15 draws (0..14).
        Ball 1 appears at indices: 0, 3, 9, 12 (4 appearances).
        Intervals: [3, 6, 3].
        Mean cycle: (3 + 6 + 3) / 3 = 12 / 3 = 4.0.
        Std cycle: sample standard deviation of [3, 6, 3] = sqrt([(3-4)^2 + (6-4)^2 + (3-4)^2] / 2)
                 = sqrt((1 + 4 + 1) / 2) = sqrt(3) ~= 1.7320508.
        Last appearance: 12 (1-indexed = 13).
        Current lag: 15 - 13 = 2.
        Cycle Z-score: (lag - mean) / std = (2 - 4.0) / sqrt(3) ~= -1.1547.
        """
        w = 15
        draws = [[] for _ in range(w)]
        for idx in [0, 3, 9, 12]:
            draws[idx].append(1)

        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit(draws)
        m = analyzer.compute_cycle_metrics()[1]

        assert m["current_lag"] == 2
        assert math.isclose(m["mean_cycle"], 4.0, abs_tol=1e-9)
        expected_std = math.sqrt(3.0)
        assert math.isclose(m["std_cycle"], expected_std, abs_tol=1e-7)
        expected_z = (2 - 4.0) / expected_std
        assert math.isclose(m["cycle_zscore"], expected_z, abs_tol=1e-7)

    def test_metric_aliases_access(self):
        """Verify metric dictionary supports lag, cycle_z_score, and z_cycle aliases."""
        draws = [[1, 2], [3, 4]]
        analyzer = GaussianCycleAnalyzer(total_numbers=5).fit(draws)
        m = analyzer.compute_cycle_metrics()[1]

        assert m["current_lag"] == m["lag"]
        assert m["cycle_zscore"] == m["cycle_z_score"]
        assert m["cycle_zscore"] == m["z_cycle"]
        assert m["mean_cycle"] == m["mu_cycle"]
        assert m["std_cycle"] == m["sigma_cycle"]


class TestRealDatasetVerification:
    """Comprehensive verification using real historical data from data/power655.jsonl."""

    def test_power655_real_dataset_integrity(self, repo_root: pathlib.Path):
        data_file = repo_root / "data" / "power655.jsonl"
        assert data_file.exists(), f"Historical data file not found at {data_file}"

        loaded = []
        with data_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    loaded.append(json.loads(line))

        assert len(loaded) > 1000, f"Expected > 1000 historical draws, got {len(loaded)}"

        # Reverse so draws are strictly chronological (oldest to newest)
        chronological_draws = list(reversed(loaded))
        analyzer = GaussianCycleAnalyzer(total_numbers=55).fit(chronological_draws)

        # 1. Total draws and pool size
        assert analyzer.total_draws == len(loaded)
        assert analyzer.total_numbers == 55

        # 2. Gaussian Z-scores
        z_scores = analyzer.compute_z_scores()
        assert len(z_scores) == 55
        z_vals = list(z_scores.values())

        # Exact standardized distribution properties on real data
        mean_z = statistics.mean(z_vals)
        std_z = statistics.stdev(z_vals)
        assert math.isclose(mean_z, 0.0, abs_tol=1e-10)
        assert math.isclose(std_z, 1.0, abs_tol=1e-10)

        # 3. 2-Sigma Band Classification
        bands = analyzer.classify_bands(threshold=2.0)
        hot = bands["outlier_hot"]
        cold = bands["outlier_cold"]
        stat = bands["stationary"]

        # All 55 numbers accounted for
        assert len(set(hot) | set(cold) | set(stat)) == 55
        assert len(hot) + len(cold) + len(stat) == 55
        # Stationary band should contain the vast majority (~95%)
        assert len(stat) >= 45

        # 4. Cycle & Lag metrics
        cycle_metrics = analyzer.compute_cycle_metrics()
        assert len(cycle_metrics) == 55

        for num in range(1, 56):
            m = cycle_metrics[num]
            assert m["current_lag"] >= 0
            assert m["mean_cycle"] > 0
            assert m["std_cycle"] >= 0
            assert not math.isnan(m["cycle_zscore"])
            assert not math.isinf(m["cycle_zscore"])

        # 5. Summary and analyze methods
        summary = analyzer.get_summary()
        assert summary["total_numbers"] == 55
        assert summary["total_draws"] == len(loaded)
        assert len(summary["band_2sigma"]) == 2
        assert summary["band_2sigma"][0] < summary["mean_frequency"] < summary["band_2sigma"][1]

        report = analyzer.analyze(chronological_draws)
        assert "numbers" in report
        assert len(report["numbers"]) == 55


class TestEdgeCasesAndRobustness:
    """Tests for edge cases: empty data, single draw, out-of-range numbers, malformed records."""

    def test_empty_draws_handling(self):
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit([])
        assert analyzer.total_draws == 0
        assert analyzer.mean_frequency == 0.0
        assert analyzer.std_frequency == 0.0

        z_scores = analyzer.compute_z_scores()
        for num in range(1, 11):
            assert z_scores[num] == 0.0

        metrics = analyzer.compute_cycle_metrics()
        for num in range(1, 11):
            assert metrics[num]["current_lag"] == 0
            assert metrics[num]["cycle_zscore"] == 0.0

    def test_single_draw_handling(self):
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit([[1, 2, 3]])
        assert analyzer.total_draws == 1

        z_scores = analyzer.compute_z_scores()
        assert len(z_scores) == 10
        assert math.isclose(statistics.mean(z_scores.values()), 0.0, abs_tol=1e-10)

        metrics = analyzer.compute_cycle_metrics()
        # Balls 1, 2, 3 were drawn in draw 0 (1-indexed: 1). Lag = 1 - 1 = 0
        assert metrics[1]["current_lag"] == 0
        assert metrics[2]["current_lag"] == 0
        assert metrics[3]["current_lag"] == 0
        # Ball 4 was not drawn. Lag = 1
        assert metrics[4]["current_lag"] == 1

    def test_filter_out_of_bounds_numbers(self):
        # Draw contains valid balls 1, 5 and invalid 0, 100, -5
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit([[0, 1, 5, 100, -5]])
        freqs = analyzer.frequencies
        assert freqs[1] == 1
        assert freqs[5] == 1
        assert freqs[2] == 0
        assert 0 not in freqs
        assert 100 not in freqs

    def test_duplicate_balls_in_single_draw_counted_once(self):
        # Malformed record with duplicate ball 7
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit([[7, 7, 7, 2]])
        freqs = analyzer.frequencies
        assert freqs[7] == 1
        assert freqs[2] == 1

    def test_input_as_dict_objects(self):
        # Supports list of dicts with 'result' or 'nums'
        rows = [
            {"result": [1, 2, 3]},
            {"nums": [1, 4, 5]},
        ]
        analyzer = GaussianCycleAnalyzer(total_numbers=10).fit(rows)
        freqs = analyzer.frequencies
        assert freqs[1] == 2
        assert freqs[2] == 1
        assert freqs[4] == 1
