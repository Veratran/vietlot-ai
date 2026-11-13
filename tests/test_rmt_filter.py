"""
Unit and Integration Test Suite for Random Matrix Theory (RMT) Correlation Filter.
Milestone M2: scripts/rmt_filter.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from scripts.rmt_filter import (
    RMTFilter,
    _pure_jacobi_eigen,
    clamp_noise_spectrum,
    clean_correlation_matrix,
    clean_correlation_matrix_rmt,
    compute_empirical_correlation,
    compute_marchenko_pastur_bounds,
)


# =============================================================================
# Tier 1: Feature Coverage (Core API Contracts)
# =============================================================================

def test_marchenko_pastur_bounds_analytical_values() -> None:
    """
    Verifies that compute_marchenko_pastur_bounds matches analytical values
    for N=55, T=1422, sigma2=1.0.
    """
    N = 55
    T = 1422
    sigma2 = 1.0
    lm, lp = compute_marchenko_pastur_bounds(N, T, sigma2=sigma2)

    assert lm < lp
    # Analytical: q = 55/1422 ≈ 0.038678, sqrt(q) ≈ 0.196667
    # lm = (1 - 0.196667)^2 ≈ 0.64534, lp = (1 + 0.196667)^2 ≈ 1.43201
    assert lm == pytest.approx(0.6453, abs=1e-4)
    assert lp == pytest.approx(1.4320, abs=1e-4)


def test_compute_empirical_correlation_from_ticket_draws() -> None:
    """
    Tests correlation calculation from a list of ticket draws (1-indexed balls).
    """
    np.random.seed(42)
    total_numbers = 55
    T = 100
    # Generate 100 random draws of 6 numbers each
    draws = [
        sorted(np.random.choice(range(1, total_numbers + 1), size=6, replace=False).tolist())
        for _ in range(T)
    ]

    C, N, out_T = compute_empirical_correlation(draws, total_numbers=total_numbers)

    assert C.shape == (total_numbers, total_numbers)
    assert N == total_numbers
    assert out_T == T
    # Diagonal should be strictly 1.0
    assert np.allclose(np.diag(C), 1.0, atol=1e-12)
    # Correlation entries must lie within [-1, 1]
    assert np.all(C >= -1.0 - 1e-12)
    assert np.all(C <= 1.0 + 1e-12)
    # Symmetry
    assert np.max(np.abs(C - C.T)) < 1e-12


def test_compute_empirical_correlation_from_binary_indicator_matrix() -> None:
    """
    Tests correlation calculation when input is already a binary indicator matrix.
    """
    np.random.seed(123)
    T = 50
    N = 45
    # Binary indicator matrix: random bernoulli with p = 6/45
    X = (np.random.rand(T, N) < (6.0 / 45.0)).astype(float)
    # Ensure every column has variance by adding at least one 1 and one 0
    X[0, :] = 1.0
    X[1, :] = 0.0

    C, out_N, out_T = compute_empirical_correlation(X, total_numbers=N)

    assert C.shape == (N, N)
    assert out_N == N
    assert out_T == T
    assert np.allclose(np.diag(C), 1.0, atol=1e-12)
    assert np.max(np.abs(C - C.T)) < 1e-12


def test_noise_clamping_trace_conservation() -> None:
    """
    Tests that clamp_noise_spectrum squashes noise eigenvalues to their mean
    while preserving the exact trace sum(lambda).
    """
    evals = np.array([3.5, 2.2, 1.2, 0.9, 0.8, 0.7, 0.5, 0.2])
    lm = 0.6
    lp = 1.4

    # Signals: > 1.4 -> [3.5, 2.2]
    # Noise: <= 1.4 -> [1.2, 0.9, 0.8, 0.7, 0.5, 0.2]
    # Noise sum = 4.3, count = 6, mean = 4.3 / 6 ≈ 0.71666667
    clamped = clamp_noise_spectrum(evals, lm, lp)

    assert len(clamped) == len(evals)
    # Signals remain untouched
    assert clamped[0] == pytest.approx(3.5, abs=1e-12)
    assert clamped[1] == pytest.approx(2.2, abs=1e-12)
    # Noise eigenvalues clamped to exact mean
    expected_noise_mean = (1.2 + 0.9 + 0.8 + 0.7 + 0.5 + 0.2) / 6.0
    for val in clamped[2:]:
        assert val == pytest.approx(expected_noise_mean, abs=1e-12)
    # Trace conservation
    assert np.sum(clamped) == pytest.approx(np.sum(evals), abs=1e-12)

    # Empty array
    assert len(clamp_noise_spectrum([], lm, lp)) == 0

    # All signals regime
    all_signals = np.array([5.0, 4.0, 3.0])
    clamped_all_signals = clamp_noise_spectrum(all_signals, lm, lambda_plus=2.0)
    assert np.array_equal(clamped_all_signals, all_signals)


def test_clean_correlation_matrix_rmt_returns_and_types() -> None:
    """
    Verifies return signature and types of clean_correlation_matrix_rmt.
    """
    np.random.seed(99)
    N = 20
    # Create valid synthetic correlation matrix
    A = np.random.randn(N, N)
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)

    cleaned_C, signals, noises = clean_correlation_matrix_rmt(
        C, lambda_plus=1.5, lambda_minus=0.5
    )

    assert isinstance(cleaned_C, np.ndarray)
    assert isinstance(signals, list)
    assert isinstance(noises, list)
    assert cleaned_C.shape == (N, N)
    assert len(signals) + len(noises) == N


def test_rmt_filter_class_fit_and_properties() -> None:
    """
    Verifies the RMTFilter class API, fit lifecycle, and property protections.
    """
    np.random.seed(42)
    N = 10
    T = 50
    draws = [
        sorted(np.random.choice(range(1, N + 1), size=4, replace=False).tolist())
        for _ in range(T)
    ]

    rmt = RMTFilter(total_numbers=N, sigma2=1.0)
    # Unfitted access must raise RuntimeError
    with pytest.raises(RuntimeError):
        _ = rmt.correlation_matrix
    with pytest.raises(RuntimeError):
        _ = rmt.cleaned_correlation_matrix
    with pytest.raises(RuntimeError):
        _ = rmt.metadata

    # Fit
    ret = rmt.fit(draws)
    assert ret is rmt
    assert rmt.is_fitted is True

    # Properties
    assert rmt.correlation_matrix.shape == (N, N)
    assert rmt.cleaned_correlation_matrix.shape == (N, N)
    assert isinstance(rmt.metadata, dict)
    assert rmt.metadata["N"] == N
    assert rmt.metadata["T"] == T


# =============================================================================
# Tier 2: Boundary & Corner Cases
# =============================================================================

def test_boundary_t_much_less_than_n() -> None:
    """
    Boundary case: T << N (e.g., T=8, N=55).
    Raw correlation matrix is singular (rank <= 7) with >= 48 zero eigenvalues.
    RMT filtering must lift zero eigenvalues to positive mean, regularizing the matrix.
    """
    np.random.seed(7)
    N = 55
    T = 8
    draws = [
        sorted(np.random.choice(range(1, N + 1), size=6, replace=False).tolist())
        for _ in range(T)
    ]

    lm, lp = compute_marchenko_pastur_bounds(N, T)
    assert lm >= 0.0
    assert lp > lm

    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)

    assert cleaned_C.shape == (N, N)
    assert metadata["is_positive_definite"] is True
    # Verify all eigenvalues of cleaned matrix are strictly positive
    evals = np.linalg.eigvalsh(cleaned_C)
    assert np.min(evals) > 1e-6
    # Matrix rank is fully restored to N
    assert np.linalg.matrix_rank(cleaned_C) == N


def test_boundary_t_much_greater_than_n() -> None:
    """
    Boundary case: T >> N (e.g., T=10000, N=10).
    Bounds tighten sharply around 1.0. Must remain numerically stable.
    """
    N = 10
    T = 10000
    lm, lp = compute_marchenko_pastur_bounds(N, T)

    assert lm < 1.0 < lp
    # q = 10/10000 = 0.001, sqrt(q) ≈ 0.0316
    # lm ≈ 0.9377, lp ≈ 1.0642
    assert lm == pytest.approx(0.9377, abs=1e-3)
    assert lp == pytest.approx(1.0642, abs=1e-3)


def test_corner_pure_wishart_noise_matrix() -> None:
    """
    Corner case: Pure uncorrelated Wishart noise.
    The cleaned correlation matrix should shrink strongly toward the Identity matrix.
    """
    np.random.seed(42)
    N = 20
    T = 1000
    # Independent Bernoulli random variables (no correlation)
    X = (np.random.rand(T, N) < 0.25).astype(float)
    C, _, _ = compute_empirical_correlation(X, total_numbers=N)

    lm, lp = compute_marchenko_pastur_bounds(N, T)
    cleaned_C, signals, noises = clean_correlation_matrix_rmt(C, lp, lm)

    # In pure noise, signals should be empty or near-empty (at most 1 from TW fluctuation)
    assert len(signals) <= 1

    # Distance to identity should shrink significantly
    I = np.eye(N)
    dist_raw = np.linalg.norm(C - I, "fro")
    dist_cleaned = np.linalg.norm(cleaned_C - I, "fro")
    assert dist_cleaned < 0.3 * dist_raw


def test_corner_identical_numbers_perfect_correlation() -> None:
    """
    Corner case: Numbers 1 and 2 always appear together (perfect correlation).
    RMT filter must isolate the co-movement signal and preserve their elevated correlation
    relative to background noise.
    """
    np.random.seed(101)
    N = 20
    T = 200
    X = np.zeros((T, N), dtype=float)
    for t in range(T):
        # Numbers 1 and 2 appear together in 40% of draws (identical co-occurrence)
        if t < int(0.4 * T):
            X[t, 0] = 1.0
            X[t, 1] = 1.0
        # Pick 4 random others
        others = np.random.choice(range(2, N), size=4, replace=False)
        for o in others:
            X[t, o] = 1.0

    C, _, _ = compute_empirical_correlation(X, total_numbers=N)
    assert C[0, 1] == pytest.approx(1.0, abs=1e-7)

    lm, lp = compute_marchenko_pastur_bounds(N, T)
    cleaned_C, signals, noises = clean_correlation_matrix_rmt(C, lp, lm)

    # Must detect at least 1 signal eigenvalue > lp
    assert len(signals) >= 1
    # Correlation between ball 1 and 2 is elevated significantly above noise
    assert cleaned_C[0, 1] >= 0.30
    assert cleaned_C[0, 1] > 5.0 * np.mean(np.abs(cleaned_C[0, 2:]))


def test_corner_disconnected_zero_frequency_number() -> None:
    """
    Corner case: Number 55 is never drawn (zero variance).
    Zero-division guard must activate, setting C_55,55 = 1.0 and off-diagonals to 0.0.
    """
    np.random.seed(55)
    N = 55
    T = 100
    # Draws exclusively from 1..54 (number 55 never drawn)
    draws = [
        sorted(np.random.choice(range(1, N), size=6, replace=False).tolist())
        for _ in range(T)
    ]

    C, _, _ = compute_empirical_correlation(draws, total_numbers=N)

    # Ball 55 index is 54
    assert C[54, 54] == 1.0
    assert np.all(C[54, :54] == 0.0)
    assert np.all(C[:54, 54] == 0.0)

    # Ensure clean_correlation_matrix runs smoothly and preserves positive definiteness
    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)
    assert metadata["is_positive_definite"] is True
    assert cleaned_C[54, 54] == 1.0


def test_corner_minimal_draw_counts_and_validation_errors() -> None:
    """
    Corner case: Edge sample sizes and invalid parameters.
    """
    # T = 2: minimal valid sample size
    draws_min = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    C, N, T = compute_empirical_correlation(draws_min, total_numbers=55)
    assert C.shape == (55, 55)
    assert T == 2

    # T = 1: invalid sample size
    with pytest.raises(ValueError, match="T must be >= 2"):
        compute_empirical_correlation([[1, 2, 3, 4, 5, 6]], total_numbers=55)

    # Empty draws
    with pytest.raises(ValueError, match="draws cannot be empty"):
        compute_empirical_correlation([], total_numbers=55)

    # total_numbers < 2
    with pytest.raises(ValueError, match="total_numbers must be >= 2"):
        compute_empirical_correlation(draws_min, total_numbers=1)

    # Invalid MP bounds params
    with pytest.raises(ValueError, match="N must be >= 1"):
        compute_marchenko_pastur_bounds(0, 100)
    with pytest.raises(ValueError, match="T must be >= 1"):
        compute_marchenko_pastur_bounds(55, 0)
    with pytest.raises(ValueError, match="sigma2 must be > 0.0"):
        compute_marchenko_pastur_bounds(55, 100, sigma2=0.0)

    # Invalid matrix shape for clean_correlation_matrix_rmt
    with pytest.raises(ValueError, match="square matrix"):
        clean_correlation_matrix_rmt(np.zeros((5, 4)), 1.5, 0.5)

    # Non-finite values in matrix
    nan_mat = np.eye(3)
    nan_mat[0, 1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        clean_correlation_matrix_rmt(nan_mat, 1.5, 0.5)

    # Matrix dimension < 2
    with pytest.raises(ValueError, match="dimension must be >= 2"):
        clean_correlation_matrix_rmt(np.array([[1.0]]), 1.5, 0.5)

    # RMTFilter invalid init
    with pytest.raises(ValueError, match="total_numbers must be >= 2"):
        RMTFilter(total_numbers=1)
    with pytest.raises(ValueError, match="sigma2 must be > 0.0"):
        RMTFilter(total_numbers=55, sigma2=-0.5)

    # Indicator matrix as list of binary lists
    indicator_lists = [
        [1 if j < 6 else 0 for j in range(10)],
        [1 if j >= 4 else 0 for j in range(10)],
    ]
    C_list_in, N_list_in, T_list_in = compute_empirical_correlation(indicator_lists, total_numbers=10)
    assert C_list_in.shape == (10, 10)
    assert N_list_in == 10
    assert T_list_in == 2


# =============================================================================
# Tier 3: Mathematical Invariants
# =============================================================================

@pytest.fixture
def sample_draws_data() -> List[List[int]]:
    np.random.seed(2026)
    N = 30
    T = 150
    return [
        sorted(np.random.choice(range(1, N + 1), size=6, replace=False).tolist())
        for _ in range(T)
    ]


def test_invariant_exact_trace_conservation(sample_draws_data: List[List[int]]) -> None:
    """
    Mathematical Invariant 1: sum(tilde_lambda) == sum(lambda) == N.
    """
    N = 30
    C, _, T = compute_empirical_correlation(sample_draws_data, total_numbers=N)
    evals = np.sort(np.linalg.eigvalsh(C))[::-1]
    lm, lp = compute_marchenko_pastur_bounds(N, T)

    clamped = clamp_noise_spectrum(evals, lm, lp)

    trace_orig = float(np.trace(C))
    trace_clamped = float(np.sum(clamped))

    assert math.isclose(trace_clamped, trace_orig, abs_tol=1e-10)
    assert math.isclose(trace_clamped, float(N), abs_tol=1e-10)


def test_invariant_exact_symmetry(sample_draws_data: List[List[int]]) -> None:
    """
    Mathematical Invariant 2: cleaned_C == cleaned_C.T (exact symmetry).
    """
    cleaned_C, _ = clean_correlation_matrix(sample_draws_data, total_numbers=30)
    asymmetry = np.max(np.abs(cleaned_C - cleaned_C.T))
    assert asymmetry < 1e-14


def test_invariant_unit_diagonal(sample_draws_data: List[List[int]]) -> None:
    """
    Mathematical Invariant 3: diag(cleaned_C) == 1.0 identically.
    """
    cleaned_C, _ = clean_correlation_matrix(sample_draws_data, total_numbers=30)
    diag_dev = np.max(np.abs(np.diag(cleaned_C) - 1.0))
    assert diag_dev < 1e-14


def test_invariant_strict_positive_definiteness(sample_draws_data: List[List[int]]) -> None:
    """
    Mathematical Invariant 4: lambda_min(cleaned_C) > 0 (strictly positive-definite).
    """
    cleaned_C, _ = clean_correlation_matrix(sample_draws_data, total_numbers=30)
    evals = np.linalg.eigvalsh(cleaned_C)
    assert np.all(evals > 1e-7)
    assert np.min(evals) > 0.0


def test_invariant_correlation_bounds_cauchy_schwarz(sample_draws_data: List[List[int]]) -> None:
    """
    Mathematical Invariant 5: -1.0 <= cleaned_C[i, j] <= 1.0.
    """
    cleaned_C, _ = clean_correlation_matrix(sample_draws_data, total_numbers=30)
    assert np.all(cleaned_C >= -1.0 - 1e-14)
    assert np.all(cleaned_C <= 1.0 + 1e-14)


# =============================================================================
# Tier 4: Signal Recovery & Real-World Validation
# =============================================================================

def test_planted_signal_recovery_synthetic_wishart() -> None:
    """
    Tier 4: Synthesize a planted correlation signal inside random Wishart noise.
    Verifies that RMT detects the signal eigenvalue and preserves planted correlation
    while suppressing noise off-diagonals.
    """
    np.random.seed(888)
    N = 25
    T = 500
    # Latent factor model: X = F @ B + Noise
    F = np.random.randn(T, 1)  # Common market factor
    # Factor loadings: first 4 numbers load heavily on F (0.8), others load 0.0
    loadings = np.zeros(N)
    loadings[:4] = 0.8
    noise = np.random.randn(T, N) * np.sqrt(1.0 - loadings**2)
    X = F * loadings[np.newaxis, :] + noise

    # Raw correlation
    C = np.corrcoef(X, rowvar=False)
    raw_planted_corr = np.mean([C[i, j] for i in range(4) for j in range(i + 1, 4)])
    raw_noise_corr = np.mean([abs(C[i, j]) for i in range(4, N) for j in range(i + 1, N)])

    lm, lp = compute_marchenko_pastur_bounds(N, T)
    cleaned_C, signals, noises = clean_correlation_matrix_rmt(C, lp, lm)

    # Must detect exactly 1 dominant signal eigenvalue > lp
    assert len(signals) == 1
    assert signals[0] > lp

    # Planted cluster correlation must remain strong
    clean_planted_corr = np.mean([cleaned_C[i, j] for i in range(4) for j in range(i + 1, 4)])
    assert clean_planted_corr >= 0.30

    # Background noise correlation must be suppressed
    clean_noise_corr = np.mean([abs(cleaned_C[i, j]) for i in range(4, N) for j in range(i + 1, N)])
    assert clean_noise_corr < 0.5 * raw_noise_corr


def test_real_world_power655_data_ingestion() -> None:
    """
    Tier 4: Test against real historical Vietlott data (data/power655.jsonl).
    """
    data_path = Path("D:/vietlot-ai/data/power655.jsonl")
    if not data_path.exists():
        pytest.skip("data/power655.jsonl not found")

    with open(data_path, "r", encoding="utf-8") as f:
        draws = [json.loads(line) for line in f if line.strip()]

    assert len(draws) >= 100, f"Expected at least 100 draws, got {len(draws)}"

    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=55)

    assert cleaned_C.shape == (55, 55)
    assert metadata["N"] == 55
    assert metadata["T"] == len(draws)
    assert metadata["is_positive_definite"] is True
    assert metadata["lambda_minus"] > 0.0
    assert metadata["lambda_plus"] > metadata["lambda_minus"]
    assert np.allclose(np.diag(cleaned_C), 1.0, atol=1e-12)
    assert np.max(np.abs(cleaned_C - cleaned_C.T)) < 1e-12


def test_pure_python_jacobi_eigensolver_parity() -> None:
    """
    Tier 4: Validates algorithmic parity of pure-Python Jacobi solver vs NumPy eigh.
    """
    np.random.seed(42)
    N = 12
    # Create symmetric matrix
    A = np.random.randn(N, N)
    C = (A + A.T) / 2.0
    C_list = C.tolist()

    evals_pure, V_pure = _pure_jacobi_eigen(C_list, max_sweeps=50)
    evals_np, V_np = np.linalg.eigh(C)
    evals_np = np.sort(evals_np)[::-1]

    # Edge cases: n=0 and n=1
    assert _pure_jacobi_eigen([]) == ([], [])
    assert _pure_jacobi_eigen([[3.14]]) == ([3.14], [[1.0]])

    # Compare eigenvalues (descending order)
    max_eval_diff = max(abs(p - n) for p, n in zip(evals_pure, evals_np))
    assert max_eval_diff < 1e-6, f"Eigenvalue mismatch: {max_eval_diff}"

    # Verify eigenvector orthogonality V^T V ≈ I
    V_mat = np.array(V_pure)
    ortho_error = np.max(np.abs(V_mat.T @ V_mat - np.eye(N)))
    assert ortho_error < 1e-6, f"Orthogonality error: {ortho_error}"

    # Verify clean_correlation_matrix with force_pure_python=True
    draws = [
        sorted(np.random.choice(range(1, 11), size=4, replace=False).tolist())
        for _ in range(30)
    ]
    cleaned_pure, meta_pure = clean_correlation_matrix(
        draws, total_numbers=10, force_pure_python=True
    )
    assert cleaned_pure.shape == (10, 10)
    assert meta_pure["is_positive_definite"] is True
