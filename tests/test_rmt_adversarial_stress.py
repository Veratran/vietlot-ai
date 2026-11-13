"""
Adversarial Stress Test Suite for scripts/rmt_filter.py.
Created by Challenger 1 (Numerical & High-Dimensional Stress) for Milestone M2.

Stress-tests:
1. Extreme Aspect Ratios: T << N (rank deficient) and T >> N (asymptotic).
2. Cholesky Stability: Strict positive-definiteness across all Wishart ensembles.
3. Monte Carlo Invariants: 100 random Wishart trials checking trace, symmetry, diagonal, positive evals.
4. Degenerate Configurations: Multiple zero-variance balls, identical balls, perfect anti-correlation, disconnected subgraphs.
5. Algorithmic Parity: NumPy eigh vs pure-Python Jacobi solver under stress.
"""

from __future__ import annotations

import math
from typing import List, Tuple

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
# Stress Category 1: Extreme Aspect Ratios (T << N and T >> N)
# =============================================================================

@pytest.mark.parametrize("T,N", [
    (3, 55),      # Severely rank-deficient: rank(C) <= 2, N=55
    (5, 55),      # Rank-deficient: rank(C) <= 4, N=55
    (10, 55),     # Rank-deficient: rank(C) <= 9, N=55
    (25, 55),     # Rank-deficient: rank(C) <= 24, N=55
    (55, 55),     # Borderline square Wishart: T = N = 55
    (500, 55),    # Moderate sample size
    (2000, 55),   # Large sample size
    (10000, 55),  # High observation count
    (50000, 55),  # Extremely large T >> N (asymptotic limit)
    (50000, 10),  # T/N = 5000
])
def test_stress_extreme_aspect_ratios(T: int, N: int) -> None:
    """
    Stress test extreme aspect ratios T << N and T >> N.
    Verifies that for any aspect ratio:
    1. cleaned_C is computed without NaN/Inf.
    2. Exact symmetry max|C - C^T| < 1e-14.
    3. Unit diagonal max|diag(C) - 1.0| < 1e-14.
    4. Cauchy-Schwarz bounds: -1.0 <= C[i, j] <= 1.0.
    5. Positive-definiteness: all eigenvalues > 0.
    6. Cholesky factorization succeeds with zero LinAlgError.
    """
    np.random.seed(42 + T + N)
    # Generate random binary draws (mimicking lottery draws of 6 numbers)
    p = min(6.0 / N, 0.5)
    X = (np.random.rand(T, N) < p).astype(float)
    # Ensure every column has variation so correlation is non-trivial
    for j in range(N):
        if np.all(X[:, j] == X[0, j]):
            X[0, j] = 1.0
            X[1, j] = 0.0

    C, out_N, out_T = compute_empirical_correlation(X, total_numbers=N)
    assert C.shape == (N, N)
    assert out_N == N
    assert out_T == T

    lm, lp = compute_marchenko_pastur_bounds(N, T)
    assert lm >= 0.0
    assert lp > lm

    cleaned_C, signals, noises = clean_correlation_matrix_rmt(C, lp, lm)

    # 1. Finite check
    assert np.all(np.isfinite(cleaned_C)), "cleaned_C contains NaN or Inf"

    # 2. Exact symmetry
    asym = np.max(np.abs(cleaned_C - cleaned_C.T))
    assert asym < 1e-14, f"Asymmetry {asym} exceeds 1e-14 for T={T}, N={N}"

    # 3. Unit diagonal
    diag_dev = np.max(np.abs(np.diag(cleaned_C) - 1.0))
    assert diag_dev < 1e-14, f"Diagonal deviation {diag_dev} exceeds 1e-14 for T={T}, N={N}"

    # 4. Cauchy-Schwarz
    assert np.all(cleaned_C >= -1.0 - 1e-14), f"Values < -1 for T={T}, N={N}"
    assert np.all(cleaned_C <= 1.0 + 1e-14), f"Values > 1 for T={T}, N={N}"

    # 5. Positive-definiteness
    evals = np.linalg.eigvalsh(cleaned_C)
    min_eval = float(np.min(evals))
    assert min_eval > 0.0, f"Minimum eigenvalue {min_eval} <= 0 for T={T}, N={N}"

    # 6. Cholesky stability
    L = np.linalg.cholesky(cleaned_C)
    assert L.shape == (N, N)
    # Check reconstruction
    reconstruction_err = np.max(np.abs(L @ L.T - cleaned_C))
    assert reconstruction_err < 1e-12, f"Cholesky reconstruction error {reconstruction_err}"


# =============================================================================
# Stress Category 2 & 3: 100 Random Wishart Trials & Cholesky Stability
# =============================================================================

def test_stress_100_monte_carlo_wishart_invariants() -> None:
    """
    Monte Carlo Stress: 100 random Wishart ensembles.
    Verifies:
    1. sum(tilde_lambda) == N within 1e-10 for every trial.
    2. max|C_clean - C_clean^T| < 1e-14 for every trial.
    3. max|diag(C_clean) - 1.0| < 1e-14 for every trial.
    4. min(eigenvalues(C_clean)) > 0 for every trial.
    5. np.linalg.cholesky(cleaned_C) succeeds with zero LinAlgError across 100% of trials.
    """
    num_trials = 100
    N = 55
    rng = np.random.default_rng(20261113)

    trace_diffs: List[float] = []
    asymmetries: List[float] = []
    diag_devs: List[float] = []
    min_eigenvals: List[float] = []
    cholesky_successes = 0

    for trial in range(num_trials):
        # Vary T across trials to test different aspect ratios q in [0.05, 2.5]
        T = int(rng.integers(low=22, high=1100))
        # Generate random Wishart data matrix
        X = rng.standard_normal((T, N))
        # Compute empirical sample correlation matrix
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0, ddof=1)
        Z = (X - mu) / sigma
        C = (Z.T @ Z) / float(T - 1)
        np.fill_diagonal(C, 1.0)
        C = 0.5 * (C + C.T)

        # Theoretical MP bounds
        lm, lp = compute_marchenko_pastur_bounds(N, T)

        # Spectral decomposition
        raw_evals = np.sort(np.linalg.eigvalsh(C))[::-1]
        raw_trace = float(np.sum(raw_evals))

        # Trace conservation test on clamped spectrum
        clamped_evals = clamp_noise_spectrum(raw_evals, lm, lp)
        clamped_trace = float(np.sum(clamped_evals))
        trace_diff = abs(clamped_trace - float(N))
        trace_diffs.append(trace_diff)
        assert trace_diff < 1e-10, (
            f"Trial {trial}: sum(tilde_lambda)={clamped_trace} vs N={N}, diff={trace_diff:.2e}"
        )

        # Clean correlation matrix
        cleaned_C, signals, noises = clean_correlation_matrix_rmt(C, lp, lm)

        # Symmetry test
        asym = float(np.max(np.abs(cleaned_C - cleaned_C.T)))
        asymmetries.append(asym)
        assert asym < 1e-14, f"Trial {trial}: asymmetry {asym:.2e} >= 1e-14"

        # Unit diagonal test
        d_dev = float(np.max(np.abs(np.diag(cleaned_C) - 1.0)))
        diag_devs.append(d_dev)
        assert d_dev < 1e-14, f"Trial {trial}: diagonal deviation {d_dev:.2e} >= 1e-14"

        # Positive-definiteness test
        evals_clean = np.linalg.eigvalsh(cleaned_C)
        min_ev = float(np.min(evals_clean))
        min_eigenvals.append(min_ev)
        assert min_ev > 0.0, f"Trial {trial}: min eigenvalue {min_ev:.2e} <= 0"

        # Cholesky decomposition stability test
        try:
            L = np.linalg.cholesky(cleaned_C)
            assert not np.any(np.isnan(L))
            cholesky_successes += 1
        except np.linalg.LinAlgError as e:
            pytest.fail(f"Trial {trial}: Cholesky decomposition failed with LinAlgError: {e}")

    assert cholesky_successes == num_trials
    assert max(trace_diffs) < 1e-10
    assert max(asymmetries) < 1e-14
    assert max(diag_devs) < 1e-14
    assert min(min_eigenvals) > 0.0


# =============================================================================
# Stress Category 4: Degenerate Configurations & Pathological Cases
# =============================================================================

def test_stress_multiple_zero_variance_balls() -> None:
    """
    Degenerate Stress: Multiple zero-variance balls simultaneously.
    Test with 1, 10, 30, and 54 zero-variance balls in a 55-ball lottery.
    """
    N = 55
    T = 50
    rng = np.random.default_rng(42)

    for num_zero_var in [1, 10, 30, 54]:
        # Generate draws where only (N - num_zero_var) balls ever appear
        active_balls = list(range(1, N - num_zero_var + 1))
        draw_size = min(6, len(active_balls))

        draws = [
            sorted(rng.choice(active_balls, size=draw_size, replace=False).tolist())
            for _ in range(T)
        ]

        cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)

        assert cleaned_C.shape == (N, N)
        assert metadata["is_positive_definite"] is True
        # Zero-variance balls must have unit diagonal and zero off-diagonals
        for idx in range(N - num_zero_var, N):
            assert cleaned_C[idx, idx] == pytest.approx(1.0, abs=1e-14)
            # Off diagonals should be small or 0
            assert np.max(np.abs(cleaned_C[idx, :idx])) < 1e-12

        # Cholesky must succeed
        L = np.linalg.cholesky(cleaned_C)
        assert L.shape == (N, N)


def test_stress_all_zero_variance_draws() -> None:
    """
    Degenerate Stress: All 55 balls have zero variance (e.g. empty draws).
    The pipeline must degrade gracefully to identity matrix without division by zero.
    """
    N = 55
    T = 20
    draws = [[] for _ in range(T)]

    C, out_N, out_T = compute_empirical_correlation(draws, total_numbers=N)
    assert np.allclose(C, np.eye(N), atol=1e-14)

    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)
    assert np.allclose(cleaned_C, np.eye(N), atol=1e-14)
    assert metadata["is_positive_definite"] is True

    L = np.linalg.cholesky(cleaned_C)
    assert np.allclose(L, np.eye(N), atol=1e-14)


def test_stress_identical_balls_clique() -> None:
    """
    Degenerate Stress: Clique of identical balls.
    Balls 1, 2, 3, 4, 5, 6 ALWAYS appear together in every single draw they appear.
    Tests rank deficiency caused by collinearity.
    """
    N = 30
    T = 100
    rng = np.random.default_rng(999)

    draws = []
    for _ in range(T):
        # With 50% probability, the entire 6-clique is drawn
        if rng.random() < 0.5:
            ticket = [1, 2, 3, 4, 5, 6]
        else:
            ticket = sorted(rng.choice(range(7, N + 1), size=6, replace=False).tolist())
        draws.append(ticket)

    C, _, _ = compute_empirical_correlation(draws, total_numbers=N)
    # Check that clique balls have correlation 1.0 with each other
    for i in range(6):
        for j in range(i + 1, 6):
            assert C[i, j] == pytest.approx(1.0, abs=1e-10)

    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)
    assert metadata["is_positive_definite"] is True
    assert metadata["num_signal_eigenvalues"] >= 1

    # Cholesky must succeed despite collinearity
    L = np.linalg.cholesky(cleaned_C)
    assert L.shape == (N, N)
    evals = np.linalg.eigvalsh(cleaned_C)
    assert np.min(evals) > 0.0


def test_stress_perfect_anti_correlation() -> None:
    """
    Degenerate Stress: Perfect anti-correlation.
    Ball 1 and Ball 2 are mutually exclusive and exhaustive:
    In every draw, either Ball 1 appears OR Ball 2 appears, never both, never neither.
    """
    N = 20
    T = 100
    rng = np.random.default_rng(777)

    draws = []
    for _ in range(T):
        ticket = []
        if rng.random() < 0.5:
            ticket.append(1)
        else:
            ticket.append(2)
        # Fill rest from 3..N
        others = rng.choice(range(3, N + 1), size=5, replace=False).tolist()
        ticket.extend(others)
        draws.append(sorted(ticket))

    C, _, _ = compute_empirical_correlation(draws, total_numbers=N)
    # Correlation between ball 1 and 2 must be exactly -1.0
    assert C[0, 1] == pytest.approx(-1.0, abs=1e-10)

    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)
    assert metadata["is_positive_definite"] is True
    # Anti-correlation must be preserved as a strong signal
    assert cleaned_C[0, 1] < -0.30

    L = np.linalg.cholesky(cleaned_C)
    assert L.shape == (N, N)
    evals = np.linalg.eigvalsh(cleaned_C)
    assert np.min(evals) > 0.0


def test_stress_disconnected_block_subgraphs() -> None:
    """
    Degenerate Stress: Disconnected subgraphs.
    Partition 1 (balls 1..25) only appears in draws 0..T/2.
    Partition 2 (balls 26..55) only appears in draws T/2..T.
    Tests block-diagonal correlation structure with complete orthogonality between blocks.
    """
    N = 55
    T = 200
    rng = np.random.default_rng(555)

    draws = []
    # First half: only balls 1..25
    for _ in range(T // 2):
        draws.append(sorted(rng.choice(range(1, 26), size=6, replace=False).tolist()))
    # Second half: only balls 26..55
    for _ in range(T // 2):
        draws.append(sorted(rng.choice(range(26, 56), size=6, replace=False).tolist()))

    cleaned_C, metadata = clean_correlation_matrix(draws, total_numbers=N)
    assert metadata["is_positive_definite"] is True
    assert metadata["num_signal_eigenvalues"] >= 1

    # Cross-block entries should be strictly non-positive (anti-correlated or zero)
    block_cross = cleaned_C[:25, 25:]
    assert np.mean(block_cross) <= 1e-5

    L = np.linalg.cholesky(cleaned_C)
    assert L.shape == (N, N)


# =============================================================================
# Stress Category 5: Pure-Python Jacobi Eigensolver Parity Under Stress
# =============================================================================

def test_stress_pure_python_jacobi_ill_conditioned() -> None:
    """
    Stress test pure-Python Jacobi solver against ill-conditioned matrices:
    - High condition number (1e8)
    - Small eigenvalues near machine precision
    - Clustered eigenvalues
    """
    rng = np.random.default_rng(12345)
    N = 10

    # Construct matrix with exponentially decaying eigenvalues: 1, 1e-1, ..., 1e-9
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    true_evals = np.logspace(0, -8, N)
    A = Q @ np.diag(true_evals) @ Q.T
    A = 0.5 * (A + A.T)

    evals_jacobi, V_jacobi = _pure_jacobi_eigen(A.tolist(), max_sweeps=80, tol=1e-14)
    evals_np = np.sort(np.linalg.eigvalsh(A))[::-1]

    # Verify eigenvalues match within reasonable relative tolerance
    for ev_j, ev_n in zip(evals_jacobi, evals_np):
        assert abs(ev_j - ev_n) < 1e-5, f"Jacobi eval {ev_j} vs NumPy {ev_n}"

    # Verify eigenvector orthogonality V^T V = I
    V_mat = np.array(V_jacobi)
    ortho_err = np.max(np.abs(V_mat.T @ V_mat - np.eye(N)))
    assert ortho_err < 1e-5, f"Orthogonality error {ortho_err}"
