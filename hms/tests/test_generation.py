"""Parameter generation: window matrices, banded Cholesky and MLPG.

The important property is that `mlpg` agrees with a brute-force dense solve, and
that the window matrix matches the delta definition used in training.  If either
drifts, every synthesised trajectory silently becomes wrong, so these are
checked numerically rather than by eye.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.features import add_dynamic_features, delta_coeffs
from hms.core.generation import (banded_cholesky, banded_solve, mlpg,
                                 stack_state_statistics, stack_streams,
                                 unstack_streams, window_bandwidth,
                                 window_matrix)


def random_banded_spd(n: int, bandwidth: int, rng) -> np.ndarray:
    matrix = np.zeros((n, n))
    for i in range(n):
        for offset in range(1, bandwidth + 1):
            j = i - offset
            if j >= 0:
                value = rng.normal()
                matrix[i, j] = value
                matrix[j, i] = value
    matrix[np.arange(n), np.arange(n)] = bandwidth + 5.0
    return matrix


def pack_lower_band(matrix: np.ndarray, bandwidth: int) -> np.ndarray:
    n = matrix.shape[0]
    packed = np.zeros((n, bandwidth + 1))
    for i in range(n):
        for offset in range(bandwidth + 1):
            if i - offset >= 0:
                packed[i, offset] = matrix[i, i - offset]
    return packed


def unpack_lower_band(packed: np.ndarray, bandwidth: int) -> np.ndarray:
    n = packed.shape[0]
    full = np.zeros((n, n))
    for i in range(n):
        for offset in range(bandwidth + 1):
            if i - offset >= 0:
                full[i, i - offset] = packed[i, offset]
    return full


@pytest.mark.parametrize("n,bandwidth", [(1, 1), (5, 1), (12, 2), (40, 4), (97, 3)])
def test_banded_cholesky_matches_dense_factorisation(n, bandwidth):
    rng = np.random.default_rng(n * 10 + bandwidth)
    matrix = random_banded_spd(n, bandwidth, rng)
    factor = unpack_lower_band(
        banded_cholesky(pack_lower_band(matrix, bandwidth), bandwidth),
        bandwidth)
    assert np.allclose(factor @ factor.T, matrix, atol=1e-8)


@pytest.mark.parametrize("n,bandwidth,dim", [
    (0, 4, 3), (1, 4, 3), (7, 0, 4), (12, 2, 5), (40, 4, 36),
])
def test_batched_banded_cholesky_matches_featurewise_factors(n, bandwidth, dim):
    """The MLPG D-axis batch must match independent packed factors tightly."""
    rng = np.random.default_rng(n * 100 + bandwidth * 10 + dim)
    packed = np.empty((n, bandwidth + 1, dim))
    expected = []
    for d in range(dim):
        matrix = random_banded_spd(n, bandwidth, rng)
        feature_band = pack_lower_band(matrix, bandwidth)
        packed[:, :, d] = feature_band
        expected.append(banded_cholesky(feature_band, bandwidth))

    batched = banded_cholesky(packed, bandwidth)
    assert batched.shape == packed.shape
    assert np.allclose(batched, np.stack(expected, axis=-1), rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("n,bandwidth", [(6, 1), (20, 2), (50, 3)])
def test_banded_solve_matches_dense_solve(n, bandwidth):
    rng = np.random.default_rng(n + bandwidth)
    matrix = random_banded_spd(n, bandwidth, rng)
    rhs = rng.normal(size=n)
    factor = banded_cholesky(pack_lower_band(matrix, bandwidth), bandwidth)
    assert np.allclose(banded_solve(factor, rhs, bandwidth),
                       np.linalg.solve(matrix, rhs), atol=1e-9)


def test_banded_solve_handles_non_positive_definite_gracefully():
    """A degenerate matrix must not produce NaN/inf (it is variance-floored)."""
    factor = banded_cholesky(np.zeros((3, 2)), 1)
    assert np.isfinite(banded_solve(factor, np.ones(3), 1)).all()


def test_batched_banded_cholesky_uses_the_scalar_pivot_floor():
    factor = banded_cholesky(np.zeros((3, 2, 2)), 1)
    assert np.allclose(factor[:, 0, :], 1e-3)  # sqrt(1e-6), as in 2-D


def test_banded_cholesky_rejects_malformed_packed_bands():
    with pytest.raises(ValueError):
        banded_cholesky(np.zeros((3, 1)), 1)          # needs bandwidth + 1 cols
    with pytest.raises(ValueError):
        banded_cholesky(np.zeros(3), 1)


@pytest.mark.parametrize("use_delta2", [False, True])
@pytest.mark.parametrize("n_frames", [3, 12, 40])
def test_window_matrix_reproduces_training_deltas(n_frames, use_delta2):
    rng = np.random.default_rng(n_frames)
    dim = 3
    trajectory = np.cumsum(rng.normal(size=(n_frames, dim)) * 0.1, axis=0)
    stream_sizes = (dim, dim, dim) if use_delta2 else (dim, dim)
    features = add_dynamic_features(trajectory, True, use_delta2, window=2)
    stacked = stack_streams(features, stream_sizes)
    matrix = window_matrix(n_frames, stream_sizes, window=2)
    assert np.allclose(matrix @ trajectory.reshape(-1),
                       stacked.reshape(-1), atol=1e-12)


@pytest.mark.parametrize("use_delta2", [False, True])
def test_mlpg_recovers_the_trajectory_that_generated_the_means(use_delta2):
    rng = np.random.default_rng(11)
    n_frames, dim = 25, 3
    trajectory = np.cumsum(rng.normal(size=(n_frames, dim)) * 0.05, axis=0)
    stream_sizes = (dim, dim, dim) if use_delta2 else (dim, dim)
    stacked = stack_streams(add_dynamic_features(trajectory, True, use_delta2),
                            stream_sizes)
    variances = np.full_like(stacked, 1e-4)
    generated = mlpg(stacked, variances, stream_sizes)
    assert np.allclose(generated, trajectory, atol=1e-6)


@pytest.mark.parametrize("use_delta2", [False, True])
def test_mlpg_matches_a_dense_reference_with_uneven_variances(use_delta2):
    rng = np.random.default_rng(12)
    n_frames, dim = 20, 2
    stream_sizes = (dim, dim, dim) if use_delta2 else (dim, dim)
    means = rng.normal(size=(n_frames * len(stream_sizes), dim))
    variances = np.exp(rng.normal(-1.0, 0.7, size=means.shape))

    matrix = window_matrix(n_frames, stream_sizes, window=2)
    precision = np.diag((1.0 / variances).reshape(-1))
    dense = np.linalg.solve(matrix.T @ precision @ matrix,
                            matrix.T @ precision @ means.reshape(-1))
    dense = dense.reshape(n_frames, dim)
    assert np.allclose(mlpg(means, variances, stream_sizes), dense, atol=1e-8)


def test_mlpg_of_a_single_state_sequence_is_the_mean():
    """With one stream there is nothing to smooth: the means come straight back."""
    rng = np.random.default_rng(13)
    means = rng.normal(size=(10, 4))
    variances = np.ones_like(means)
    assert np.allclose(mlpg(means, variances, (4,), smooth=True), means)


def test_mlpg_smoothing_off_returns_static_means():
    rng = np.random.default_rng(14)
    static = rng.normal(size=(20, 3))
    means = np.concatenate([static, np.zeros_like(static)])   # stream-major
    variances = np.ones_like(means)
    assert np.allclose(mlpg(means, variances, (3, 3), smooth=False), static)


def step_statistics(dim=1, n_frames=40, step_at=20, step=5.0,
                    static_variance=0.5, delta_variance=0.02):
    """Stacked statistics for a sequence that steps once, with flat deltas."""
    static = np.zeros((n_frames, dim))
    static[step_at:] = step
    delta = np.zeros_like(static)
    means = np.concatenate([static, delta])
    variances = np.concatenate([np.full_like(static, static_variance),
                                np.full_like(delta, delta_variance)])
    return means, variances


def test_mlpg_interpolates_between_competing_states():
    """A smooth trajectory must not jump between two distant state means."""
    dim = 1
    means, variances = step_statistics(dim, delta_variance=0.01)
    generated = mlpg(means, variances, (dim, dim))[:, 0]
    steps = np.abs(np.diff(generated))
    # a raw step would be 5.0 in a single frame; MLPG spreads it out
    assert steps.max() < 2.0, "MLPG should spread the jump over several frames"
    interior = ((generated > 0.5) & (generated < 4.5)).sum()
    assert interior >= 8, f"transition only spans {interior} frames"
    assert generated[0] == pytest.approx(0.0, abs=0.5)
    assert generated[-1] == pytest.approx(5.0, abs=0.5)


def test_variance_scale_relaxes_the_dynamic_constraints():
    dim = 1
    means, variances = step_statistics(dim, delta_variance=0.01)
    largest_step = lambda x: np.abs(np.diff(x[:, 0])).max()
    literal = mlpg(means, variances, (dim, dim), variance_scale=0.02)
    normal = mlpg(means, variances, (dim, dim))
    relaxed = mlpg(means, variances, (dim, dim), variance_scale=50.0)
    # a larger scale means weaker delta constraints, so the trajectory is
    # allowed to follow the (stepped) static means more closely
    assert (largest_step(literal) < largest_step(normal)
            < largest_step(relaxed))
    # and the extremes really are different trajectories, not rounding
    assert not np.allclose(relaxed, literal, atol=1e-2)
    with pytest.raises(ValueError):
        mlpg(means, variances, (dim, dim), variance_scale=0.0)


def test_stack_and_unstack_streams_are_inverse():
    rng = np.random.default_rng(16)
    features = rng.normal(size=(7, 6))
    stacked = stack_streams(features, (3, 3))
    assert stacked.shape == (14, 3)
    assert np.allclose(unstack_streams(stacked, 7, 3), features)
    with pytest.raises(ValueError):
        stack_streams(features, (2, 2))


def test_window_bandwidth_follows_the_number_of_streams():
    assert window_bandwidth((3,)) == 0
    assert window_bandwidth((3, 3), window=2) == 4
    assert window_bandwidth((3, 3, 3), window=2) == 8


def test_stack_state_statistics_shapes():
    stream_means = [np.zeros((3, 2)), np.ones((3, 2))]
    stream_variances = [np.full((3, 2), 0.5), np.full((3, 2), 0.25)]
    means, variances = stack_state_statistics(stream_means, stream_variances,
                                              [4, 2, 3])
    assert means.shape == (18, 2)          # (4+2+3) frames x 2 streams
    assert variances.shape == means.shape
    assert np.allclose(means[:9], 0.0)     # static block first
    assert np.allclose(means[9:], 1.0)     # then the delta block
    empty = stack_state_statistics(stream_means, stream_variances, [0, 0, 0])
    assert empty[0].shape == (0, 2)


def test_delta_coefficients_are_antisymmetric_and_sum_to_zero():
    coeffs = delta_coeffs(3)
    assert len(coeffs) == 7
    assert coeffs.sum() == pytest.approx(0.0)
    assert np.allclose(coeffs, -coeffs[::-1])
