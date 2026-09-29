"""Parameter generation: state sequence -> smooth acoustic trajectories.

The model gives a mean vector and a variance for every frame of the *static*
features and of their deltas.  Sampling each frame independently would give a
discontinuous, buzzy trajectory; the standard fix from HMM-based speech
synthesis is **maximum-likelihood parameter generation (MLPG)**: pick the
trajectory ``c`` that maximises the likelihood of the stacked static+delta
sequence,

    c = argmax_c (W c - mu)^T R^{-1} (W c - mu)
      = (W^T R^{-1} W)^{-1} W^T R^{-1} mu

with ``W`` the window matrix that maps a trajectory to static + delta streams,
``mu``/``R`` the per-frame means and (diagonal) covariances from the model, and
``R^{-1}`` weighting each frame by how confident the model is.

``W`` is built to reproduce `hms.core.features.add_dynamic_features` exactly
(same window coefficients, same edge replication), so generation inverts
training.

Efficiency
----------
``W^T R^{-1} W`` is symmetric and banded, with bandwidth
``2 * window * (n_streams - 1)``.  `banded_cholesky` / `banded_solve` work on a
packed band, giving an O(T * bandwidth^2) solve per feature dimension (a few
million operations for a whole song) instead of an O(T^3) dense solve, and
without the T^2 memory of a dense matrix.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from hms.core.features import delta_coeffs

#: Multiplies the model variance before the solve.  >1 relaxes the constraint
#: (smoother trajectory, means followed more loosely); the standard "variance
#: relaxation" knob from HMM-based synthesis.
DEFAULT_VARIANCE_SCALE = 1.0

#: Floor for variances entering the precision matrix.
VARIANCE_FLOOR = 1e-8


def window_bandwidth(stream_sizes: Sequence[int], window: int = 2) -> int:
    """Lower bandwidth of ``W^T R^{-1} W``."""
    return max(0, 2 * int(window) * (len(tuple(stream_sizes)) - 1))


def stack_streams(features: np.ndarray, stream_sizes: Sequence[int]
                  ) -> np.ndarray:
    """Training layout (T, sum(stream_sizes)) -> MLPG layout (T * S, D).

    `hms.core.features.add_dynamic_features` concatenates the dynamic streams
    along the *feature* axis, while `mlpg`'s window matrix lists complete
    streams one after another down the *row* axis.  This converts between them.
    """
    features = np.atleast_2d(np.asarray(features, dtype=np.float64))
    n_frames, total = features.shape
    stream_sizes = tuple(int(s) for s in stream_sizes)
    dim = stream_sizes[0]
    if any(s != dim for s in stream_sizes) or total != dim * len(stream_sizes):
        raise ValueError(f"feature matrix {features.shape} does not match "
                         f"stream_sizes {stream_sizes}")
    return features.reshape(n_frames, len(stream_sizes), dim) \
                   .transpose(1, 0, 2).reshape(-1, dim)


def unstack_streams(stacked: np.ndarray, n_frames: int, dim: int) -> np.ndarray:
    """Inverse of `stack_streams`: (T * S, D) -> (T, S * D)."""
    stacked = np.atleast_2d(np.asarray(stacked, dtype=np.float64))
    n_streams = stacked.shape[0] // n_frames
    return stacked.reshape(n_streams, n_frames, dim) \
                  .transpose(1, 0, 2).reshape(n_frames, -1)


def stream_taps(n_frames: int, derivative: int, window: int
                ) -> List[Tuple[np.ndarray, float]]:
    """Non-zero entries of one derivative stream's rows of ``W``.

    Returns a list of ``(target_frames, weight)`` pairs: row ``t`` of that
    stream has ``W[t, target_frames_p[t]] += weight_p``.  The index clipping
    reproduces `hms.core.features.add_dynamic_features` exactly, including the
    *double* clipping that makes a delta-delta differ from a plain convolution
    at utterance boundaries.
    """
    if derivative < 1:
        raise ValueError("derivative must be >= 1")
    coeffs = delta_coeffs(window)
    taps = np.arange(2 * window + 1)
    frames = np.arange(n_frames)

    def clip(idx: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(idx, 0), n_frames - 1)

    pairs: List[Tuple[np.ndarray, float]] = [(clip(frames + k - window), coeffs[k])
                                            for k in taps]
    for _ in range(derivative - 1):
        composed: List[Tuple[np.ndarray, float]] = []
        for target, weight in pairs:
            for k in taps:
                composed.append((clip(target + k - window), weight * coeffs[k]))
        pairs = composed
    return pairs


def window_matrix(n_frames: int, stream_sizes: Sequence[int],
                  window: int = 2) -> np.ndarray:
    """Dense ``W``, shape (n_frames * n_streams * D, n_frames * D).

    Provided for tests and documentation; `mlpg` never builds it.
    """
    stream_sizes = tuple(int(s) for s in stream_sizes)
    n_streams = len(stream_sizes)
    if n_streams < 1:
        raise ValueError("stream_sizes must be non-empty")
    dim = stream_sizes[0]
    if any(s != dim for s in stream_sizes):
        raise ValueError("stream_sizes must all be equal (one per dynamic stream)")
    rows = n_frames * n_streams * dim
    cols = n_frames * dim
    w = np.zeros((rows, cols), dtype=np.float64)
    w[:cols] = np.eye(cols)                      # static block
    row = cols
    for derivative in range(1, n_streams):        # derivative blocks
        frame_rows = (row + np.arange(n_frames)[:, None] * dim
                      + np.arange(dim)[None, :])
        for target, weight in stream_taps(n_frames, derivative, window):
            cols_idx = target[:, None] * dim + np.arange(dim)[None, :]
            # np.add.at: taps collide at the utterance boundaries
            np.add.at(w, (frame_rows, cols_idx), weight)
        row += n_frames * dim
    return w


def banded_cholesky(a_band: np.ndarray, bandwidth: int) -> np.ndarray:
    """Cholesky factor of a symmetric positive definite banded matrix.

    ``a_band`` has shape ``(n, bandwidth + 1)`` with
    ``a_band[i, d] = A[i - d, i]`` (LAPACK "lower" packed band storage).
    Returns ``L`` in the same layout with ``L[i, d] = L_factor[i - d, i]``.

    MLPG may also pass independent feature dimensions together as
    ``(n, bandwidth + 1, D)``.  Each final-axis slice uses the same packed
    representation and is factored independently.  Keeping that axis together
    lets the short band operations run in NumPy across all features instead of
    repeating the Python loops once per feature.
    """
    a_band = np.asarray(a_band, dtype=np.float64)
    if a_band.ndim not in (2, 3) or a_band.shape[1] != bandwidth + 1:
        raise ValueError(f"packed band must have {bandwidth + 1} columns, "
                         f"got {a_band.shape}")
    n = a_band.shape[0]
    lower = np.zeros_like(a_band)

    if a_band.ndim == 2:
        # Preserve the scalar packed-band path for callers using the original
        # public representation.  The batched path below is for MLPG's D-axis.
        for i in range(n):
            max_d = min(bandwidth, i)
            # Walk left-to-right inside the row (descending offset): L[i, j]
            # needs the already-computed entries further left in the same row.
            for d in range(max_d, -1, -1):
                j = i - d
                value = a_band[i, d]
                # subtract sum_k L[i, k] * L[j, k] over the columns k < j
                # where both factors are inside the band (k >= i - bandwidth)
                for k in range(max(0, i - bandwidth), j):
                    value -= lower[i, i - k] * lower[j, j - k]
                if i == j:
                    lower[i, 0] = np.sqrt(value) if value > 0 else 1e-6
                else:
                    lower[i, d] = value / lower[j, 0]
        return lower

    # The factor entries in one packed row still depend on each other, so the
    # frame and (tiny) bandwidth loops remain serial.  Their arithmetic is
    # independent over feature dimensions, however.  `value` is a D-vector,
    # and preserving the scalar loop's descending offset order also preserves
    # its rounding behaviour for each feature.
    for i in range(n):
        max_d = min(bandwidth, i)
        for d in range(max_d, 0, -1):
            value = a_band[i, d].copy()
            j = i - d
            for offset in range(max_d, d, -1):
                value -= lower[i, offset] * lower[j, offset - d]
            lower[i, d] = value / lower[j, 0]

        value = a_band[i, 0].copy()
        for offset in range(max_d, 0, -1):
            value -= lower[i, offset] * lower[i, offset]
        # Match the scalar path's finite fallback for a non-positive (or NaN)
        # pivot without branching once per feature.
        lower[i, 0] = np.sqrt(np.where(value > 0, value, 1e-6))
    return lower


def banded_solve(lower: np.ndarray, b: np.ndarray, bandwidth: int) -> np.ndarray:
    """Solve ``A x = b`` given ``A``'s packed-banded Cholesky factor.

    ``lower`` has shape ``(n, bandwidth + 1)`` with ``lower[i, d] =
    L[i - d, i]`` (the layout `banded_cholesky` returns), and ``b`` has shape
    ``(n,)``; the result is the length-``n`` solution ``x``.

    MLPG may also pass independent feature dimensions together as
    ``(n, bandwidth + 1, D)`` with right-hand sides ``(n, D)``.  Each final-axis
    slice is solved independently.  Keeping that axis together lets the short
    band operations run in NumPy across all features instead of repeating the
    Python loops once per feature.
    """
    lower = np.asarray(lower, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if lower.ndim not in (2, 3):
        raise ValueError(f"packed factor must be 2-D or 3-D, "
                         f"got {lower.shape}")
    if lower.shape[1] < bandwidth + 1:
        raise ValueError(f"packed factor must have at least {bandwidth + 1} "
                         f"columns, got {lower.shape}")
    n = lower.shape[0]

    if lower.ndim == 2:
        if b.ndim != 1 or b.shape[0] != n:
            raise ValueError(f"right-hand side must have shape {(n,)}, "
                             f"got {b.shape}")
        # Preserve the single-feature path for callers using the original
        # public representation.  The batched path below is for MLPG's D-axis.
        y = np.zeros(n, dtype=np.float64)
        for i in range(n):                              # L y = b
            acc = 0.0
            for d in range(1, min(bandwidth, i) + 1):
                acc += lower[i, d] * y[i - d]
            y[i] = (b[i] - acc) / lower[i, 0]
        x = np.zeros(n, dtype=np.float64)
        for i in range(n - 1, -1, -1):                  # L^T x = y
            acc = 0.0
            for d in range(1, min(bandwidth, n - 1 - i) + 1):
                acc += lower[i + d, d] * x[i + d]
            x[i] = (y[i] - acc) / lower[i, 0]
        return x

    dim = lower.shape[2]
    if b.ndim != 2 or b.shape != (n, dim):
        raise ValueError(f"right-hand side must have shape {(n, dim)}, "
                         f"got {b.shape}")

    # A triangular substitution is serial along the frames (y[i] needs
    # y[i - 1], ...), so the frame and (tiny) bandwidth loops stay in Python.
    # The arithmetic is independent over feature dimensions, however: `acc` is
    # a D-vector, and accumulating the band products in the same ascending
    # offset order as the single-feature loop above preserves its rounding
    # exactly, feature by feature.
    y = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):                                  # L y = b
        acc = np.zeros(dim, dtype=np.float64)
        for d in range(1, min(bandwidth, i) + 1):
            acc += lower[i, d] * y[i - d]
        y[i] = (b[i] - acc) / lower[i, 0]

    # L^T x = y, in place: each y[i] is read exactly once more (by x[i]) and is
    # dead afterwards, so the solution overwrites the forward result instead
    # of allocating a second (n, D) buffer.
    for i in range(n - 1, -1, -1):
        acc = np.zeros(dim, dtype=np.float64)
        for d in range(1, min(bandwidth, n - 1 - i) + 1):
            acc += lower[i + d, d] * y[i + d]
        y[i] = (y[i] - acc) / lower[i, 0]
    return y


def mlpg(means: np.ndarray, variances: np.ndarray,
         stream_sizes: Sequence[int], window: int = 2,
         variance_scale: float = DEFAULT_VARIANCE_SCALE,
         smooth: bool = True) -> np.ndarray:
    """Maximum-likelihood trajectory through per-frame Gaussians.

    Parameters
    ----------
    means, variances : (T * n_streams, D)
        Stacked static+delta statistics, in the row order produced by
        `hms.core.features.add_dynamic_features`.
    stream_sizes : tuple
        One entry per dynamic stream (all equal to D).
    window : int
        Delta window used during training.
    variance_scale : float
        Multiplies the *dynamic* streams' variances: >1 weakens the delta
        constraints (the trajectory follows the per-frame means more literally,
        i.e. keeps more dynamic detail), <1 strengthens them and flattens the
        trajectory.  1.0 is the unmodified maximum-likelihood solution.
    smooth : bool
        ``False`` returns the static means unchanged (no dynamic features).

    Returns
    -------
    (T, D) static trajectory.
    """
    means = np.atleast_2d(np.asarray(means, dtype=np.float64))
    variances = np.atleast_2d(np.asarray(variances, dtype=np.float64))
    stream_sizes = tuple(int(s) for s in stream_sizes)
    n_streams = len(stream_sizes)
    rows, dim = means.shape
    if rows % n_streams:
        raise ValueError(f"{rows} stacked rows is not a multiple of "
                         f"{n_streams} streams")
    n_frames = rows // n_streams
    if n_frames == 0:
        return np.zeros((0, dim))
    if not smooth or n_streams == 1:
        return means[:n_frames].copy()

    if any(s != dim for s in stream_sizes):
        raise ValueError("stream_sizes must all equal the feature dimension")

    prec = 1.0 / np.maximum(variances, VARIANCE_FLOOR)
    variance_scale = float(variance_scale)
    if variance_scale <= 0:
        raise ValueError("variance_scale must be positive")
    if variance_scale != 1.0:
        # Scaling *all* precisions by the same factor leaves the solution
        # unchanged (the normal equations are homogeneous), so the relaxation
        # knob scales the derivative streams only.
        prec = np.concatenate([prec[:n_frames],
                               prec[n_frames:] / variance_scale])

    coeffs = delta_coeffs(window)
    n_taps = 2 * window + 1
    bandwidth = window_bandwidth(stream_sizes, window)
    frames = np.arange(n_frames)

    # ---- assemble the packed band of M = W^T R^-1 W and the rhs = W^T R^-1 mu
    band = np.zeros((n_frames, bandwidth + 1, dim), dtype=np.float64)
    rhs = np.zeros((n_frames, dim), dtype=np.float64)

    # static stream
    band[:, 0, :] += prec[:n_frames]
    rhs += prec[:n_frames] * means[:n_frames]

    # derivative streams: each stacked row t is a weighted sum over the taps of
    # that derivative (see `stream_taps`), so it contributes to the band at
    # (min(i, j), |i - j|) for every pair of taps, including i == j.
    for derivative in range(1, n_streams):
        sl = slice(derivative * n_frames, (derivative + 1) * n_frames)
        p = prec[sl]                                   # (T, D)
        mu = means[sl]
        taps = stream_taps(n_frames, derivative, window)
        for target, weight in taps:                    # rhs = W^T R^-1 mu
            np.add.at(rhs, target, p * mu * weight)
        for target_i, weight_i in taps:
            for target_j, weight_j in taps:
                product = weight_i * weight_j
                if product == 0.0:
                    continue
                # Only fill the upper triangle: M is symmetric, and iterating
                # every ordered pair would count off-diagonal entries twice
                # (while diagonal entries genuinely need both orders).
                upper = target_i <= target_j
                if not upper.any():
                    continue
                np.add.at(band, (target_i[upper], target_j[upper] - target_i[upper]),
                          (p * product)[upper])

    # ---- solve per feature dimension
    # Pack all independent feature bands together, then factor their shared
    # frame/band structure in one vectorised Cholesky call.  Assembly is done,
    # so reuse its linear-size buffer rather than keep both layouts in memory.
    # Each final-axis slice is exactly the packed representation previously
    # built per feature.
    packed = band
    # short utterances may be narrower than the nominal bandwidth
    for offset in range(1, min(bandwidth, n_frames - 1) + 1):
        # The source/destination overlap while shifting this diagonal.  Copy
        # once per (tiny) band offset so the packed values are not clobbered.
        packed[offset:, offset, :] = packed[:n_frames - offset, offset, :].copy()
        packed[:offset, offset, :] = 0.0
    lower = banded_cholesky(packed, bandwidth)

    # Solve every feature dimension together: the substitutions share their
    # frame/band structure, and `banded_solve` vectorises the band arithmetic
    # over the feature axis (bit-identical to solving each feature on its own).
    return banded_solve(lower, rhs, bandwidth)


def stack_state_statistics(stream_means: Sequence[np.ndarray],
                           stream_variances: Sequence[np.ndarray],
                           state_frames: Sequence[int]
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-state statistics -> per-frame *stacked* statistics for `mlpg`.

    ``stream_means`` holds one (n_states, D) array per dynamic stream (static
    first, then deltas); ``state_frames[i]`` is the number of static frames
    state ``i`` occupies.  Returns stacked (T * n_streams, D) arrays in the
    exact row order `hms.core.features.add_dynamic_features` produces.
    """
    counts = np.asarray(list(state_frames), dtype=np.int64)
    if counts.sum() == 0:
        dim = np.atleast_2d(stream_means[0]).shape[1]
        return np.zeros((0, dim)), np.zeros((0, dim))
    means = [np.repeat(np.atleast_2d(m), counts, axis=0) for m in stream_means]
    variances = [np.repeat(np.atleast_2d(v), counts, axis=0)
                 for v in stream_variances]
    return np.concatenate(means, axis=0), np.concatenate(variances, axis=0)
