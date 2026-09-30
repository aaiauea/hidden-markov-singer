"""Experimental, optional global-variance (GV) generation on static features.

Training records the *mean of utterance-level* variances in normalized static
feature space. At synthesis, MLPG supplies an initial (T, D) static trajectory
``c0``. For each feature independently, we minimize

    E(c) = (1 / 2T) ||W(c - c0)||^2_P
         + (weight / 2) ((GV(c) - target) / q)^2,
    GV(c) = mean_t((c_t - mean(c))^2),
    q = max(target, GV(c0), GV_SCALE_FLOOR).

``W`` and ``P`` are the same dynamic windows and floored/scaled precisions as
MLPG. When ``c0`` is the MLPG optimum, the first term is exactly the *increase*
in the MLPG negative log-likelihood (up to solve roundoff). It anchors GV to the
acoustic model, including delta constraints; unlike multiplying the trajectory
by a variance ratio, the updates need not be uniform in time. With
``smooth=False`` it is instead a local quadratic anchor around the returned
static means. The second term is an experimental variance penalty, not a
learned GV likelihood: only per-feature target variances are estimated, not the
uncertainty of those estimates. ``q`` makes the weight dimensionless and avoids
dividing by zero for flat features.

This module does not change MLPG's optimized assembly, factorization or solve.
The optional optimizer uses W and W^T for a diagonally preconditioned gradient
step with a per-feature backtracking line search. It never assembles a dense
matrix or factors the band again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from hms.core.generation import VARIANCE_FLOOR, _coalesced_stream_taps

# These floors affect GV only, not MLPG or the stored training statistics.
GV_SCALE_FLOOR = 1e-6              # scale for a nearly flat feature's penalty
GV_ACTIVE_VARIANCE = 1e-12         # a constant trajectory has zero GV gradient
_MAX_LINE_SEARCH = 12


@dataclass
class GlobalVarianceStats:
    """Per-static-feature GV targets in normalized feature space.

    ``utterances`` counts usable training utterances (at least two frames).
    Deltas and delta-deltas are *not* entries in ``target_variance``.
    """

    target_variance: np.ndarray
    utterances: int

    def __post_init__(self) -> None:
        target = np.asarray(self.target_variance, dtype=np.float64)
        if target.ndim != 1 or not target.size or not np.isfinite(target).all() \
                or (target < 0).any():
            raise ValueError("GV targets must be a nonempty 1-D array of finite, "
                             "non-negative variances")
        if not isinstance(self.utterances, (int, np.integer)) \
                or isinstance(self.utterances, (bool, np.bool_)) \
                or self.utterances < 1:
            raise ValueError("GV utterances must be a positive integer")
        self.target_variance = target.copy()
        self.utterances = int(self.utterances)

    def to_dict(self) -> dict:
        return {"space": "normalized_static",
                "utterances": self.utterances,
                "target_variance": self.target_variance.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalVarianceStats":
        if not isinstance(data, dict) or data.get("space") != "normalized_static":
            raise ValueError("GV section must describe normalized_static features")
        return cls(target_variance=data.get("target_variance", []),
                   utterances=data.get("utterances", 0))


def trajectory_variance(trajectory: np.ndarray) -> np.ndarray:
    """Population variance of each static trajectory around its *own* mean.

    Accepts (T, D) or (T,), and returns (D,) or a scalar respectively. Empty
    trajectories have zero variance; non-finite inputs/results are rejected.
    """
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim not in (1, 2) or not np.isfinite(values).all():
        raise ValueError("GV trajectory must be a finite 1-D or 2-D array")
    if not len(values):
        return np.zeros(values.shape[1:])
    with np.errstate(over="ignore", invalid="ignore"):
        centered = values - values.mean(axis=0)
        variance = np.mean(centered * centered, axis=0)
    if not np.isfinite(variance).all():
        raise ValueError("GV trajectory variance is not finite")
    return variance


def variance_gradient(trajectory: np.ndarray) -> np.ndarray:
    """d GV(c) / d c[t, d] = 2 * (c[t, d] - mean(c[:, d])) / T."""
    values = np.asarray(trajectory, dtype=np.float64)
    if values.ndim not in (1, 2) or not np.isfinite(values).all():
        raise ValueError("GV trajectory must be a finite 1-D or 2-D array")
    if not len(values):
        return np.zeros_like(values)
    gradient = 2.0 / len(values) * (values - values.mean(axis=0))
    if not np.isfinite(gradient).all():
        raise ValueError("GV variance gradient is not finite")
    return gradient


def estimate_global_variance(utterances: Iterable[np.ndarray],
                             static_dim: int) -> Optional[GlobalVarianceStats]:
    """Mean of per-utterance static GV vectors, streaming one utterance at a time.

    The caller supplies normalized feature matrices in FeatureSpec's layout:
    columns [:static_dim] are static, followed by delta/delta-delta blocks.
    Utterances with fewer than two frames carry no usable temporal variance.
    Each usable utterance has equal weight (rather than pooling corpus frames
    around one corpus-wide mean, which would include between-utterance shifts).
    """
    total = np.zeros(static_dim, dtype=np.float64)
    count = 0
    for features in utterances:
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] < static_dim:
            raise ValueError("GV training features must include the static block")
        if len(features) < 2:
            continue
        total += trajectory_variance(features[:, :static_dim])
        count += 1
    return GlobalVarianceStats(total / count, count) if count else None


class _StaticWindows:
    """W, W^T and diag(W^T P W) for independent static feature trajectories.

    Reuse MLPG's coalesced taps: its double-clipped delta-delta and replicated
    utterance boundaries are important, especially for T shorter than a window.
    Coalescing before squaring is essential for the diagonal at those edges.
    """

    def __init__(self, n_frames: int, n_streams: int, window: int):
        self.n_frames = n_frames
        self.n_streams = n_streams
        self.taps = [_coalesced_stream_taps(n_frames, derivative, window)[1:]
                     for derivative in range(1, n_streams)]

    def apply(self, static: np.ndarray) -> np.ndarray:
        n = self.n_frames
        out = np.zeros((n * self.n_streams, static.shape[1]), dtype=np.float64)
        out[:n] = static
        for derivative, (targets, weights) in enumerate(self.taps, start=1):
            block = out[derivative * n:(derivative + 1) * n]
            for k in range(weights.shape[1]):
                if np.any(weights[:, k]):
                    block += static[targets[:, k]] * weights[:, k, None]
        return out

    def transpose(self, stacked: np.ndarray) -> np.ndarray:
        n = self.n_frames
        out = stacked[:n].copy()
        for derivative, (targets, weights) in enumerate(self.taps, start=1):
            block = stacked[derivative * n:(derivative + 1) * n]
            for k in range(weights.shape[1]):
                if np.any(weights[:, k]):
                    np.add.at(out, targets[:, k], block * weights[:, k, None])
        return out

    def diagonal(self, precision: np.ndarray) -> np.ndarray:
        n = self.n_frames
        out = precision[:n].copy()
        for derivative, (targets, weights) in enumerate(self.taps, start=1):
            block = precision[derivative * n:(derivative + 1) * n]
            for k in range(weights.shape[1]):
                if np.any(weights[:, k]):
                    np.add.at(out, targets[:, k], block * weights[:, k, None] ** 2)
        return out


def optimize_global_variance(
        trajectory: np.ndarray, variances: np.ndarray,
        stream_sizes: Sequence[int], target_variance: np.ndarray, *,
        window: int = 2, variance_scale: float = 1.0,
        weight: float = 1.0, iterations: int = 20,
        return_steps: bool = False) -> np.ndarray | tuple[np.ndarray, int]:
    """Iteratively encourage static GV without losing the MLPG likelihood.

    Inputs are in *normalized* feature space. ``trajectory`` is MLPG's (T,D)
    static output, ``variances`` its stream-major (T*S,D) model variances, and
    ``target_variance`` holds precisely D static targets. No training deltas are
    treated as separate features. Does not mutate any inputs.

    A zero weight/iteration count or T < 2 returns a copy. An exactly constant
    feature with a positive target also stays constant: its variance gradient
    is zero and inventing structure/noise would be arbitrary. Every accepted
    update decreases the objective for that feature; a bounded step and finite
    checks protect against degenerate statistics or runaway trajectories.
    With ``return_steps=True``, return ``(trajectory, successful_step_count)``
    for benchmarking; the normal return value is just the trajectory.
    """
    def finish(values: np.ndarray, steps: int):
        return (values, steps) if return_steps else values

    c0 = np.asarray(trajectory, dtype=np.float64)
    if c0.ndim != 2 or not np.isfinite(c0).all():
        raise ValueError("GV trajectory must have shape (T, D) and be finite")
    n_frames, dim = c0.shape
    sizes = tuple(int(size) for size in stream_sizes)
    if not sizes or any(size != dim for size in sizes):
        raise ValueError("GV stream sizes must all equal the static dimension")
    model_variances = np.asarray(variances, dtype=np.float64)
    if model_variances.shape != (n_frames * len(sizes), dim) \
            or not np.isfinite(model_variances).all():
        raise ValueError("GV requires finite (T * streams, static_dim) variances")
    target = np.asarray(target_variance, dtype=np.float64)
    if target.shape != (dim,) or not np.isfinite(target).all() \
            or (target < 0).any():
        raise ValueError("GV target variance must have one finite, non-negative "
                         "value per static feature")
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool) \
            or iterations < 0:
        raise ValueError("GV iterations must be a non-negative integer")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("GV weight must be finite and non-negative")
    if not np.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError("GV variance_scale must be finite and positive")
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("GV window must be a positive integer")
    if n_frames < 2 or iterations == 0 or weight == 0:
        return finish(c0.copy(), 0)

    start_var = trajectory_variance(c0)
    active = (start_var > GV_ACTIVE_VARIANCE) & (start_var != target)
    if not active.any():
        return finish(c0.copy(), 0)
    scale = np.maximum(np.maximum(start_var, target), GV_SCALE_FLOOR)
    precision = 1.0 / np.maximum(model_variances, VARIANCE_FLOOR)
    if variance_scale != 1.0:
        precision[n_frames:] /= variance_scale  # dynamic streams only, as in MLPG
    if not np.isfinite(precision).all():
        raise ValueError("GV precision is not finite (check variance_scale)")
    windows = _StaticWindows(n_frames, len(sizes), window)
    diagonal = np.maximum(windows.diagonal(precision), VARIANCE_FLOOR)
    if not np.isfinite(diagonal).all():
        raise ValueError("GV preconditioner is not finite")

    c = c0.copy()
    displacement = np.zeros_like(c)
    w_displacement = np.zeros_like(precision)
    steps_done = 0
    for _ in range(iterations):
        variance = trajectory_variance(c)
        centered = c - c.mean(axis=0)
        # Multiply the objective gradient by T: its MLPG part is A(c-c0),
        # and dGV/dc = 2(c-mean(c))/T. The common factor cancels in the
        # preconditioned direction / line search.
        with np.errstate(over="ignore", invalid="ignore"):
            relative_error = (variance - target) / scale
            gradient = windows.transpose(precision * w_displacement)
            gradient += 2.0 * weight * centered * (relative_error / scale)
            direction = -gradient / diagonal
        direction[:, ~active] = 0.0
        if not np.isfinite(direction).all():
            break
        # A bound on each iteration's displacement in normalized feature
        # units, not a cap on the target variance. Line search enforces the
        # stronger condition: the exact objective must improve.
        limit = 0.5 * np.sqrt(scale)
        max_step = np.max(np.abs(direction), axis=0)
        direction *= np.minimum(1.0, limit / np.maximum(max_step, 1e-30))
        if not np.any(direction):
            break
        w_direction = windows.apply(direction)
        centered_direction = direction - direction.mean(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            # Along c + a*d, both terms can be evaluated from short quadratic
            # polynomials: ||W(delta+a*d)||_P^2 and GV(c+a*d). No extra W
            # evaluation or full trajectory copy is needed for backtracking.
            likelihood = 0.5 / n_frames * np.sum(
                precision * w_displacement ** 2, axis=0)
            cross_likelihood = np.sum(
                precision * w_displacement * w_direction, axis=0) / n_frames
            direction_likelihood = 0.5 / n_frames * np.sum(
                precision * w_direction ** 2, axis=0)
            cross_variance = 2.0 / n_frames * np.sum(
                centered * centered_direction, axis=0)
            direction_variance = np.mean(centered_direction ** 2, axis=0)
            before = likelihood + 0.5 * weight * relative_error ** 2

        step = np.ones(dim, dtype=np.float64)
        accepted = np.zeros(dim, dtype=bool)
        taken = np.zeros(dim, dtype=np.float64)
        for _ in range(_MAX_LINE_SEARCH):
            with np.errstate(over="ignore", invalid="ignore"):
                next_variance = np.maximum(
                    0.0, variance + step * cross_variance
                    + step ** 2 * direction_variance)
                after = (likelihood + step * cross_likelihood
                         + step ** 2 * direction_likelihood
                         + 0.5 * weight * ((next_variance - target) / scale) ** 2)
            better = active & ~accepted & np.isfinite(after) & (after < before)
            taken[better] = step[better]
            accepted |= better
            if np.all(accepted | ~active):
                break
            step[~accepted] *= 0.5
        if not accepted.any():
            break
        displacement += direction * taken[None, :]
        w_displacement += w_direction * taken[None, :]
        c = c0 + displacement
        if not np.isfinite(c).all():  # last resort for extreme but finite inputs
            return finish(c0.copy(), 0)
        steps_done += 1
        # Features are independent; don't keep trying a converged feature while
        # other features continue to improve.
        active &= accepted
    return finish(c, steps_done)
