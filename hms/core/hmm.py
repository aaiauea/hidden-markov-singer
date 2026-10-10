"""Left-to-right HMMs with GMM emissions, for phoneme units.

Structure
---------
One HMM per phoneme, one state per *segment* of that phoneme::

    enter -> [s0] -> [s1] -> [s2] -> exit
              ^       ^       ^
              |       |       |     (self loops)

Transitions are left-to-right with a self loop; ``allow_skip`` adds s_i -> s_i+2
so the model can sing through a phoneme faster than its shortest state
duration.  Self-loop probabilities are *estimated from the training data*
(``p_self = 1 - 1 / mean_state_duration``) rather than fixed, so the model
knows how long each state usually lasts, and the per-state duration statistics
gathered here are what `hms.core.duration` uses to allocate states to a
requested phoneme duration at synthesis time.

Training
--------
Both estimators are *embedded*: the phone boundaries come from the label files,
the state boundaries are hidden.

``train(..., method="viterbi")``  (default)
    Segmental k-means / embedded Viterbi.  Iteration 0 splits every labelled
    occurrence among the states with a duration-proportional segmentation;
    later iterations segment with the Viterbi path, re-fit each state's GMM on
    the frames it won, and refresh the duration statistics.  This is the
    classic statistical-parametric-synthesis recipe and it is stable on small
    corpora.

``train(..., method="baum_welch")``
    Forward-backward re-estimation, with the state posteriors used as sample
    weights for the GMM update.  The textbook estimator, useful as a refinement
    and as a correctness reference in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms.core.gmm import (COVARIANCE_TYPES, DEFAULT_VAR_FLOOR, EPS, DiagGMM)

#: Self-loop probability before any data has been seen.
DEFAULT_SELF_LOOP = 0.5

#: Bounds on the self-loop probability (keeps every state reachable).
MIN_SELF_LOOP = 0.05
MAX_SELF_LOOP = 0.999


@dataclass
class StateDurationStats:
    """Log-frame duration statistics for one HMM state."""

    mean: float = 0.0
    variance: float = 1.0
    count: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "variance": float(self.variance),
                "count": int(self.count)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "StateDurationStats":
        return cls(float(d.get("mean", 0.0)), float(d.get("variance", 1.0)),
                   int(d.get("count", 0)))


@dataclass
class HMMState:
    """One state: a GMM emission plus duration and voicing statistics."""

    gmm: DiagGMM
    duration: StateDurationStats = field(default_factory=StateDurationStats)
    #: Probability that a frame in this state is voiced (used by `pitch`).
    voiced_prob: float = 1.0

    @property
    def n_free_params(self) -> int:
        return self.gmm.n_free_params + 2


class LeftToRightHMM:
    """A phoneme HMM: ``n_states`` left-to-right states with GMM emissions."""

    def __init__(self, n_states: int = 3, allow_skip: bool = False,
                 covariance_type: str = "diag") -> None:
        if n_states < 1:
            raise ValueError("an HMM needs at least one state")
        if covariance_type not in COVARIANCE_TYPES:
            raise ValueError(f"covariance_type must be one of {COVARIANCE_TYPES}, "
                             f"got {covariance_type!r}")
        self.n_states = int(n_states)
        self.allow_skip = bool(allow_skip)
        self.covariance_type = covariance_type
        self.states: List[Optional[HMMState]] = [None] * self.n_states
        self.self_loops: np.ndarray = np.full(self.n_states, DEFAULT_SELF_LOOP)
        self.dim: Optional[int] = None

    # -- structure ---------------------------------------------------------

    def transition_matrix(self) -> np.ndarray:
        """(n+1) x (n+1) matrix over states plus a non-emitting exit state."""
        n = self.n_states
        a = np.zeros((n + 1, n + 1), dtype=np.float64)
        for i in range(n):
            loop = float(np.clip(self.self_loops[i], MIN_SELF_LOOP, MAX_SELF_LOOP))
            if i < n - 1:
                a[i, i] = loop
                if self.allow_skip and i < n - 2:
                    a[i, i + 1] = (1.0 - loop) * 0.8
                    a[i, i + 2] = (1.0 - loop) * 0.2
                else:
                    a[i, i + 1] = 1.0 - loop
            else:
                a[i, i] = loop
                a[i, n] = 1.0 - loop
        return a

    @property
    def n_free_params(self) -> int:
        return sum(s.n_free_params for s in self.states if s is not None)

    def is_trained(self) -> bool:
        return all(s is not None for s in self.states)

    # -- durations ---------------------------------------------------------

    def duration_proportions(self) -> np.ndarray:
        """Expected share of a phoneme's duration taken by each state."""
        means = np.array([max(s.duration.mean, 0.0) if s else 0.0
                          for s in self.states])
        if means.sum() <= 0:
            return np.full(self.n_states, 1.0 / self.n_states)
        weights = np.exp(means - means.max())
        return weights / weights.sum()

    def duration_mean_frames(self) -> np.ndarray:
        """Per-state mean duration in frames."""
        return np.array([np.exp(s.duration.mean) if s else 1.0
                         for s in self.states])

    # -- likelihoods -------------------------------------------------------

    def emission_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """(T, n_states) matrix of per-frame emission log likelihoods."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = np.full((len(X), self.n_states), -np.inf)
        for i, state in enumerate(self.states):
            if state is not None:
                out[:, i] = state.gmm.log_likelihood(X)
        return out

    def viterbi(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Best left-to-right state sequence for the frames in ``X``."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        t_frames = len(X)
        if t_frames == 0:
            return np.zeros(0, dtype=np.int64), 0.0
        ll = self.emission_log_likelihood(X)
        n = self.n_states
        a = self.transition_matrix()
        # Log-transition table, computed once per call instead of per (t, j).
        log_a = np.log(np.maximum(a, EPS))
        log_trans = log_a[:n, :n]
        all_states = np.arange(n)

        score = np.full((t_frames, n), -np.inf)
        back = np.zeros((t_frames, n), dtype=np.int64)
        score[0] = ll[0] + log_a[0, :n]
        for t in range(1, t_frames):
            previous = score[t - 1]
            if not np.isfinite(previous).any():
                # Every path has died: nothing can be recovered from here on.
                back[t] = all_states
                score[t] = -np.inf
                continue
            # candidates[i, j] = score(t-1, i) + log a(i -> j); the column-wise
            # argmax picks the same predecessor as a per-state argmax would
            # (first maximum on ties), so the recursion is bit-for-bit equal.
            candidates = previous[:, None] + log_trans
            best_i = candidates.argmax(axis=0)
            back[t] = best_i
            score[t] = candidates[best_i, all_states] + ll[t]

        exit_scores = score[-1] + log_a[:n, n]
        last = int(np.argmax(exit_scores))
        path = np.zeros(t_frames, dtype=np.int64)
        path[-1] = last
        for t in range(t_frames - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(exit_scores[last])

    def forward_log(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Log-space forward algorithm; returns (alpha (T, n), log p(X))."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        ll = self.emission_log_likelihood(X)
        a = self.transition_matrix()
        n = self.n_states
        t_frames = len(X)
        alpha = np.full((t_frames, n), -np.inf)
        alpha[0] = np.log(np.maximum(a[0, :n], EPS)) + ll[0]
        for t in range(1, t_frames):
            for j in range(n):
                incoming = alpha[t - 1] + np.log(np.maximum(a[:n, j], EPS))
                m = incoming.max()
                alpha[t, j] = ((m + np.log(np.exp(incoming - m).sum()))
                               if np.isfinite(m) else -np.inf) + ll[t, j]
        outgoing = alpha[-1] + np.log(np.maximum(a[:n, n], EPS))
        m = outgoing.max()
        log_prob = (float(m + np.log(np.exp(outgoing - m).sum()))
                    if np.isfinite(m) else float(-np.inf))
        return alpha, log_prob

    def backward_log(self, X: np.ndarray) -> np.ndarray:
        """Log-space backward algorithm; returns beta (T, n)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        ll = self.emission_log_likelihood(X)
        a = self.transition_matrix()
        n = self.n_states
        t_frames = len(X)
        beta = np.full((t_frames, n), -np.inf)
        beta[-1] = np.log(np.maximum(a[:n, n], EPS))
        for t in range(t_frames - 2, -1, -1):
            for i in range(n):
                outgoing = (np.log(np.maximum(a[i, :n], EPS)) + ll[t + 1]
                            + beta[t + 1])
                m = outgoing.max()
                beta[t, i] = (m + np.log(np.exp(outgoing - m).sum())
                              if np.isfinite(m) else -np.inf)
        return beta

    def log_likelihood(self, X: np.ndarray) -> float:
        """Exact log p(X | model) via the forward algorithm."""
        return self.forward_log(np.atleast_2d(X))[1]

    def state_posteriors(self, X: np.ndarray) -> np.ndarray:
        """gamma[t, i] = P(state i at time t | X), shape (T, n_states)."""
        alpha, log_prob = self.forward_log(X)
        beta = self.backward_log(X)
        log_gamma = alpha + beta - log_prob
        m = log_gamma.max(axis=1, keepdims=True)
        gamma = np.exp(log_gamma - m)
        return gamma / np.maximum(gamma.sum(axis=1, keepdims=True), EPS)

    # -- segmentation helpers ---------------------------------------------

    @staticmethod
    def proportional_segmentation(n_frames: int, n_states: int,
                                  proportions: Optional[np.ndarray] = None
                                  ) -> np.ndarray:
        """Split ``n_frames`` among states by expected state durations.

        Returns the n_states + 1 boundary array.  Every state gets at least one
        frame when there are enough frames to go round; if there are fewer
        frames than states the surplus states receive an empty span, which the
        synthesis path resolves by merging (see `hms.core.duration`).
        """
        boundaries = np.zeros(n_states + 1, dtype=np.int64)
        if n_frames <= 0:
            return boundaries
        if proportions is None or len(proportions) != n_states \
                or proportions.sum() <= 0:
            proportions = np.full(n_states, 1.0 / n_states)
        if n_frames < n_states:
            boundaries[1:] = np.minimum(np.arange(1, n_states + 1), n_frames)
            return boundaries
        counts = np.maximum(
            np.floor(proportions / proportions.sum() * n_frames).astype(np.int64), 1)
        while counts.sum() > n_frames:
            idx = int(np.argmax(counts))
            if counts[idx] <= 1:
                break
            counts[idx] -= 1
        while counts.sum() < n_frames:
            counts[int(np.argmax(proportions))] += 1
        boundaries[1:] = np.cumsum(counts)
        return boundaries

    def min_state_frames(self, n_frames: int) -> int:
        """Occupancy floor for one state, as a quarter of its equal share.

        Without a floor, embedded Viterbi happily gives a state 1-2 frames of a
        100-frame vowel, which then (a) trains a near-useless emission for it
        and (b) makes the duration model allocate it almost no frames at
        synthesis, effectively deleting the state.
        """
        if n_frames < 2 * self.n_states:
            return 1
        return max(1, int(round(0.25 * n_frames / self.n_states)))

    @staticmethod
    def _enforce_min_occupancy(bounds: np.ndarray, min_frames: int
                               ) -> np.ndarray:
        """Nudge interior boundaries until every state has >= min_frames.

        Frames are taken from the nearest state that is above the floor, so the
        acoustically-driven boundaries move as little as possible.
        """
        bounds = bounds.copy()
        n_states = len(bounds) - 1
        if min_frames <= 1:
            return bounds
        for _ in range(16 * n_states):
            counts = np.diff(bounds)
            starving = np.where(counts < min_frames)[0]
            if len(starving) == 0:
                break
            target = int(starving[0])
            surplus = counts - min_frames
            order = [target - 1, target + 1] + [
                j for j in range(n_states) if j not in (target - 1, target + 1)]
            donor = next((j for j in order
                          if 0 <= j < n_states and surplus[j] > 0), None)
            if donor is None:
                break                          # nothing left to give: accept it
            if donor < target:
                bounds[donor + 1: target + 1] -= 1
            else:
                bounds[target + 1: donor + 1] += 1
        return bounds

    def segment(self, X: np.ndarray, fallback: bool = True,
                min_frames: Optional[int] = None) -> np.ndarray:
        """Frame index -> state index: monotone, and non-degenerate.

        Uses the Viterbi path, repairs it so no state falls below the occupancy
        floor, and falls back to a duration-proportional split when the path
        collapses (which happens while the model is still flat).
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n_frames = len(X)
        if n_frames == 0:
            return np.zeros(0, dtype=np.int64)
        floor = self.min_state_frames(n_frames) if min_frames is None \
            else max(1, int(min_frames))
        path, _ = self.viterbi(X)
        path = np.maximum.accumulate(np.clip(path, 0, self.n_states - 1))
        if n_frames < self.n_states:
            return path

        counts = np.bincount(path, minlength=self.n_states)
        if fallback and (counts == 0).any():
            # the acoustic path collapsed (flat model, or a very short
            # segment): start from the duration-proportional split instead
            bounds = self._enforce_min_occupancy(
                self.proportional_segmentation(n_frames, self.n_states,
                                               self.duration_proportions()),
                floor)
        else:
            bounds = np.zeros(self.n_states + 1, dtype=np.int64)
            for state in range(self.n_states):
                where = np.where(path == state)[0]
                bounds[state + 1] = (where.max() + 1) if len(where) \
                    else bounds[state]
            bounds[-1] = n_frames
            bounds = self._enforce_min_occupancy(bounds, floor)
        repaired = np.repeat(np.arange(self.n_states), np.diff(bounds))
        if len(repaired) != n_frames or (np.diff(bounds) <= 0).any():
            return path                                # repair failed: keep raw
        return repaired

    # -- training ----------------------------------------------------------

    def train(self, sequences: Sequence[np.ndarray], n_components: int = 1,
              covariance_type: str = "diag", n_iterations: int = 5,
              var_floor_ratio: float = DEFAULT_VAR_FLOOR, seed: int = 0,
              method: str = "viterbi",
              voiced: Optional[Sequence[np.ndarray]] = None) -> None:
        """Train on every labelled occurrence of this phoneme.

        ``sequences`` holds one (T_i, D) frame matrix per occurrence, already
        cut to the phoneme's labelled boundaries.  ``voiced`` optionally holds
        one boolean array per occurrence.
        """
        if not sequences:
            raise ValueError("no training sequences given")
        self.covariance_type = covariance_type
        self.dim = sequences[0].shape[1]
        all_frames = np.concatenate(sequences, axis=0)
        if not np.isfinite(all_frames).all():
            raise ValueError("training features contain non-finite values")

        # Bootstrap every state with the same global Gaussian; the first
        # iteration's duration-proportional split breaks the symmetry.
        for i in range(self.n_states):
            self.states[i] = HMMState(
                gmm=DiagGMM.fit(all_frames, n_components=n_components,
                                covariance_type=covariance_type,
                                var_floor_ratio=var_floor_ratio, seed=seed + i),
                duration=StateDurationStats(0.0, 1.0, len(sequences)))

        if method == "baum_welch":
            self._train_baum_welch(sequences, voiced, n_components,
                                   covariance_type, n_iterations,
                                   var_floor_ratio, seed)
            return
        if method != "viterbi":
            raise ValueError(f"unknown training method {method!r}")

        for iteration in range(max(1, n_iterations)):
            frames_by_state: List[List[np.ndarray]] = [[] for _ in range(self.n_states)]
            voiced_by_state: List[List[np.ndarray]] = [[] for _ in range(self.n_states)]
            runs_by_state: List[List[int]] = [[] for _ in range(self.n_states)]

            for index, sequence in enumerate(sequences):
                if iteration == 0:
                    boundaries = self.proportional_segmentation(
                        len(sequence), self.n_states, self.duration_proportions())
                    path = np.repeat(np.arange(self.n_states),
                                     np.diff(boundaries))[:len(sequence)]
                else:
                    path = self.segment(sequence)
                for i in range(self.n_states):
                    where = np.where(path == i)[0]
                    if len(where) == 0:
                        continue
                    frames_by_state[i].append(sequence[where])
                    runs_by_state[i].append(len(where))
                    if voiced is not None:
                        voiced_by_state[i].append(
                            np.asarray(voiced[index], dtype=bool)[where])

            for i in range(self.n_states):
                frames = self._stack(frames_by_state[i])
                if len(frames) == 0:
                    continue
                self.states[i].gmm = DiagGMM.fit(
                    frames, n_components=n_components,
                    covariance_type=covariance_type,
                    var_floor_ratio=var_floor_ratio,
                    seed=seed + 100 * (iteration + 1) + i)
                if runs_by_state[i]:
                    logs = np.log(np.maximum(np.array(runs_by_state[i] or [1],
                                                      dtype=np.float64), 1.0))
                    self.states[i].duration = StateDurationStats(
                        float(logs.mean()), float(max(logs.var(), 1e-4)),
                        int(len(logs)))
                if voiced is not None and voiced_by_state[i]:
                    votes = np.concatenate(voiced_by_state[i], axis=0)
                    if len(votes):
                        self.states[i].voiced_prob = float(np.mean(votes))

            means = np.array([np.exp(s.duration.mean) for s in self.states])
            for i in range(self.n_states):
                self.self_loops[i] = float(np.clip(1.0 - 1.0 / max(means[i], 1.0),
                                                   MIN_SELF_LOOP, MAX_SELF_LOOP))

    def _train_baum_welch(self, sequences: Sequence[np.ndarray],
                          voiced: Optional[Sequence[np.ndarray]],
                          n_components: int, covariance_type: str,
                          n_iterations: int, var_floor_ratio: float,
                          seed: int) -> None:
        for iteration in range(max(1, n_iterations)):
            frames_by_state: List[List[np.ndarray]] = [[] for _ in range(self.n_states)]
            weights_by_state: List[List[np.ndarray]] = [[] for _ in range(self.n_states)]
            voiced_by_state: List[List[np.ndarray]] = [[] for _ in range(self.n_states)]

            for index, sequence in enumerate(sequences):
                gamma = self.state_posteriors(sequence)
                for i in range(self.n_states):
                    w = gamma[:, i]
                    active = w > 1e-6
                    if active.any():
                        frames_by_state[i].append(sequence[active])
                        weights_by_state[i].append(w[active])
                        if voiced is not None:
                            voiced_by_state[i].append(
                                np.asarray(voiced[index], dtype=bool)[active])

            for i in range(self.n_states):
                if not frames_by_state[i]:
                    continue
                frames = np.concatenate(frames_by_state[i], axis=0)
                weights = np.concatenate(weights_by_state[i], axis=0)
                self.states[i].gmm = DiagGMM.fit(
                    frames, n_components=n_components,
                    covariance_type=covariance_type, weights=weights,
                    var_floor_ratio=var_floor_ratio,
                    seed=seed + 100 * (iteration + 1) + i)
                # effective per-state duration = weight mass / number of visits
                total = float(weights.sum())
                visits = max(int((gamma[:, i] > 0.5).sum()), 1)
                mean = max(total / visits, 1.0)
                self.states[i].duration = StateDurationStats(
                    float(np.log(mean)), 1e-3, visits)
                self.self_loops[i] = float(np.clip(1.0 - 1.0 / mean,
                                                   MIN_SELF_LOOP, MAX_SELF_LOOP))
                if voiced is not None and voiced_by_state[i]:
                    votes = np.concatenate(voiced_by_state[i], axis=0)
                    if len(votes):
                        self.states[i].voiced_prob = float(np.mean(votes))

    @staticmethod
    def _stack(chunks: Sequence[np.ndarray]) -> np.ndarray:
        chunks = [c for c in chunks if len(c)]
        if not chunks:
            return np.zeros((0, 0))
        return np.concatenate(chunks, axis=0)

    # -- serialisation -----------------------------------------------------

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "self_loops": np.asarray(self.self_loops, dtype=np.float64),
            "duration_mean": np.array([s.duration.mean for s in self.states]),
            "duration_variance": np.array([s.duration.variance
                                           for s in self.states]),
            "duration_count": np.array([s.duration.count for s in self.states]),
            "voiced_prob": np.array([s.voiced_prob for s in self.states]),
            "weights": np.stack([s.gmm.weights for s in self.states]),
            "means": np.stack([s.gmm.means for s in self.states]),
            "variances": np.stack([s.gmm.variances for s in self.states]),
        }

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray], allow_skip: bool = False,
                    covariance_type: str = "diag") -> "LeftToRightHMM":
        weights = np.atleast_2d(arrays["weights"])
        n_states = weights.shape[0]
        hmm = cls(n_states=n_states, allow_skip=allow_skip,
                  covariance_type=covariance_type)
        hmm.self_loops = np.asarray(arrays["self_loops"], dtype=np.float64)
        for i in range(n_states):
            gmm = DiagGMM(weights[i], arrays["means"][i], arrays["variances"][i],
                          covariance_type)
            hmm.states[i] = HMMState(
                gmm=gmm,
                duration=StateDurationStats(
                    float(arrays["duration_mean"][i]),
                    float(arrays["duration_variance"][i]),
                    int(arrays["duration_count"][i])),
                voiced_prob=float(arrays["voiced_prob"][i]))
        hmm.dim = hmm.states[0].gmm.dim
        return hmm

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        k = self.states[0].gmm.n_components if self.states[0] else 0
        return f"<LeftToRightHMM states={self.n_states} K={k}>"
