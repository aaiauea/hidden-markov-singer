"""Duration model and duration-aware state allocation.

Two jobs:

1. **Predict how long a phoneme should last** when the score does not say
   (`predict`).  Per phoneme, a log-normal model of the total duration, with a
   context term for "is this phoneme in a note with a neighbour of the same
   type" -- deliberately tiny, because for singing the *score* usually carries
   the timing and the model only has to fill in the gaps.

2. **Allocate a known total duration across HMM states** (`allocate`).  This is
   what makes the state sequence duration aware: for a phoneme held for N
   frames, the states are given frames in proportion to their learned mean
   durations, with every state guaranteed at least one frame (and surplus
   frames handed to the longest state when N is tiny).

Both are plain numpy: no duration HMM, no state-level duration distributions
beyond the per-state log-mean/variance captured during HMM training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

#: Frames handed to a state when there is nothing better to go on.
DEFAULT_STATE_FRAMES = 2.0


@dataclass
class DurationStats:
    """Log-duration statistics for one phoneme (in frames)."""

    mean: float = 0.0
    variance: float = 0.5
    count: int = 0
    #: Mean duration in frames, for convenience / inspection.
    frames: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "variance": float(self.variance),
                "count": int(self.count)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "DurationStats":
        stats = cls(float(d.get("mean", 0.0)), float(d.get("variance", 0.5)),
                    int(d.get("count", 0)))
        stats.frames = float(np.exp(stats.mean))
        return stats


class DurationModel:
    """Per-phoneme log-duration model + state allocation."""

    def __init__(self, stats: Optional[Dict[str, DurationStats]] = None,
                 variance_scale: float = 1.0) -> None:
        self.stats: Dict[str, DurationStats] = stats or {}
        #: >1 adds timing variation to predicted durations (0 = deterministic).
        self.variance_scale = float(variance_scale)
        #: Used for phonemes with no fitted statistics: `DEFAULT_STATE_FRAMES`
        #: per state (the duration model itself is context free, so a phone we
        #: have never seen gets a short, neutral duration rather than zero).
        self._fallback = DurationStats(float(np.log(DEFAULT_STATE_FRAMES)),
                                       1.0, 0)
        self._fallback.frames = DEFAULT_STATE_FRAMES

    # -- estimation --------------------------------------------------------

    def fit(self, phone_durations: Dict[str, List[float]]) -> None:
        """Collect per-phoneme durations in frames."""
        for phone, durations in phone_durations.items():
            values = np.asarray([d for d in durations if d > 0], dtype=np.float64)
            if len(values) == 0:
                continue
            logs = np.log(values)
            self.stats[phone] = DurationStats(
                float(logs.mean()), float(max(logs.var(), 1e-4)), len(values))
            self.stats[phone].frames = float(values.mean())

    def has(self, phone: str) -> bool:
        return phone in self.stats

    def mean_frames(self, phone: str) -> float:
        """Context-free mean duration in frames."""
        stats = self.stats.get(phone)
        if stats is None or stats.count == 0:
            return self._fallback.frames
        return float(np.exp(stats.mean))

    # -- prediction --------------------------------------------------------

    def predict(self, phones: Sequence[str], frame_period: float,
                tempo: float = 1.0,
                rng: Optional[np.random.Generator] = None,
                speak: bool = True) -> np.ndarray:
        """Total frames for each phoneme in ``phones``.

        ``speak=False`` returns the deterministic mean (no sampling), which is
        what you want when the score already fixed the timing.
        """
        out = np.zeros(len(phones), dtype=np.float64)
        for i, phone in enumerate(phones):
            stats = self.stats.get(phone)
            if stats is None or stats.count == 0:
                base = self._fallback.frames
                variance = 0.0
            else:
                base = float(np.exp(stats.mean))
                variance = stats.variance if not speak else (
                    stats.variance * self.variance_scale)
            if speak and variance > 0.0 and rng is not None:
                base = float(np.exp(np.log(max(base, 1.0))
                                    + rng.normal(0.0, np.sqrt(variance))))
            out[i] = max(base / max(tempo, 1e-3), 1.0)
        return out

    # -- state allocation --------------------------------------------------

    @staticmethod
    def allocate(total_frames: int, proportions: Sequence[float]) -> np.ndarray:
        """Split ``total_frames`` across states -> per-state frame counts.

        Guarantees: the counts sum exactly to ``total_frames``, each state gets
        at least one frame whenever ``total_frames >= n_states``, and the
        remainder goes to whichever state is furthest below its expected share.
        """
        proportions = np.asarray(list(proportions), dtype=np.float64)
        n = len(proportions)
        total_frames = int(total_frames)
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        if total_frames <= 0:
            return np.zeros(n, dtype=np.int64)
        if total_frames < n:
            # fewer frames than states: give one frame each, drop the rest
            counts = np.zeros(n, dtype=np.int64)
            counts[:total_frames] = 1
            return counts
        if proportions.sum() <= 0:
            proportions = np.full(n, 1.0 / n)
        proportions = proportions / proportions.sum()
        counts = np.maximum(np.floor(proportions * total_frames).astype(np.int64), 1)
        # distribute/remove the remainder by largest deviation from the target
        guard = 0
        while counts.sum() < total_frames and guard < 10 * total_frames + 10:
            deficit = proportions * total_frames - counts
            counts[int(np.argmax(deficit))] += 1
            guard += 1
        while counts.sum() > total_frames and guard < 10 * total_frames + 10:
            surplus = counts - proportions * total_frames
            candidate = int(np.argmax(surplus))
            if counts[candidate] <= 1:
                candidates = np.where(counts > 1)[0]
                if len(candidates) == 0:
                    break
                candidate = int(candidates[np.argmax(counts[candidates])])
            counts[candidate] -= 1
            guard += 1
        return counts

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "variance_scale": self.variance_scale,
            "stats": {phone: s.to_dict() for phone, s in sorted(self.stats.items())},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "DurationModel":
        stats = {phone: DurationStats.from_dict(v)
                 for phone, v in (d.get("stats") or {}).items()}
        return cls(stats, variance_scale=float(d.get("variance_scale", 1.0)))
