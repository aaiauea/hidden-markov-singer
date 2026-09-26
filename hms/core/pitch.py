"""Pitch model: note-conditioned F0 generation.

Design
------
The requested musical note **drives** F0; the statistical model only supplies
the *variation around* it.  Concretely, for every training frame the trainer
subtracts the note that is being sung and stores the remainder in feature
dimension 0 -- in semitones:

    relative[t] = 1200 * log2(f0[t] / note_hz[t]) / 100  (i.e. semitones)

so the acoustic HMM learns *this singer's* habits -- onset scoops, vibrato,
sharp/flat drift, the way pitch falls into a phrase -- independently of which
note is being sung.  At synthesis time the score's note is added back and the
model's learned deviation is added on top.  That is the whole trick, and it is
why HMS does not simply copy the training speaker's F0 contour.

This module implements four things:

1. `note_relative_pitch` / `absolute_pitch` -- the training/synthesis transform,
   including interpolation of unvoiced gaps (an unvoiced frame carries no pitch
   information; leaving the raw value in the feature vector would teach the
   model nonsense).
2. Per-(phoneme, state) relative-pitch statistics, used for inspection, for the
   deterministic ``state_means`` pitch source, and as a prior when a phoneme has
   too little data to trust the acoustic model's version of dimension 0.
3. `Vibrato` -- a separate, explicit component, as it should be: a delay, a
   rate, a depth, optional randomisation.  Enabled per voice in
   `parameters.yaml`, and its defaults can be *measured* from training data
   (`estimate_vibrato`).  Kept out of the HMM because MLPG smooths away exactly
   the fast oscillation that makes vibrato sound alive.
4. `voiced_mask` -- frames that should carry F0, from the HMM states' learned
   voicing probabilities plus the phoneme class (unvoiced consonants are never
   voiced, vowels nearly always are).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Default assumptions for a phoneme we have no statistics for.
DEFAULT_RELATIVE_MEAN = 0.0      # semitones away from the note
DEFAULT_RELATIVE_STD = 0.35


@dataclass
class PitchStats:
    """Relative-pitch statistics for one HMM state (semitones re. the note)."""

    mean: float = DEFAULT_RELATIVE_MEAN
    variance: float = DEFAULT_RELATIVE_STD ** 2
    count: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "variance": float(self.variance),
                "count": int(self.count)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PitchStats":
        return cls(float(d.get("mean", DEFAULT_RELATIVE_MEAN)),
                   float(d.get("variance", DEFAULT_RELATIVE_STD ** 2)),
                   int(d.get("count", 0)))


@dataclass
class Vibrato:
    """Explicit vibrato component (deliberately outside the HMM)."""

    enabled: bool = False
    rate_hz: float = 5.5
    depth_semitones: float = 0.6
    delay_ms: float = 120.0
    #: 0 = perfectly periodic, 1 = fully random rate/depth per note
    randomness: float = 0.15
    #: Depth fade-in time constant (ms) after the delay.
    attack_ms: float = 80.0
    waveform: str = "sine"        # "sine" | "triangle"

    def __post_init__(self) -> None:
        values = (self.rate_hz, self.depth_semitones, self.delay_ms,
                  self.randomness, self.attack_ms)
        try:
            if not all(np.isfinite(value) for value in values):
                raise ValueError("vibrato settings must be finite")
        except TypeError as exc:
            raise ValueError("vibrato settings must be numeric") from exc
        if self.rate_hz <= 0:
            raise ValueError("vibrato rate_hz must be positive")
        if self.depth_semitones < 0 or self.delay_ms < 0 or self.attack_ms < 0:
            raise ValueError("vibrato depth, delay and attack must be non-negative")
        if not 0 <= self.randomness <= 1:
            raise ValueError("vibrato randomness must be in [0, 1]")
        if self.waveform not in ("sine", "triangle"):
            raise ValueError("vibrato waveform must be 'sine' or 'triangle'")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Vibrato":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def contour(self, n_frames: int, frame_period: float, note_index: int,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Vibrato offset (semitones) for one note of ``n_frames`` frames.

        The delay is measured from the note onset, so a short note simply does
        not get vibrato -- which is what singers do.
        """
        if not self.enabled or n_frames <= 0:
            return np.zeros(n_frames)
        dt = frame_period / 1000.0
        t = np.arange(n_frames) * dt

        rate = float(self.rate_hz)
        depth = float(self.depth_semitones)
        delay = self.delay_ms / 1000.0
        if rng is not None and self.randomness > 0:
            rate *= 1.0 + self.randomness * rng.normal()
            depth *= 1.0 + self.randomness * rng.normal()
            delay *= 1.0 + self.randomness * rng.normal()
        phase = 0.0
        if rng is not None:
            phase = float(rng.uniform(0, 2 * np.pi))

        elapsed = t - delay
        onset = np.clip(elapsed / max(self.attack_ms / 1000.0, 1e-6), 0.0, 1.0)
        waveform = np.sin(2 * np.pi * rate * elapsed + phase)
        if self.waveform == "triangle":
            waveform = 2.0 / np.pi * np.arcsin(np.sin(2 * np.pi * rate * elapsed
                                                       + phase))
        return depth * np.where(elapsed > 0, onset, 0.0) * waveform


class PitchModel:
    """Relative-pitch statistics, vibrato and voicing."""

    def __init__(self, stats: Optional[Dict[str, List[PitchStats]]] = None,
                 vibrato: Optional[Vibrato] = None,
                 voiced_prior: Optional[Dict[str, float]] = None,
                 pitch_variation: float = 1.0) -> None:
        #: phoneme -> per-state PitchStats
        self.stats: Dict[str, List[PitchStats]] = stats or {}
        self.vibrato = vibrato or Vibrato()
        #: phoneme -> probability of being voiced
        self.voiced_prior: Dict[str, float] = voiced_prior or {}
        #: Scale on the sampled deviation from the mean (0 = deterministic).
        self.pitch_variation = float(pitch_variation)

    # -- fitting -----------------------------------------------------------

    def add_state_statistics(self, phoneme: str,
                             per_state_pitch: Sequence[Sequence[np.ndarray]],
                             per_state_voiced: Sequence[Sequence[np.ndarray]]
                             ) -> None:
        """Accumulate per-state relative-pitch statistics.

        ``per_state_pitch[state]`` holds one array of *voiced* relative-pitch
        frames per occurrence of that phoneme that the trained HMM assigned to
        ``state``; ``per_state_voiced`` the matching voicing flags (used for the
        phoneme's voiced prior).
        """
        stats: List[PitchStats] = []
        every_voiced: List[np.ndarray] = []
        for state_pitch, state_voiced in zip(per_state_pitch, per_state_voiced):
            values = [np.asarray(v, dtype=np.float64).reshape(-1)
                      for v in state_pitch if len(np.asarray(v))]
            if values:
                flat = np.concatenate(values)
                flat = flat[np.isfinite(flat)]
            else:
                flat = np.zeros(0)
            if flat.size:
                stats.append(PitchStats(float(flat.mean()),
                                        float(max(flat.var(), 1e-4)),
                                        int(flat.size)))
            else:
                stats.append(PitchStats())
            every_voiced.extend(np.asarray(v, dtype=bool).reshape(-1)
                                for v in state_voiced
                                if len(np.asarray(v)))
        self.stats[phoneme] = stats
        if every_voiced:
            self.voiced_prior[phoneme] = float(np.concatenate(every_voiced).mean())

    def state_means(self, phoneme: str) -> np.ndarray:
        """Per-state mean relative pitch (semitones), with backoff."""
        stats = self.stats.get(phoneme)
        if not stats:
            return np.zeros(0)
        return np.array([s.mean for s in stats])

    def vibrato_defaults_from_data(self, f0: np.ndarray, voiced: np.ndarray,
                                   fs: float,
                                   frame_period: float) -> Optional[dict]:
        """Measure a rough vibrato rate/depth from a sustained segment."""
        estimate = estimate_vibrato(f0, voiced, fs, frame_period)
        if estimate is None:
            return None
        rate, depth = estimate
        self.vibrato.rate_hz = float(np.clip(rate, 3.0, 9.0))
        self.vibrato.depth_semitones = float(np.clip(depth, 0.1, 1.5))
        return {"rate_hz": self.vibrato.rate_hz,
                "depth_semitones": self.vibrato.depth_semitones}

    # -- generation --------------------------------------------------------

    def voiced_mask(self, phones: Sequence[str], state_voiced: np.ndarray,
                    phoneme_voiced: Sequence[bool]) -> np.ndarray:
        """Frames that should carry F0.

        Combines the HMM's learned per-state voicing probability with the
        phoneme's own class: an unvoiced consonant can never be voiced, and a
        vowel or silence is decided by the state statistics.
        """
        mask = np.asarray(state_voiced, dtype=np.float64) > 0.5
        for t, (phone, allowed) in enumerate(zip(phones, phoneme_voiced)):
            if not allowed:
                mask[t] = False
            elif phone in self.voiced_prior and self.voiced_prior[phone] > 0.9:
                mask[t] = True
        return mask

    def generate(self, note_semitones: np.ndarray, voiced: np.ndarray,
                 rng: Optional[np.random.Generator] = None,
                 note_ids: Optional[np.ndarray] = None,
                 frame_period: float = 5.0,
                 relative_trajectory: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
        """Absolute F0 contour in semitones (NaN where unvoiced).

        ``relative_trajectory`` is the acoustic model's generated deviation from
        the note (from the MLPG output, semitones); the note itself comes from
        ``note_semitones``.  When it is ``None`` the per-state means of this
        model are used instead -- the deterministic ``state_means`` mode.
        """
        note_semitones = np.asarray(note_semitones, dtype=np.float64)
        voiced = np.asarray(voiced, dtype=bool)
        variation = (np.asarray(relative_trajectory, dtype=np.float64)
                     if relative_trajectory is not None
                     else np.zeros_like(note_semitones))

        f0 = note_semitones + variation

        if self.vibrato.enabled:
            f0 = f0 + self._vibrato_for_frames(note_ids, voice_mask=voiced,
                                               frame_period=frame_period,
                                               rng=rng)
        return np.where(voiced, f0, np.nan)

    def _vibrato_for_frames(self, note_ids: Optional[np.ndarray],
                            voice_mask: np.ndarray, frame_period: float,
                            rng: Optional[np.random.Generator]) -> np.ndarray:
        n = len(voice_mask)
        if note_ids is None:
            return self.vibrato.contour(n, frame_period, 0, rng)
        out = np.zeros(n, dtype=np.float64)
        note_ids = np.asarray(note_ids)
        for index, note_id in enumerate(dict.fromkeys(note_ids.tolist())):
            where = np.where(note_ids == note_id)[0]
            if len(where) == 0:
                continue
            # vibrato only while the state model says the frame is voiced
            segment = voice_mask[where]
            contour = self.vibrato.contour(len(where), frame_period, index, rng)
            out[where] = np.where(segment, contour, 0.0)
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "pitch_variation": self.pitch_variation,
            "vibrato": self.vibrato.to_dict(),
            "voiced_prior": {k: float(v) for k, v in sorted(
                self.voiced_prior.items())},
            "stats": {phone: [s.to_dict() for s in states]
                      for phone, states in sorted(self.stats.items())},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PitchModel":
        stats = {phone: [PitchStats.from_dict(s) for s in states]
                 for phone, states in (d.get("stats") or {}).items()}
        return cls(stats=stats,
                   vibrato=Vibrato.from_dict(d.get("vibrato") or {}),
                   voiced_prior={k: float(v) for k, v in
                                 (d.get("voiced_prior") or {}).items()},
                   pitch_variation=float(d.get("pitch_variation", 1.0)))


# --------------------------------------------------------------------------
# Training-time transforms
# --------------------------------------------------------------------------


def note_relative_pitch(f0: np.ndarray, note_hz: np.ndarray,
                        voiced: np.ndarray,
                        f_ref: float) -> np.ndarray:
    """Absolute F0 (Hz) -> note-relative semitones, gaps interpolated.

    Unvoiced frames keep a smooth interpolated value instead of a meaningless
    zero, so the acoustic model is not asked to memorise noise.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    note_hz = np.asarray(note_hz, dtype=np.float64)
    relative = np.zeros(len(f0), dtype=np.float64)
    if len(f0) == 0:
        return relative
    safe_note = np.where(note_hz > 0, note_hz, 440.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(voiced & (f0 > 0), f0, np.nan) / safe_note
    relative[:] = 12.0 * np.log2(ratio)
    return interpolate_gaps(relative)


def interpolate_gaps(values: np.ndarray, valid: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """Linearly interpolate NaN entries; constant extrapolation at the edges."""
    values = np.asarray(values, dtype=np.float64).copy()
    mask = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool)
    if not mask.any():
        return np.zeros_like(values)
    if mask.all():
        return values
    index = np.arange(len(values))
    values[~mask] = np.interp(index[~mask], index[mask], values[mask])
    return values


def absolute_semitones(relative: np.ndarray, note_hz: np.ndarray,
                       f_ref: float) -> np.ndarray:
    """Note-relative semitones + note -> absolute semitones re. ``f_ref``."""
    relative = np.asarray(relative, dtype=np.float64)
    note_hz = np.asarray(note_hz, dtype=np.float64)
    safe = np.where(note_hz > 0, note_hz, np.nan)
    return 12.0 * np.log2(safe / f_ref) + relative


def estimate_vibrato(f0: np.ndarray, voiced: np.ndarray, fs: float,
                     frame_period: float) -> Optional[Tuple[float, float]]:
    """Rough (rate_hz, depth_semitones) estimate from a F0 track.

    Works on the longest voiced run: removes the local mean, finds the dominant
    periodicity of the residual within 3-9 Hz, and reports its amplitude.  Used
    to seed `Vibrato` defaults from a training set; not used in training itself.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    if voiced.sum() < 50:
        return None
    # longest voiced run
    runs, start = [], None
    for i, flag in enumerate(voiced):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(voiced)))
    if not runs:
        return None
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    segment = f0[lo:hi]
    if len(segment) < 40:
        return None

    semitones = 12.0 * np.log2(np.maximum(segment, 1e-6) / np.median(segment))
    # remove a slow trend (the note, and any portamento)
    window = max(3, int(round(0.25 * 1000.0 / frame_period)))
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    trend = np.convolve(semitones, kernel, mode="same")
    residual = semitones - trend
    residual = residual[window:len(residual) - window] if len(residual) > 3 * window \
        else residual
    if len(residual) < 20:
        return None

    spectrum = np.abs(np.fft.rfft(residual * np.hanning(len(residual))))
    freqs = np.fft.rfftfreq(len(residual), d=frame_period / 1000.0)
    band = (freqs >= 3.0) & (freqs <= 9.0)
    if not band.any():
        return None
    peak = int(np.argmax(spectrum[band]))
    rate = float(freqs[band][peak])
    depth = float(np.std(residual) * np.sqrt(2.0))
    return rate, depth
