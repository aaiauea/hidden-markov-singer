"""Generate a small example corpus with a *formant synthesizer*.

This module is a **data source**, not part of the engine.  It exists because HMS
ships without any audio, and a singing synthesizer you cannot run is not much of
a deliverable: `hms demo` writes a handful of WAVs plus matching label files, so
the whole pipeline (extract -> train -> synth) can be exercised in seconds.

Note the division of responsibility, because it is deliberate and mirrors the
design rule "do not hardcode vowel formants in the engine":

* the engine (`hms.core`, `hms.vocoder`) has **no** knowledge of formants or
  vowel identities.  It only ever sees WORLD parameters and label files;
* this file knows the formants of a synthetic *singer* -- exactly the way a real
  recording would -- and emits it as audio plus labels.

The generated voice is a source-filter model: a glottal pulse train with jitter
and a noise floor, filtered by a cascade of two-pole resonators.  Stops get a
closure/burst, fricatives get noise, nasals get a low-frequency murmur.  It has
a mild onset scoop and vibrato, so note-relative pitch modelling has something
to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms.data import wavio

# --------------------------------------------------------------------------
# The synthetic singer's vocal tract (demo data only)
# --------------------------------------------------------------------------

#: (F1..F4) in Hz and bandwidths for the demo vowels.
VOWEL_FORMANTS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "a": ((800, 1200, 2500, 3500), (90, 110, 140, 180)),
    "e": ((500, 1900, 2500, 3400), (80, 100, 140, 180)),
    "i": ((300, 2300, 3000, 3600), (70, 100, 150, 190)),
    "o": ((500, 900, 2400, 3300), (80, 100, 140, 180)),
    "u": ((320, 800, 2400, 3300), (70, 100, 140, 180)),
}

#: Consonant classes for the demo inventory.
CONSONANTS: Dict[str, dict] = {
    "m": {"class": "nasal", "formants": ((250, 1100, 2300, 3200), (80, 120, 150, 180))},
    "n": {"class": "nasal", "formants": ((250, 1700, 2600, 3300), (80, 120, 150, 180))},
    "l": {"class": "liquid", "formants": ((400, 1000, 2600, 3400), (80, 110, 150, 180))},
    "r": {"class": "liquid", "formants": ((450, 1200, 1600, 3300), (90, 120, 160, 180))},
    "v": {"class": "voiced_fricative", "formants": ((400, 1100, 2200, 3200),
                                                    (90, 130, 200, 220)),
          "noise": 0.25, "noise_band": (1200.0, 6000.0)},
    "z": {"class": "voiced_fricative", "formants": ((350, 1600, 2600, 3400),
                                                    (90, 130, 200, 220)),
          "noise": 0.35, "noise_band": (3000.0, 9000.0)},
    "s": {"class": "unvoiced_fricative", "formants": ((500, 1600, 2600, 3400),
                                                      (200, 300, 400, 400)),
          "noise": 1.0, "noise_band": (4000.0, 12000.0)},
    "f": {"class": "unvoiced_fricative", "formants": ((500, 1400, 2400, 3300),
                                                      (200, 300, 400, 400)),
          "noise": 1.0, "noise_band": (1500.0, 7000.0)},
    "k": {"class": "stop", "formants": ((400, 1800, 2600, 3400), (120, 200, 250, 250)),
          "burst": (1500.0, 6000.0), "burst_ms": 12.0, "closure_ratio": 0.5},
    "t": {"class": "stop", "formants": ((400, 1900, 2700, 3400), (120, 200, 250, 250)),
          "burst": (3000.0, 9000.0), "burst_ms": 8.0, "closure_ratio": 0.5},
    "p": {"class": "stop", "formants": ((400, 1100, 2200, 3300), (120, 200, 250, 250)),
          "burst": (800.0, 3000.0), "burst_ms": 6.0, "closure_ratio": 0.6},
    "h": {"class": "aspiration", "formants": ((500, 1500, 2500, 3400),
                                              (300, 400, 500, 500)),
          "noise": 1.0, "noise_band": (500.0, 6000.0)},
    "sil": {"class": "silence"},
}


@dataclass
class SegmentSpec:
    """One segment to render: phoneme, duration in ms, MIDI note (None = unvoiced)."""

    phone: str
    duration_ms: float
    note: Optional[float] = None


@dataclass
class SingerConfig:
    """Knobs of the demo voice (all demo-side; nothing here reaches the model)."""

    fs: int = 44100
    jitter: float = 0.004          # period perturbation (shimmer/jitter)
    noise_floor: float = 0.0015    # breath noise, keeps WORLD's D4C happy
    vibrato_hz: float = 5.4
    vibrato_semitones: float = 0.5
    vibrato_delay_ms: float = 150.0
    scoop_semitones: float = -0.6  # start below the target and slide up
    scoop_ms: float = 70.0
    drift_semitones: float = 0.15  # slow, phrase-long pitch drift
    tilt_db_per_octave: float = -9.0
    seed: int = 7

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


class DemoSinger:
    """Renders a phoneme/note script to audio."""

    def __init__(self, config: Optional[SingerConfig] = None) -> None:
        self.config = config or SingerConfig()
        self.rng = np.random.default_rng(self.config.seed)

    # -- public API --------------------------------------------------------

    def render(self, script: Sequence[SegmentSpec]) -> np.ndarray:
        """Render segments back to back into one waveform."""
        config = self.config
        dt = 1.0 / config.fs
        durations = np.array([s.duration_ms for s in script], dtype=np.float64)
        if durations.sum() <= 0:
            return np.zeros(0)
        boundaries = np.concatenate([[0.0], np.cumsum(durations)]) / 1000.0
        total = int(round(boundaries[-1] * config.fs))
        f0 = self._f0_contour(script, boundaries, dt, total)

        audio = np.zeros(total + config.fs, dtype=np.float64)   # room for tails
        for index, segment in enumerate(script):
            start = int(round(boundaries[index] * config.fs))
            end = int(round(boundaries[index + 1] * config.fs))
            if end <= start:
                continue
            piece = self._render_segment(
                segment, f0[start:end], start / config.fs, end - start)
            audio[start:end] += piece

        # short crossfades remove the clicks between segments
        audio = self._crossfade(audio, boundaries, crossfade_ms=8.0)
        audio = audio[:total]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.85
        return audio

    def label_lines(self, name: str, script: Sequence[SegmentSpec]
                    ) -> List[str]:
        """Ground-truth labels for a script, in the HMS label format."""
        lines = []
        cursor = 0.0
        for segment in script:
            start = cursor / 1000.0
            cursor += segment.duration_ms
            end = cursor / 1000.0
            note = "-" if segment.note is None else f"{segment.note:g}"
            lines.append(f"{name}\t{start:.4f}\t{end:.4f}\t{segment.phone}\t{note}")
        return lines

    # -- internals ---------------------------------------------------------

    def _f0_contour(self, script: Sequence[SegmentSpec],
                    boundaries: np.ndarray, dt: float,
                    total: int) -> np.ndarray:
        """Continuous F0 (Hz) with scoop, drift and vibrato."""
        config = self.config
        times = np.arange(total) * dt
        notes = []
        for index, segment in enumerate(script):
            note = segment.note
            if note is None:
                note = notes[-1] if notes else 60.0
            notes.append(note)
        notes = np.array(notes, dtype=np.float64)

        f0 = np.zeros(total, dtype=np.float64)
        for index, note in enumerate(notes):
            start, end = boundaries[index], boundaries[index + 1]
            mask = (times >= start) & (times < end)
            if not mask.any():
                continue
            local = times[mask] - start
            base = 440.0 * 2.0 ** ((note - 69.0) / 12.0)

            # onset scoop: start low, slide to the note
            scoop_time = max(config.scoop_ms / 1000.0, 1e-3)
            progress = np.clip(local / scoop_time, 0.0, 1.0)
            scoop = config.scoop_semitones * (1.0 - progress) ** 2

            # phrase drift + vibrato with a delay
            drift = config.drift_semitones * np.sin(
                2 * np.pi * 0.17 * times[mask])
            after = local - config.vibrato_delay_ms / 1000.0
            attack = np.clip(after / 0.12, 0.0, 1.0)
            vibrato = config.vibrato_semitones * attack * np.sin(
                2 * np.pi * config.vibrato_hz * after)

            semitones = scoop + drift + vibrato
            f0[mask] = base * 2.0 ** (semitones / 12.0)

        # portamento across note boundaries (a singer slides, they don't jump)
        glide = max(1, int(0.03 * config.fs))
        for boundary in boundaries[1:-1]:
            index = int(boundary * config.fs)
            lo, hi = max(0, index - glide), min(total, index + glide)
            if hi - lo < 2:
                continue
            previous = f0[lo - 1] if lo > 0 else f0[lo]
            target = f0[hi] if hi < total else f0[hi - 1]
            f0[lo:hi] = np.linspace(previous, target, hi - lo)
        return f0

    def _articulation(self, phone: str) -> dict:
        """Tract configuration for a phone: vowels come from VOWEL_FORMANTS."""
        if phone in CONSONANTS:
            return CONSONANTS[phone]
        if phone in VOWEL_FORMANTS:
            formants, bandwidths = VOWEL_FORMANTS[phone]
            return {"class": "vowel", "formants": (formants, bandwidths)}
        raise KeyError(f"demo singer has no definition for {phone!r}")

    def _render_segment(self, segment: SegmentSpec, f0: np.ndarray,
                        offset_seconds: float, n_samples: int) -> np.ndarray:
        spec = self._articulation(segment.phone)
        if spec["class"] == "silence":
            return np.zeros(n_samples)

        source = np.zeros(n_samples, dtype=np.float64)
        unvoiced_source = self.rng.standard_normal(n_samples)

        if spec["class"] in ("unvoiced_fricative", "aspiration"):
            source = unvoiced_source * 0.35
        elif spec["class"] == "stop":
            closure = int(n_samples * spec.get("closure_ratio", 0.5))
            burst_len = int(spec["burst_ms"] / 1000.0 * self.config.fs)
            source = np.zeros(n_samples)
            lo, hi = closure, min(n_samples, closure + burst_len)
            source[lo:hi] = unvoiced_source[lo:hi] * 0.9
        else:
            if len(f0):
                source = self._glottal_source(f0, n_samples)
            else:
                source = self._glottal_source(
                    np.full(n_samples, 200.0), n_samples)
            if spec.get("noise"):
                source = source + unvoiced_source * spec["noise"] * 0.25

        formants, bandwidths = spec["formants"]
        filtered = self._formant_filter(source, formants, bandwidths)
        if spec["class"] in ("unvoiced_fricative", "aspiration") \
                or spec.get("noise_band") and spec["class"] == "stop":
            band = spec.get("noise_band") or spec.get("burst")
            filtered = self._band_shape(filtered, band)
        return filtered * self._segment_gain(spec["class"])

    def _glottal_source(self, f0: np.ndarray, n_samples: int) -> np.ndarray:
        """Impulse train convolved with a glottal pulse shape, plus jitter."""
        config = self.config
        out = np.zeros(n_samples, dtype=np.float64)
        pulse_len = max(4, int(0.6 * config.fs / max(np.max(f0), 1e-6)))
        pulse = self._glottal_pulse(pulse_len)

        position = 0.0
        index = 0
        while index < n_samples:
            period = config.fs / max(f0[min(index, len(f0) - 1)], 1e-6)
            period *= 1.0 + config.jitter * self.rng.standard_normal()
            start = int(round(position))
            if start >= n_samples:
                break
            stop = min(n_samples, start + len(pulse))
            out[start:stop] += pulse[:stop - start]
            position += max(period, 2.0)
            index = int(position)
        out += self.rng.standard_normal(n_samples) * config.noise_floor
        return out

    @staticmethod
    def _glottal_pulse(n: int) -> np.ndarray:
        """A simple Rosenberg-style glottal pulse (low-passed impulse)."""
        t = np.linspace(0.0, 1.0, n)
        opening = np.sin(np.pi * np.clip(t / 0.4, 0, 1)) ** 2
        closing = np.exp(-((t - 0.4) / 0.12) ** 2)
        pulse = np.where(t < 0.4, opening, closing)
        pulse = np.diff(pulse, prepend=pulse[0])       # radiation at the lips
        return pulse / max(np.max(np.abs(pulse)), 1e-9)

    def _formant_filter(self, source: np.ndarray, formants: Sequence[float],
                        bandwidths: Sequence[float]) -> np.ndarray:
        """Cascade of two-pole resonators, plus a spectral tilt."""
        dt = 1.0 / self.config.fs
        out = source.copy()
        for frequency, bandwidth in zip(formants, bandwidths):
            if frequency <= 0 or frequency >= self.config.fs / 2:
                continue
            r = np.exp(-np.pi * bandwidth * dt)
            theta = 2 * np.pi * frequency * dt
            a1 = -2 * r * np.cos(theta)
            a2 = r * r
            gain = (1 - a1 - a2) / 2.0
            out = self._iir(out, [gain], [1.0, a1, a2])
        # spectral tilt: -X dB per octave around 500 Hz
        tilt = self.config.tilt_db_per_octave
        if tilt:
            out = self._tilt(out, tilt)
        return out

    @staticmethod
    def _iir(x: np.ndarray, b: Sequence[float], a: Sequence[float]
             ) -> np.ndarray:
        """Direct-form I biquad (single pass, Python loop over taps only)."""
        y = np.zeros_like(x)
        b = list(b)
        a = list(a)
        for n in range(len(x)):
            acc = 0.0
            for k, coefficient in enumerate(b):
                if n - k >= 0:
                    acc += coefficient * x[n - k]
            for k, coefficient in enumerate(a[1:], start=1):
                if n - k >= 0:
                    acc -= coefficient * y[n - k]
            y[n] = acc
        return y

    def _tilt(self, x: np.ndarray, db_per_octave: float) -> np.ndarray:
        """Cheap spectral tilt by mixing the signal with its running mean."""
        window = max(3, int(self.config.fs / 4000.0))
        smooth = np.convolve(x, np.ones(window) / window, mode="same")
        weight = min(abs(db_per_octave) / 24.0, 0.9)
        return x - weight * smooth

    def _band_shape(self, x: np.ndarray, band: Sequence[float]) -> np.ndarray:
        lo, hi = float(band[0]), float(band[1])
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.config.fs)
        shape = np.exp(-0.5 * ((np.log(np.maximum(freqs, 1.0))
                                - np.log(np.sqrt(lo * hi)))
                               / (0.5 * np.log(hi / lo) + 0.7)) ** 2)
        return np.fft.irfft(spectrum * shape, len(x))

    @staticmethod
    def _segment_gain(klass: str) -> float:
        return {"nasal": 0.5, "liquid": 0.8, "voiced_fricative": 0.6,
                "unvoiced_fricative": 0.5, "stop": 0.7, "aspiration": 0.4
                }.get(klass, 1.0)

    def _crossfade(self, audio: np.ndarray, boundaries: np.ndarray,
                   crossfade_ms: float) -> np.ndarray:
        out = audio.copy()
        length = max(2, int(crossfade_ms / 1000.0 * self.config.fs))
        for boundary in boundaries[1:-1]:
            index = int(boundary * self.config.fs)
            lo, hi = max(1, index - length // 2), min(len(out) - 1,
                                                      index + length // 2)
            if hi - lo < 4:
                continue
            ramp = np.linspace(0.5, 1.0, hi - lo)
            out[lo:hi] *= ramp
        return out


# --------------------------------------------------------------------------
# The bundled example corpus
# --------------------------------------------------------------------------


def default_phrases() -> List[Tuple[str, List[SegmentSpec]]]:
    """The example corpus: short sung phrases covering the demo inventory."""
    def s(phone: str, ms: float, note: Optional[float] = None) -> SegmentSpec:
        return SegmentSpec(phone, ms, note)

    phrases: List[Tuple[str, List[SegmentSpec]]] = []

    # sustained vowels on different notes (the core of the pitch model)
    phrases.append(("vowel_scale_a", [
        s("sil", 120), s("a", 520, 60), s("sil", 90), s("a", 500, 62),
        s("sil", 90), s("a", 520, 64), s("sil", 160)]))
    phrases.append(("vowel_scale_i", [
        s("sil", 120), s("i", 480, 60), s("sil", 80), s("i", 460, 65),
        s("sil", 80), s("i", 500, 67), s("sil", 160)]))
    phrases.append(("vowel_scale_u", [
        s("sil", 120), s("u", 500, 55), s("sil", 80), s("u", 480, 59),
        s("sil", 80), s("u", 520, 62), s("sil", 160)]))
    phrases.append(("vowel_scale_o", [
        s("sil", 120), s("o", 500, 57), s("sil", 80), s("o", 480, 60),
        s("sil", 80), s("o", 500, 64), s("sil", 160)]))
    phrases.append(("vowel_scale_e", [
        s("sil", 120), s("e", 480, 59), s("sil", 80), s("e", 480, 62),
        s("sil", 80), s("e", 500, 66), s("sil", 160)]))

    # syllables: consonant + vowel transitions
    phrases.append(("syllables_ma", [
        s("sil", 110), s("m", 90, 60), s("a", 380, 60),
        s("m", 80, 64), s("a", 360, 64), s("m", 80, 67), s("a", 400, 67),
        s("sil", 150)]))
    phrases.append(("syllables_la", [
        s("sil", 110), s("l", 80, 62), s("a", 360, 62),
        s("l", 70, 65), s("a", 340, 65), s("l", 70, 69), s("a", 380, 69),
        s("sil", 150)]))
    phrases.append(("syllables_na", [
        s("sil", 110), s("n", 80, 57), s("a", 360, 57),
        s("n", 70, 60), s("a", 340, 60), s("n", 70, 64), s("a", 380, 64),
        s("sil", 150)]))
    phrases.append(("syllables_sa", [
        s("sil", 110), s("s", 110), s("a", 360, 62),
        s("s", 100), s("a", 340, 65), s("s", 100), s("a", 380, 67),
        s("sil", 150)]))
    phrases.append(("syllables_fa", [
        s("sil", 110), s("f", 100), s("a", 360, 60),
        s("f", 90), s("a", 340, 64), s("f", 90), s("a", 380, 66),
        s("sil", 150)]))
    phrases.append(("syllables_ta", [
        s("sil", 110), s("t", 80), s("a", 360, 62),
        s("t", 70), s("a", 340, 65), s("t", 70), s("a", 380, 68),
        s("sil", 150)]))
    phrases.append(("syllables_ka", [
        s("sil", 110), s("k", 90), s("a", 360, 59),
        s("k", 80), s("a", 340, 63), s("k", 80), s("a", 380, 66),
        s("sil", 150)]))
    phrases.append(("syllables_va", [
        s("sil", 110), s("v", 90, 60), s("a", 360, 60),
        s("v", 80, 63), s("a", 340, 63), s("v", 80, 67), s("a", 380, 67),
        s("sil", 150)]))
    phrases.append(("syllables_za", [
        s("sil", 110), s("z", 90, 58), s("a", 360, 58),
        s("z", 80, 62), s("a", 340, 62), s("z", 80, 65), s("a", 380, 65),
        s("sil", 150)]))

    # a small melodic phrase: the kind of thing you would actually sing
    phrases.append(("phrase_melody", [
        s("sil", 120), s("m", 90, 60), s("a", 420, 60),
        s("l", 80, 64), s("a", 380, 64), s("l", 80, 67), s("a", 400, 67),
        s("n", 80, 69), s("a", 460, 69), s("sil", 200)]))
    phrases.append(("phrase_rise", [
        s("sil", 120), s("s", 100), s("a", 300, 62), s("s", 90),
        s("a", 300, 65), s("s", 90), s("a", 300, 69), s("s", 90),
        s("a", 480, 72), s("sil", 200)]))
    phrases.append(("phrase_low", [
        s("sil", 120), s("n", 80, 53), s("a", 420, 53),
        s("m", 80, 55), s("a", 380, 55), s("n", 80, 58), s("a", 420, 58),
        s("sil", 200)]))
    phrases.append(("phrase_long_note", [
        s("sil", 120), s("o", 900, 64), s("u", 700, 62), s("sil", 220)]))

    return phrases


def make_dataset(out_dir, fs: int = 44100,
                 singer: Optional[SingerConfig] = None,
                 label_jitter_ms: float = 0.0,
                 seed: int = 7,
                 phrases: Optional[Sequence[Tuple[str, List[SegmentSpec]]]] = None,
                 renderer: Optional[DemoSinger] = None) -> dict:
    """Write the example corpus to ``out_dir``.

    Creates ``wav/<name>.wav``, ``labels.tsv`` (ground truth, for training) and
    ``score.tsv`` (the same segments, for synthesis).  Returns a summary dict.

    ``phrases`` (default: :func:`default_phrases`) and ``renderer`` let callers
    build a different corpus with the same file layout -- a second voice, or a
    second language with a different phoneme inventory -- which is what the
    cross-language transfer tests and examples need.
    """
    out_dir = Path(out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    config = singer or SingerConfig(fs=fs, seed=seed)
    config.fs = fs
    renderer = renderer if renderer is not None else DemoSinger(config)
    phrases = list(phrases) if phrases is not None else default_phrases()
    rng = np.random.default_rng(seed + 1)

    label_lines: List[str] = ["# utt_id\tonset\toffset\tphone\tnote"]
    score_lines: List[str] = ["# utt_id\tonset\toffset\tphone\tnote"]
    total_seconds = 0.0
    n_segments = 0

    for name, script in phrases:
        audio = renderer.render(script)
        wavio.write_wav(wav_dir / f"{name}.wav", audio, fs)

        lines = renderer.label_lines(name, script)
        total_seconds += len(audio) / fs
        n_segments += len(script)
        score_lines.extend(lines)
        if label_jitter_ms > 0:
            lines = _jitter_lines(lines, label_jitter_ms, rng)
        label_lines.extend(lines)   # training labels may be deliberately noisy

    (out_dir / "labels.tsv").write_text("\n".join(label_lines) + "\n",
                                        encoding="utf-8")
    (out_dir / "score.tsv").write_text("\n".join(score_lines) + "\n",
                                       encoding="utf-8")
    return {
        "utterances": len(phrases),
        "segments": n_segments,
        "seconds": total_seconds,
        "wav_dir": str(wav_dir),
        "labels": str(out_dir / "labels.tsv"),
        "score": str(out_dir / "score.tsv"),
    }


def _jitter_lines(lines: Sequence[str], jitter_ms: float,
                  rng: np.random.Generator) -> List[str]:
    """Perturb label boundaries the way a human labeller would."""
    parsed = []
    for line in lines:
        parts = line.split("\t")
        parsed.append((parts[0], float(parts[1]), float(parts[2]), parts[3],
                       parts[4]))
    out = []
    durations = [0.0] * len(parsed)
    for i, (_, start, end, _, _) in enumerate(parsed):
        durations[i] = max(0.01, end - start)
    bounds = [parsed[0][1]]
    for duration in durations:
        bounds.append(bounds[-1] + duration
                      + rng.normal(0.0, jitter_ms / 1000.0))
    for i, (name, _, _, phone, note) in enumerate(parsed):
        out.append(f"{name}\t{bounds[i]:.4f}\t{bounds[i + 1]:.4f}\t{phone}\t{note}")
    return out
