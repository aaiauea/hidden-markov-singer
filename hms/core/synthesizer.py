"""Synthesis pipeline: score -> audio.

    score (phonemes + notes + durations)
        -> per-phoneme state allocation from the requested durations
        -> per-frame model statistics (mean, variance) per state
        -> MLPG trajectory generation
        -> note-conditioned F0 (+ learned deviation + optional vibrato)
        -> voicing mask from the learned state voicing probabilities
        -> WORLD parameters -> waveform

Each stage is independently replaceable:

* `hms.core.duration.DurationModel.allocate` decides the state sequence;
* `hms.core.generation.mlpg` turns state statistics into a trajectory;
* `hms.core.pitch.PitchModel` turns the trajectory + score notes into F0;
* `hms.vocoder` turns (f0, sp, ap) into samples.

Options that matter in practice (all in `SynthesisConfig`):

``variance_scale``
    Scales the *dynamic* (delta) variances.  >1 weakens the delta constraints,
    so the trajectory follows the per-frame means more literally (more detail,
    livelier); <1 strengthens them and flattens/smooths the trajectory.  1.0
    reproduces the model.
``pitch_variation``
    0 makes F0 exactly the note plus the model's mean deviation
    (deterministic, useful for testing and for "straight" singing); 1 uses the
    learned variation as-is.  Values >1 exaggerate it.
``f0_source``
    ``acoustic`` (default) takes F0 from the generated trajectory -- i.e. the
    HMM's own model of this singer's pitch habits; ``state_means`` rebuilds the
    contour from the per-(phoneme, state) pitch statistics instead, which is
    blunter but more predictable for tiny models.
``duration_mode``
    ``score`` honours the score's segment boundaries (normal use);
    ``model`` asks the duration model to predict them, for scores that only
    list phonemes and notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms.core import labels as labels_module
from hms.core.duration import DurationModel
from hms.core.features import AcousticFrameSequence
from hms.core.generation import mlpg, stack_streams
from hms.core.model import HMSModel


@dataclass
class SynthesisConfig:
    """Knobs for one synthesis run."""

    variance_scale: float = 1.0
    pitch_variation: float = 1.0
    f0_source: str = "acoustic"        # "acoustic" | "state_means"
    duration_mode: str = "score"       # "score" | "model"
    tempo: float = 1.0
    vibrato: Optional[bool] = None     # None -> whatever the model says
    vibrato_depth: Optional[float] = None
    vibrato_rate: Optional[float] = None
    transpose: float = 0.0             # semitones added to every note
    smooth: bool = True
    seed: Optional[int] = None
    mixture: str = "dominant"          # "dominant" | "marginal"
    vocoder: Optional[str] = None      # override the backend
    #: inter-phoneme smoothing of the generated F0, in frames (0 = off)
    pitch_smoothing: int = 0


@dataclass
class SynthesisResult:
    """Rendered audio plus everything that produced it."""

    audio: np.ndarray
    params: AcousticFrameSequence
    phones: List[str]
    notes: np.ndarray                  # per frame MIDI note
    state_ids: np.ndarray              # per frame (state index within phoneme)
    state_sequence: List[str]          # per frame, "phone/state"
    durations: np.ndarray              # per frame phoneme segment index
    f0_semitones: np.ndarray           # absolute, NaN when unvoiced
    diagnostics: List[str]

    @property
    def duration(self) -> float:
        return float(len(self.audio) / self.params.fs)


class Synthesizer:
    """Renders a score with a trained :class:`HMSModel`."""

    def __init__(self, model: HMSModel, config: Optional[SynthesisConfig] = None,
                 log=None) -> None:
        self.model = model
        self.config = config or SynthesisConfig()
        self.log = log or (lambda message: None)
        self._vocoder = None

    @property
    def vocoder(self):
        if self._vocoder is None:
            from hms.vocoder import get_vocoder
            self._vocoder = get_vocoder(self.config.vocoder or "auto",
                                        fft_size=self.model.spec.fft_size)
        return self._vocoder

    # -- plan the state sequence ------------------------------------------

    def plan(self, score: labels_module.Score, default_note: float = 60.0
             ) -> Tuple[List[str], List[int], List[int], np.ndarray,
                        List[int], List[str]]:
        """Turn a score into (phone, segment, frame) aligned state indices.

        Returns ``(frame_phones, state_ids, segment_ids, note_per_frame,
        segment_frames, diagnostics)`` -- the diagnostics list is returned so
        the caller can show it, instead of being written into hidden state.
        Notes are MIDI numbers (float, so fractional detuning is allowed).
        """
        config = self.config
        spec = self.model.spec
        frame_phones: List[str] = []
        state_ids: List[int] = []
        segment_ids: List[int] = []
        notes: List[float] = []
        segment_frames: List[int] = []
        diagnostics: List[str] = []

        for utterance in score:
            if config.duration_mode == "model" or utterance.end <= utterance.start:
                # no usable timing in the score: let the duration model decide
                phones = [s.phone for s in utterance.segments] or ["sil"]
                predicted = self.model.duration_model.predict(
                    phones, spec.frame_period, tempo=config.tempo,
                    rng=np.random.default_rng(config.seed), speak=False)
                segments = [(phone, int(max(round(frames), 1)),
                             segment.note)
                            for phone, frames, segment
                            in zip(phones, predicted, utterance.segments)]
            else:
                spans = labels_module.segment_boundaries(
                    utterance, spec.frame_period)
                segments = []
                previous_end = 0
                for phone, lo, hi, note in spans:
                    if lo > previous_end:
                        # a gap in the labels is silence
                        segments.append((self.model.phoneme_set.silence,
                                         lo - previous_end, None))
                    segments.append((phone, max(1, hi - lo), note))
                    previous_end = hi

            for index, (phone, frames, note) in enumerate(segments):
                hmm = self.model.get_or_backoff(phone)
                if self.model.get_hmm(phone) is None:
                    diagnostics.append(
                        f"{utterance.name}: phoneme {phone!r} is not in the "
                        f"model; using the backoff model")
                counts = DurationModel.allocate(frames, hmm.duration_proportions())
                for state, count in enumerate(counts):
                    frame_phones.extend([phone] * int(count))
                    state_ids.extend([state] * int(count))
                    segment_ids.extend([index] * int(count))
                    notes.extend([default_note if note is None else float(note)]
                                 * int(count))
                segment_frames.append(int(sum(counts)))

        note_per_frame = np.asarray(notes, dtype=np.float64)
        if self.config.transpose:
            note_per_frame = note_per_frame + float(self.config.transpose)
        return (frame_phones, state_ids, segment_ids, note_per_frame,
                segment_frames, diagnostics)

    # -- per-frame statistics ---------------------------------------------

    def frame_statistics(self, frame_phones: Sequence[str],
                         state_ids: Sequence[int]
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Stacked (T * n_streams, D) means and variances for MLPG."""
        spec = self.model.spec
        dim = spec.dim
        n_frames = len(frame_phones)

        means = np.zeros((n_frames, dim), dtype=np.float64)
        variances = np.zeros((n_frames, dim), dtype=np.float64)
        cache: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        use_dominant = self.config.mixture == "dominant"

        for t in range(n_frames):
            phone = frame_phones[t]
            state = int(state_ids[t])
            key = (phone, state)
            if key not in cache:
                hmm = self.model.get_or_backoff(phone)
                state = min(state, hmm.n_states - 1)
                gmm = hmm.states[state].gmm
                cache[key] = gmm.predictive_mean(1.0, use_dominant=use_dominant)
            mean, variance = cache[key]
            means[t] = mean
            variances[t] = variance

        # per-state values are in normalised feature space; MLPG works there
        return (stack_streams(means, spec.stream_sizes),
                stack_streams(variances, spec.stream_sizes))

    # -- voicing and pitch -------------------------------------------------

    def voicing(self, frame_phones: Sequence[str], state_ids: Sequence[int]
                ) -> np.ndarray:
        """Per-frame voiced mask from the learned state/phone statistics."""
        voiced = np.zeros(len(frame_phones), dtype=bool)
        for t, (phone, state) in enumerate(zip(frame_phones, state_ids)):
            hmm = self.model.get_or_backoff(phone)
            state = min(int(state), hmm.n_states - 1)
            probability = hmm.states[state].voiced_prob
            definition = self.model.phoneme_set.resolve(phone)
            if definition is not None and not definition.voiced:
                voiced[t] = False
            elif definition is not None and definition.voiced \
                    and self.model.pitch_model.voiced_prior.get(phone, 1.0) > 0.9:
                voiced[t] = True
            else:
                voiced[t] = probability > 0.5
        return voiced

    def pitch(self, static: np.ndarray, note_per_frame: np.ndarray,
              voiced: np.ndarray, phones: Sequence[str],
              state_ids: Sequence[int], segment_ids: Sequence[int],
              rng: np.random.Generator) -> np.ndarray:
        """Absolute F0 in semitones (NaN where unvoiced).

        F0 = score note
             + learned deviation from the model (MLPG trajectory of feature 0)
             + optional vibrato (explicit component, if enabled)

        The note is the anchor: the training speaker's absolute pitch is never
        copied, only the statistics of how they move *around* a note.
        """
        spec = self.model.spec
        config = self.config
        note_semitones = 12.0 * np.log2(
            labels_module.midi_to_hz(note_per_frame) / spec.f0_ref_hz)

        if config.f0_source == "state_means":
            # rebuild the contour from per-(phoneme, state) pitch statistics,
            # then smooth it across state boundaries
            series = np.zeros(len(static), dtype=np.float64)
            for index, phone in enumerate(phones):
                means = self.model.pitch_model.state_means(phone)
                if len(means):
                    state = min(int(state_ids[index]), len(means) - 1)
                    series[index] = means[state]
            relative = np.convolve(series, np.ones(5) / 5.0, mode="same") \
                if len(series) >= 5 else series
            if len(relative):
                relative = relative - relative.mean()
        else:
            relative = static[:, 0] * float(config.pitch_variation)

        vibrato_backup = self.model.pitch_model.vibrato.enabled
        if config.vibrato is not None:
            self.model.pitch_model.vibrato.enabled = bool(config.vibrato)
        if config.vibrato_depth is not None:
            self.model.pitch_model.vibrato.depth_semitones = \
                float(config.vibrato_depth)
        if config.vibrato_rate is not None:
            self.model.pitch_model.vibrato.rate_hz = float(config.vibrato_rate)

        f0 = self.model.pitch_model.generate(
            note_semitones, voiced, rng=rng, note_ids=np.asarray(segment_ids),
            frame_period=spec.frame_period, relative_trajectory=relative)
        self.model.pitch_model.vibrato.enabled = vibrato_backup

        if config.pitch_smoothing > 0:
            f0 = self._smooth_nan(f0, int(config.pitch_smoothing))
        return f0

    def _clamp_pitch(self, f0_semitones: np.ndarray
                     ) -> Tuple[np.ndarray, int]:
        """Keep F0 inside the analysed range.

        Notes above the ceiling (easy to reach by transposing) would otherwise
        be decoded as unvoiced frames -- the vocoder has no spectral envelope
        up there and the model was never trained on them -- which sounds like
        the top of the melody dropping out.  Clamping keeps the note audible
        and reports it instead.
        """
        spec = self.model.spec
        low = 12.0 * np.log2(max(spec.f0_floor, 1e-3) / spec.f0_ref_hz)
        high = 12.0 * np.log2(spec.f0_ceil / spec.f0_ref_hz)
        voiced = np.isfinite(f0_semitones)
        clamped = np.where(voiced, np.clip(f0_semitones, low, high),
                           f0_semitones)
        count = int(np.sum(voiced & ((f0_semitones < low)
                                     | (f0_semitones > high))))
        return clamped, count

    @staticmethod
    def _smooth_nan(values: np.ndarray, width: int) -> np.ndarray:
        """Moving average that ignores unvoiced (NaN) frames."""
        if width <= 1 or len(values) < 2:
            return values
        kernel = np.ones(2 * width + 1)
        valid = np.isfinite(values)
        filled = np.where(valid, values, 0.0)
        numerator = np.convolve(filled, kernel, mode="same")
        denominator = np.convolve(valid.astype(float), kernel, mode="same")
        smoothed = numerator / np.maximum(denominator, 1e-9)
        return np.where(valid, smoothed, np.nan)

    # -- main entry point --------------------------------------------------

    def synthesize(self, score: labels_module.Score,
                   default_note: float = 60.0) -> SynthesisResult:
        config = self.config
        spec = self.model.spec
        rng = np.random.default_rng(config.seed)
        diagnostics: List[str] = []

        (frame_phones, state_ids, segment_ids, notes, _segments,
         plan_diagnostics) = self.plan(score, default_note=default_note)
        diagnostics.extend(plan_diagnostics)
        if not frame_phones:
            raise ValueError("the score produced no frames")

        means, variances = self.frame_statistics(frame_phones, state_ids)
        trajectory = mlpg(means, variances, spec.stream_sizes,
                          window=spec.delta_window,
                          variance_scale=config.variance_scale,
                          smooth=config.smooth)
        # back out of the normalisation into feature space
        static = self.model.denormalize(trajectory)

        voiced = self.voicing(frame_phones, state_ids)
        f0_semitones = self.pitch(static, notes, voiced, frame_phones,
                                  state_ids, segment_ids, rng)
        f0_semitones, clipped = self._clamp_pitch(f0_semitones)
        if clipped:
            diagnostics.append(
                f"{clipped} frame(s) fell outside the model's F0 range "
                f"({spec.f0_floor:.0f}-{spec.f0_ceil:.0f} Hz) and were clamped")

        parameters = spec.decode(static, f0_semitones=f0_semitones)
        audio = self.vocoder.synthesize(parameters)

        state_sequence = [f"{phone}/{state}"
                          for phone, state in zip(frame_phones, state_ids)]
        return SynthesisResult(
            audio=audio, params=parameters, phones=list(frame_phones),
            notes=notes, state_ids=np.asarray(state_ids, dtype=np.int64),
            state_sequence=state_sequence,
            durations=np.asarray(segment_ids, dtype=np.int64),
            f0_semitones=f0_semitones, diagnostics=diagnostics)


def synthesize(model: HMSModel, score: labels_module.Score,
               config: Optional[SynthesisConfig] = None,
               default_note: float = 60.0, log=None) -> SynthesisResult:
    """Functional entry point used by the CLI."""
    return Synthesizer(model, config, log).synthesize(score,
                                                      default_note=default_note)
