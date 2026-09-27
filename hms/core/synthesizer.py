"""Synthesis pipeline: score -> audio.

    score (phonemes + notes + durations)
        -> per-phoneme state allocation from the requested durations
        -> per-frame model statistics (mean, variance) per state
        -> MLPG acoustic trajectory generation
        -> score target F0 (+ optional learned deviation / vibrato)
        -> voicing mask from the HMM state probabilities
        -> WORLD parameters -> waveform

Each stage is independently replaceable:

* `hms.core.duration.DurationModel.allocate` decides the state sequence;
* `hms.core.generation.mlpg` turns state statistics into a trajectory;
* `hms.core.pitch.PitchModel` provides optional learned deviations and vibrato;
  the score alone supplies target F0 in the default mode;
* `hms.vocoder` turns (f0, sp, ap) into samples.

Options that matter in practice (all in `SynthesisConfig`):

``variance_scale``
    Scales the *dynamic* (delta) variances.  >1 weakens the delta constraints,
    so the trajectory follows the per-frame means more literally (more detail,
    livelier); <1 strengthens them and flattens/smooths the trajectory.  1.0
    reproduces the model.
``pitch_variation``
    Scales the optional learned deviation in ``acoustic`` and ``state_means``
    modes; 0 suppresses that deviation and values >1 exaggerate it. It has no
    effect in the default ``score`` mode.
``f0_source``
    ``score`` (default) uses the requested musical note directly as target F0;
    no learned pitch statistics are needed. ``acoustic`` optionally adds the
    note-relative F0 trajectory learned jointly by the acoustic HMM, while
    ``state_means`` uses the separate per-(phoneme, state) pitch statistics.
    Both learned-deviation modes remain optional extensions to score-driven
    synthesis.
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
    f0_source: str = "score"           # "score" | "acoustic" | "state_means"
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

    def __post_init__(self) -> None:
        if self.f0_source not in ("score", "acoustic", "state_means"):
            raise ValueError("f0_source must be 'score', 'acoustic' or 'state_means'")
        if self.duration_mode not in ("score", "model"):
            raise ValueError("duration_mode must be 'score' or 'model'")
        if self.mixture not in ("dominant", "marginal"):
            raise ValueError("mixture must be 'dominant' or 'marginal'")
        if self.vocoder not in (None, "auto", "native", "pyworld", "builtin"):
            raise ValueError("vocoder must be auto, native, pyworld or builtin")
        numeric = [self.variance_scale, self.pitch_variation, self.tempo,
                   self.transpose]
        numeric.extend(v for v in (self.vibrato_depth, self.vibrato_rate)
                       if v is not None)
        try:
            if not all(np.isfinite(value) for value in numeric):
                raise ValueError("synthesis settings must be finite")
        except TypeError as exc:
            raise ValueError("synthesis settings must be numeric") from exc
        if self.variance_scale <= 0 or self.tempo <= 0:
            raise ValueError("variance_scale and tempo must be positive")
        if self.pitch_variation < 0:
            raise ValueError("pitch_variation must be non-negative")
        if self.vibrato_depth is not None and self.vibrato_depth < 0:
            raise ValueError("vibrato_depth must be non-negative")
        if self.vibrato_rate is not None and self.vibrato_rate <= 0:
            raise ValueError("vibrato_rate must be positive")
        if self.pitch_smoothing < 0:
            raise ValueError("pitch_smoothing must be non-negative")


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
        #: Per-frame resolved HMMs from the last `plan` call, populated only
        #: when the model carries sparse context HMMs; lets
        #: `frame_statistics`/`voicing` use exactly the units `plan` chose
        #: without changing the six-value plan() API.
        self._frame_units = None

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
        #: resolved HMM per frame (context-aware models only); keeps the
        #: public plan() return signature unchanged.
        frame_units = [] if self.model.contexts else None
        self._frame_units = None
        next_segment_id = 0
        duration_rng = np.random.default_rng(config.seed)

        for utterance in score:
            if config.duration_mode == "model" or utterance.end <= utterance.start:
                # no usable timing in the score: let the duration model decide
                phones = [s.phone for s in utterance.segments] or ["sil"]
                predicted = self.model.duration_model.predict(
                    phones, spec.frame_period, tempo=config.tempo,
                    rng=duration_rng, speak=True)
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

            # Context neighbourhood: the sibling segments of this utterance;
            # utterance edges use the inventory's existing silence symbol --
            # there are no BOS/EOS tokens.
            segment_phones = [self.model.phoneme_set.canonical(phone)
                              for phone, _frames, _note in segments]
            silence = self.model.phoneme_set.silence
            for seg_index, (phone, frames, note) in enumerate(segments):
                if self.model.contexts:
                    pre = segment_phones[seg_index - 1] if seg_index > 0 \
                        else silence
                    post = segment_phones[seg_index + 1] \
                        if seg_index < len(segments) - 1 else silence
                    _key, hmm, tier = self.model.resolve_unit(pre, phone, post)
                else:
                    hmm = self.model.get_or_backoff(phone)
                    tier = None
                if self.model.get_hmm(phone) is None \
                        and tier in (None, "class", "global"):
                    diagnostics.append(
                        f"{utterance.name}: phoneme {phone!r} is not in the "
                        f"model; using the backoff model")
                counts = DurationModel.allocate(frames, hmm.duration_proportions())
                for state, count in enumerate(counts):
                    frame_phones.extend([phone] * int(count))
                    state_ids.extend([state] * int(count))
                    segment_ids.extend([next_segment_id] * int(count))
                    notes.extend([default_note if note is None else float(note)]
                                 * int(count))
                    if frame_units is not None:
                        frame_units.extend([hmm] * int(count))
                segment_frames.append(int(sum(counts)))
                next_segment_id += 1

        self._frame_units = frame_units
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
        cache: Dict[Tuple[object, int], Tuple[np.ndarray, np.ndarray]] = {}
        use_dominant = self.config.mixture == "dominant"
        units = self._active_frame_units(n_frames)

        for t in range(n_frames):
            phone = frame_phones[t]
            state = int(state_ids[t])
            if units is not None:
                hmm = units[t]
                key = (id(hmm), state)
            else:
                hmm = self.model.get_or_backoff(phone)
                key = (phone, state)
            if key not in cache:
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

    def _active_frame_units(self, n_frames: int) -> Optional[List]:
        """The per-frame HMMs chosen by `plan`, when context models apply.

        ``None`` means: resolve per phoneme exactly as before (`get_or_backoff`).
        """
        units = self._frame_units
        if units is not None and len(units) == n_frames:
            return units
        return None

    def voicing(self, frame_phones: Sequence[str], state_ids: Sequence[int]
                ) -> np.ndarray:
        """Per-frame voiced mask from the learned state/phone statistics."""
        voiced = np.zeros(len(frame_phones), dtype=bool)
        units = self._active_frame_units(len(frame_phones))
        for t, (phone, state) in enumerate(zip(frame_phones, state_ids)):
            hmm = units[t] if units is not None \
                else self.model.get_or_backoff(phone)
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
             + optional learned deviation (acoustic or state_means source)
             + optional vibrato (explicit component, if enabled)

        The default ``score`` source uses the score note with zero deviation.
        Learned pitch behavior is opt-in; absolute training-speaker F0 is never
        copied.
        """
        spec = self.model.spec
        config = self.config
        note_semitones = 12.0 * np.log2(
            labels_module.midi_to_hz(note_per_frame) / spec.f0_ref_hz)

        if config.f0_source == "score":
            # Base score-driven synthesis: the supplied note is the complete
            # target F0; no learned pitch statistics or HMM F0 trajectory are
            # consulted. Optional vibrato can still be added below.
            relative = np.zeros(len(static), dtype=np.float64)
        elif config.f0_source == "state_means":
            # Optional pitch-model mode: use per-(phoneme, state) statistics,
            # then smooth them across state boundaries.
            series = np.zeros(len(static), dtype=np.float64)
            for index, phone in enumerate(phones):
                means = self.model.pitch_model.state_means(phone)
                if len(means):
                    state = min(int(state_ids[index]), len(means) - 1)
                    series[index] = means[state]
            relative = np.convolve(series, np.ones(5) / 5.0, mode="same") \
                if len(series) >= 5 else series
            # Re-centre on voiced frames only: silence statistics are not sung.
            if voiced.any():
                relative = relative - relative[voiced].mean()
            elif len(relative):
                relative = relative - relative.mean()
            relative *= float(config.pitch_variation)
        else:  # acoustic: optional note-relative trajectory from the HMM
            relative = static[:, 0] * float(config.pitch_variation)

        # Make synthesis-time vibrato overrides local to this render instead of
        # mutating the saved voice (depth/rate used to leak into later renders).
        from hms.core.pitch import PitchModel, Vibrato
        vibrato = Vibrato.from_dict(self.model.pitch_model.vibrato.to_dict())
        if config.vibrato is not None:
            vibrato.enabled = bool(config.vibrato)
        if config.vibrato_depth is not None:
            vibrato.depth_semitones = float(config.vibrato_depth)
        if config.vibrato_rate is not None:
            vibrato.rate_hz = float(config.vibrato_rate)
        pitch_model = PitchModel(vibrato=vibrato)

        f0 = pitch_model.generate(
            note_semitones, voiced, rng=rng, note_ids=np.asarray(segment_ids),
            frame_period=spec.frame_period, relative_trajectory=relative)

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
