"""Synthesis pipeline: score -> audio.

    score (phonemes + notes + durations)
        -> per-phoneme state allocation from the requested durations
        -> per-frame model statistics (mean, variance) per state
        -> MLPG acoustic trajectory generation
        -> score target F0 (+ optional learned deviation / vibrato)
        -> voicing mask from the HMM state probabilities
        -> WORLD parameters -> waveform

F0 and the spectral envelope are separate all the way through. The model
supplies the envelope (and aperiodicity) for a phoneme state; the score or the
caller supplies the pitch; the vocoder is handed both. An F0 outside the range
the model was trained on therefore costs you the *statistics* of that pitch, not
the pitch itself: the boundary acoustic region is reused and the requested F0
reaches the vocoder unchanged. See `_out_of_range_pitch_diagnostics`.

Each stage is independently replaceable:

* `hms.core.duration.DurationModel.allocate` decides the state sequence;
* `hms.core.generation.mlpg` turns state statistics into a trajectory;
* `hms.core.pitch.PitchModel` provides optional learned deviations and vibrato;
  the score alone supplies target F0 in the default mode;
* `hms.vocoder` turns (f0, sp, ap) into samples.

F0 can also be supplied from outside.  Passing ``f0=`` to `synthesize` replaces
the generated contour outright -- see `external_f0_to_semitones`:

    score / learned pitch / vibrato -> generated F0 --.
                                                      +--> F0 -> vocoder
    caller-supplied f0 trajectory --------------------'

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

``f0=`` (argument, not a config field)
    Optional external F0 trajectory, one value per *synthesis frame* in Hz,
    with ``0.0`` marking unvoiced frames (WORLD's convention).  When given it
    is authoritative: the score note, the learned deviation, the generated
    vibrato and ``pitch_smoothing`` are all skipped for that render.  See
    `external_f0_to_semitones`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms.core import labels as labels_module
from hms.core.duration import DurationModel
from hms.core.features import (AcousticFrameSequence, FeatureSpec,
                              hz_to_semitone, semitone_to_hz)
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


# --------------------------------------------------------------------------
# External F0 override
# --------------------------------------------------------------------------


def external_f0_to_semitones(f0, n_frames: int, spec: FeatureSpec
                             ) -> np.ndarray:
    """Validate a caller-supplied F0 trajectory and convert it for synthesis.

    Parameters
    ----------
    f0 : array-like, shape (n_frames,)
        Absolute F0 in **Hz**, one value per *synthesis frame* -- the frames
        `Synthesizer.plan` produces for the whole score, not per-phoneme
        arrays.  Unvoiced frames use the project's existing WORLD convention:
        ``0.0`` Hz (anything below ``spec.voiced_threshold`` counts as
        unvoiced, exactly as in `FeatureSpec.encode`).
    n_frames : int
        Number of frames the render will produce.
    spec : FeatureSpec
        Supplies the reference frequency and the voicing threshold.

    Returns
    -------
    numpy.ndarray
        The trajectory in the synthesizer's internal representation: absolute
        F0 in semitones re. ``spec.f0_ref_hz``, with ``NaN`` on unvoiced
        frames -- the same array `Synthesizer.pitch` returns.

    Notes
    -----
    The trajectory is *authoritative*: it is converted, never augmented.
    Nothing is added on top of it -- no score note, no learned deviation, no
    generated vibrato -- and it is never resampled, interpolated, truncated or
    padded: a trajectory that does not line up with the frame sequence is an
    error, not an approximation. Its values are used exactly as supplied,
    including frequencies outside the model's trained F0 range, which are
    reported through `SynthesisResult.diagnostics` rather than altered.

    Raises
    ------
    ValueError
        For every malformed input the vocoder could only fail on much later:
        wrong length, wrong shape, an empty trajectory, non-numeric values,
        NaN/inf, or negative frequencies.  The message names the offending
        frame index and says how unvoiced frames are written.
    """
    if n_frames <= 0:
        raise ValueError("external F0: the score produced no frames to align "
                         "the trajectory to")
    try:
        raw = np.asarray(f0)
    except (TypeError, ValueError) as exc:
        raise ValueError("external F0 must be a 1-D array of numbers, got "
                         f"{type(f0).__name__}") from exc
    if raw.dtype.kind not in "fiu":
        raise ValueError("external F0 must be a 1-D array of real numbers, "
                         f"got dtype '{raw.dtype}'")
    values = raw.astype(np.float64)
    if values.ndim != 1:
        raise ValueError("external F0 must have exactly one value per "
                         f"frame (shape ({n_frames},)), got shape "
                         f"{values.shape}; pass a flat array "
                         f"(e.g. f0.reshape(-1))")
    if values.size == 0:
        raise ValueError("external F0 trajectory is empty; this render "
                         f"needs {n_frames} frame(s), one value per "
                         f"frame (0.0 Hz marks an unvoiced frame)")
    if values.size != n_frames:
        raise ValueError(
            f"external F0 has {values.size} frame(s) but the render has "
            f"{n_frames}; supply exactly one value per synthesis frame -- "
            f"HMS never resizes, interpolates or pads an explicit F0 "
            f"trajectory")
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        index = int(np.argmax(non_finite))
        kind = "NaN" if np.isnan(values[index]) else "infinite"
        raise ValueError(
            f"external F0 must be finite: frame {index} is {kind} and "
            f"{int(non_finite.sum())} of {values.size} frame(s) are "
            f"non-finite; unvoiced frames are 0.0 Hz, not NaN")
    negative = values < 0.0
    if negative.any():
        index = int(np.argmax(negative))
        raise ValueError(
            f"external F0 must be non-negative: frame {index} is "
            f"{values[index]:.6g} Hz and {int(negative.sum())} of "
            f"{values.size} frame(s) are negative; unvoiced frames "
            f"are 0.0 Hz")

    voiced = values >= float(spec.voiced_threshold)
    return np.where(voiced,
                    hz_to_semitone(np.maximum(values, 1e-12), spec.f0_ref_hz),
                    np.nan)


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

    def plan(self, score: labels_module.Score, default_note: float = 60.0,
             *, check_pitch_range: bool = True
             ) -> Tuple[List[str], List[int], List[int], np.ndarray,
                        List[int], List[str]]:
        """Turn a score into (phone, segment, frame) aligned state indices.

        Returns ``(frame_phones, state_ids, segment_ids, note_per_frame,
        segment_frames, diagnostics)`` -- the diagnostics list is returned so
        the caller can show it, instead of being written into hidden state.
        Notes are MIDI numbers (float, so fractional detuning is allowed).

        ``check_pitch_range`` (default) adds one diagnostic per utterance
        when an *effective* note -- score note + transpose, including the
        default note for unnoted segments -- is not a valid MIDI number or
        falls outside the model's trained F0 range, and normalises the former
        to the nearest valid MIDI note (see `_valid_midi_notes`). Parsed
        scores are already within the MIDI range, so this is what catches
        ``--transpose`` and programmatic notes; it is skipped when an external
        F0 trajectory replaces the score pitch entirely, because then the notes
        are not rendered at all.
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

            if check_pitch_range:
                diagnostics.extend(
                    self._note_range_diagnostics(
                        utterance.name, segments, default_note,
                        float(config.transpose), spec))

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
        if check_pitch_range:
            # Only meaningful when the score drives the pitch: an external F0
            # trajectory replaces it, so the notes are not rendered at all and
            # saying anything about them would be noise.
            note_per_frame = self._valid_midi_notes(note_per_frame,
                                                    diagnostics)
        return (frame_phones, state_ids, segment_ids, note_per_frame,
                segment_frames, diagnostics)

    @staticmethod
    def _valid_midi_notes(notes: np.ndarray, diagnostics: List[str]
                          ) -> np.ndarray:
        """Normalise the notes that are not MIDI notes at all.

        MIDI 0-127 is the *format's* range, not the model's: a note outside it
        names no musical pitch at all and can only arrive through `transpose`,
        a `default_note` override or a programmatic score (parsed label files
        are already checked). Such a note is rendered at the nearest valid one
        and reported, which keeps every rendered frequency below Nyquist.

        Notes *inside* the MIDI range are never touched here, however far their
        frequency is from the model's trained range -- see
        `_out_of_range_pitch_diagnostics`.
        """
        notes = np.asarray(notes, dtype=np.float64)
        finite = np.isfinite(notes)
        invalid = finite & ((notes < labels_module.MIDI_NOTE_MIN)
                            | (notes > labels_module.MIDI_NOTE_MAX))
        if not invalid.any():
            return notes
        offending = np.unique(notes[invalid])

        def describe(value: float) -> str:
            nearest = float(np.clip(value, labels_module.MIDI_NOTE_MIN,
                                    labels_module.MIDI_NOTE_MAX))
            return (f"{value:g} -> {nearest:g} "
                    f"(~{labels_module.midi_to_hz(nearest):.0f} Hz)")

        listing = ", ".join(describe(value) for value in offending[:5])
        if len(offending) > 5:
            listing += f", ... ({int(invalid.sum())} frame(s))"
        diagnostics.append(
            f"note(s) {listing} are outside the valid MIDI range "
            f"[{labels_module.MIDI_NOTE_MIN:g}, {labels_module.MIDI_NOTE_MAX:g}]; "
            f"they are rendered at the nearest valid MIDI note")
        return np.where(invalid, np.clip(notes, labels_module.MIDI_NOTE_MIN,
                                         labels_module.MIDI_NOTE_MAX), notes)

    @staticmethod
    def _note_range_diagnostics(utterance_name: str,
                                segments: List[Tuple[str, int,
                                                     Optional[float]]],
                                default_note: float, transposition: float,
                                spec: FeatureSpec) -> List[str]:
        """Name the notes the model has no observation for.

        Two distinct problems, two distinct messages:

        * an effective note outside the valid MIDI range 0-127 (only possible
          via transpose / default note / programmatic scores, since parsed
          files are already checked) is not a musical note at all; it is
          normalised to the nearest valid one by `_valid_midi_notes`;
        * a *valid* MIDI note whose frequency lies outside the model's trained
          F0 range (``f0_floor``-``f0_ceil``) is perfectly legitimate -- the
          model simply has no observation there -- so it is warned about and
          then rendered at exactly the requested pitch.
        """
        low_hz, high_hz = spec.f0_floor, spec.f0_ceil
        low_midi = float(labels_module.hz_to_midi(low_hz))
        high_midi = float(labels_module.hz_to_midi(high_hz))
        invalid_midi: set = set()
        beyond_range: set = set()
        for _phone, _frames, note in segments:
            effective = round(
                (default_note if note is None else float(note))
                + transposition, 4)
            if effective < labels_module.MIDI_NOTE_MIN \
                    or effective > labels_module.MIDI_NOTE_MAX:
                invalid_midi.add(effective)
            elif effective < low_midi or effective > high_midi:
                beyond_range.add(effective)
        diagnostics: List[str] = []
        if invalid_midi:
            listing = ", ".join(f"{value:g}"
                                for value in sorted(invalid_midi)[:5])
            if len(invalid_midi) > 5:
                listing += f", ... ({len(invalid_midi)} notes)"
            diagnostics.append(
                f"{utterance_name}: note(s) {listing} (score note + "
                f"transpose {transposition:+g}, incl. the default note "
                f"{default_note:g} for unnoted segments) are outside the "
                f"valid MIDI range [{labels_module.MIDI_NOTE_MIN:g}, "
                f"{labels_module.MIDI_NOTE_MAX:g}]; they are not musical "
                f"notes and are rendered at the nearest valid MIDI note")
        if beyond_range:
            listing = ", ".join(
                f"{value:g} (~{labels_module.midi_to_hz(value):.0f} Hz)"
                for value in sorted(beyond_range)[:5])
            if len(beyond_range) > 5:
                listing += f", ... ({len(beyond_range)} notes)"
            diagnostics.append(
                f"{utterance_name}: note(s) {listing} fall outside the "
                f"model's trained F0 range ({low_hz:g}-{high_hz:g} Hz, "
                f"MIDI {low_midi:.0f}-{high_midi:.0f}); the model has no "
                f"observation at that pitch, so they are rendered at the "
                f"requested F0 using the boundary acoustic statistics")
        return diagnostics

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

    def _out_of_range_pitch_diagnostics(self, f0_semitones: np.ndarray
                                         ) -> List[str]:
        """Name the frames whose F0 the model has no observation for.

        The model is not *queried by F0*. Its per-(phoneme, state) statistics
        are a mel-cepstrum spectral envelope and a band aperiodicity: a
        description of the vocal tract, not of a particular pitch, learned from
        whatever frames the singer happened to sing that phone on. So for an
        F0 outside the trained range there is no second, better envelope to
        interpolate towards -- the nearest acoustic region HMS has *is* the
        boundary one it just used, and reusing it is the Sinsy rule ("use the
        closest observed F0's acoustic parameters") reduced to what this
        architecture can actually express.

        What is *not* reused is the F0 itself. The requested pitch is passed
        through to `FeatureSpec.decode` and from there to the vocoder verbatim,
        so a note above `f0_ceil` is sung at its own frequency instead of
        sliding down to the ceiling. This reports the situation and changes
        nothing.
        """
        spec = self.model.spec
        low = 12.0 * np.log2(max(spec.f0_floor, 1e-3) / spec.f0_ref_hz)
        high = 12.0 * np.log2(spec.f0_ceil / spec.f0_ref_hz)
        voiced = np.isfinite(f0_semitones)
        if not voiced.any():
            return []
        below = voiced & (f0_semitones < low)
        above = voiced & (f0_semitones > high)
        if not (below.any() or above.any()):
            return []

        def hz(values: np.ndarray) -> np.ndarray:
            return semitone_to_hz(values, spec.f0_ref_hz)

        reach: List[str] = []
        if below.any():
            reach.append(f"down to {float(hz(f0_semitones[below]).min()):.1f} Hz")
        if above.any():
            reach.append(f"up to {float(hz(f0_semitones[above]).max()):.1f} Hz")
        return [
            f"{int(below.sum() + above.sum())} of {int(voiced.sum())} voiced "
            f"frame(s) request F0 outside the model's trained F0 range "
            f"{spec.f0_floor:.0f}-{spec.f0_ceil:.0f} Hz ({', '.join(reach)}); "
            f"the boundary acoustic statistics for those frames are reused "
            f"unchanged and the requested F0 is preserved"
        ]

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
                   default_note: float = 60.0,
                   f0=None) -> SynthesisResult:
        """Render `score`.

        ``f0`` is an optional external F0 trajectory (Hz, one value per
        synthesis frame, 0.0 for unvoiced).  When it is given it replaces the
        generated contour -- score F0, learned deviation and vibrato are all
        skipped -- so the caller is the only source of pitch for that render.
        See `external_f0_to_semitones` for the units, the voicing convention
        and the validation.  With ``f0=None`` (the default) nothing changes.
        """
        config = self.config
        spec = self.model.spec
        rng = np.random.default_rng(config.seed)
        diagnostics: List[str] = []

        (frame_phones, state_ids, segment_ids, notes, _segments,
         plan_diagnostics) = self.plan(score, default_note=default_note,
                                       check_pitch_range=f0 is None)
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

        if f0 is None:
            voiced = self.voicing(frame_phones, state_ids)
            f0_semitones = self.pitch(static, notes, voiced, frame_phones,
                                      state_ids, segment_ids, rng)
        else:
            # External F0 override: the caller's trajectory is authoritative,
            # so the generated one (score note, learned deviation, vibrato) is
            # not even computed. Everything downstream -- decoding to WORLD
            # parameters, the vocoder -- is untouched, and the supplied F0 is
            # used exactly as given, trained range or not.
            f0_semitones = external_f0_to_semitones(
                f0, len(frame_phones), spec)
            diagnostics.append(
                f"external F0 override: {len(f0_semitones)} frame(s) "
                f"supplied by the caller "
                f"({int(np.isfinite(f0_semitones).sum())} voiced)")
        diagnostics.extend(self._out_of_range_pitch_diagnostics(f0_semitones))

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
               default_note: float = 60.0, f0=None,
               log=None) -> SynthesisResult:
    """Functional entry point used by the CLI.

    ``f0`` is the optional external F0 trajectory (Hz, one value per synthesis
    frame, 0.0 for unvoiced) that replaces the generated contour; see
    `external_f0_to_semitones`.
    """
    return Synthesizer(model, config, log).synthesize(
        score, default_note=default_note, f0=f0)
