"""Training pipeline: corpus -> model.

    labels + wavs
        -> per-utterance WORLD analysis          (vocoder backend)
        -> per-frame phoneme / note alignment     (labels.py)
        -> note-relative static features          (features.py + pitch.py)
        -> dynamic features + normalisation       (features.py)
        -> per-phoneme HMM training with the
           state split learned from the data      (hmm.py)
        -> duration / pitch / voicing statistics  (duration.py, pitch.py)
        -> model directory                        (model.py)

Nothing here is a neural network and nothing is a black box: the whole acoustic
model is a few thousand numbers you can read out of ``model.yaml`` / ``hmm.npz``
and reason about.

Data efficiency
---------------
The defaults are chosen so that a handful of sung phrases trains a usable voice:

* phoneme *classes* set the state/component budget (`phonemes.yaml`), so the
  short, rare phones stay simple and only the vowels spend parameters;
* every state's GMM is fitted with a variance floor relative to the data's own
  variance, so tiny datasets cannot produce degenerate (infinitely peaked)
  Gaussians;
* component weights below ``MIN_WEIGHT`` are pruned after EM;
* ``corpus_splits`` (below) optionally pools several takes of the same phoneme
  *within* different notes, which is what makes a small pitch range generalise;
* by default all phonemes share one covariance style and one mixture size per
  class -- no per-phoneme tuning.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms import __version__
from hms.core import labels as labels_module
from hms.core.duration import DurationModel
from hms.core.features import (AcousticFrameSequence, FeatureSpec,
                               add_dynamic_features)
from hms.core.hmm import LeftToRightHMM
from hms.core.model import HMSModel, ModelStats
from hms.core.phonemes import PhonemeSet
from hms.core.pitch import PitchModel, note_relative_pitch
from hms.data import wavio
from hms.vocoder import get_vocoder


@dataclass
class TrainingConfig:
    """Every knob the trainer has.  Mirrors `config/parameters.yaml`."""

    # -- data -------------------------------------------------------------
    wav_dir: Optional[str] = None
    label_file: Optional[str] = None
    time_unit: str = "seconds"          # "seconds" | "frames"
    default_note: float = 60.0          # MIDI note for segments without one
    #: glob-free list of audio extensions tried when matching utterance ids
    audio_extensions: Tuple[str, ...] = (".wav",)

    # -- acoustic ---------------------------------------------------------
    fs: int = 44100
    frame_period: float = 5.0
    fft_size: Optional[int] = None      # None -> whatever the vocoder uses
    n_mcep: int = 30
    n_band: int = 5
    use_delta: bool = True
    use_delta2: bool = False
    f0_ceiling: float = 800.0
    f0_floor: float = 71.0
    f0_estimation: str = "dio"
    refine_f0: bool = True
    vocoder: str = "auto"

    # -- model ------------------------------------------------------------
    covariance_type: str = "diag"       # "diag" | "tied"
    var_floor_ratio: float = 1e-3
    training_method: str = "viterbi"    # "viterbi" | "baum_welch"
    n_iterations: int = 5
    normalize_scale: bool = True
    seed: int = 0
    allow_skip: bool = False
    #: minimum frames of a phoneme before it is modelled at all
    min_phoneme_frames: int = 20

    # -- duration / pitch -------------------------------------------------
    vibrato_enabled: bool = False
    vibrato_estimate_from_data: bool = True
    pitch_variation: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingConfig":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown training options: {sorted(unknown)}")
        if "audio_extensions" in data:
            data["audio_extensions"] = tuple(data["audio_extensions"])
        return cls(**data)


class Corpus:
    """A label file plus the audio it references."""

    def __init__(self, score: labels_module.Score, wav_dir: Path,
                 extensions: Sequence[str] = (".wav",)) -> None:
        self.score = score
        self.wav_dir = Path(wav_dir)
        self.extensions = tuple(extensions)

    def audio_path(self, name: str) -> Optional[Path]:
        for extension in self.extensions:
            candidate = self.wav_dir / f"{name}{extension}"
            if candidate.exists():
                return candidate
        return None

    def missing_audio(self) -> List[str]:
        return [u.name for u in self.score
                if self.audio_path(u.name) is None]

    @property
    def total_seconds(self) -> float:
        return float(sum(u.end - u.start for u in self.score))


@dataclass
class UtteranceData:
    """One analysed utterance, ready to be cut into phoneme segments."""

    name: str
    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray
    phones: List[str]                     # per frame
    notes: np.ndarray                     # per frame, MIDI numbers
    voiced: np.ndarray                    # per frame, bool
    note_semitones: np.ndarray            # per frame, semitones re. f0_ref
    relative_pitch: np.ndarray            # per frame, semitones re. the note
    features: np.ndarray                  # (T, dim) with dynamic streams
    phoneme_spans: List[Tuple[str, int, int]]
    diagnostics: List[str] = field(default_factory=list)


class Trainer:
    """Runs the training pipeline and returns an :class:`HMSModel`."""

    def __init__(self, config: TrainingConfig,
                 phoneme_set: Optional[PhonemeSet] = None,
                 log=None) -> None:
        self.config = config
        self.phoneme_set = phoneme_set or PhonemeSet.default()
        self.log = log or (lambda message: None)
        self.spec: Optional[FeatureSpec] = None
        self._vocoder = None

    # -- setup -------------------------------------------------------------

    @property
    def vocoder(self):
        if self._vocoder is None:
            self._vocoder = get_vocoder(self.config.vocoder,
                                        fft_size=self.config.fft_size)
        return self._vocoder

    def build_spec(self) -> FeatureSpec:
        # the FFT size is a property of the sample rate for WORLD, so ask the
        # backend rather than assuming its 44.1 kHz default applies here
        fft_size = (self.config.fft_size
                    or self.vocoder.fft_size_for(self.config.fs))
        return FeatureSpec(
            fs=self.config.fs, frame_period=self.config.frame_period,
            fft_size=fft_size, n_mcep=self.config.n_mcep,
            n_band=self.config.n_band, use_delta=self.config.use_delta,
            use_delta2=self.config.use_delta2, f0_floor=self.config.f0_floor,
            f0_ceil=self.config.f0_ceiling,
            f0_estimation=self.config.f0_estimation,
            refine_f0=self.config.refine_f0)

    # -- stage 1: analyse --------------------------------------------------

    def analyse_corpus(self, corpus: Corpus) -> List[UtteranceData]:
        """WORLD analysis + label alignment for every utterance."""
        self.spec = self.spec or self.build_spec()
        spec = self.spec
        hop_ms = spec.frame_period / 1000.0
        out: List[UtteranceData] = []

        for utterance in corpus.score:
            path = corpus.audio_path(utterance.name)
            if path is None:
                self.log(f"  ! {utterance.name}: no audio found, skipped")
                continue
            signal, fs = wavio.read_wav(path)
            if fs != spec.fs:
                raise ValueError(
                    f"{path.name} is {fs} Hz but the configuration expects "
                    f"{spec.fs} Hz; resample it or change `fs` "
                    f"(HMS does not resample implicitly)")
            sequence = self.vocoder.analyze_to_sequence(
                signal, fs, frame_period=spec.frame_period,
                f0_floor=spec.f0_floor, f0_ceil=spec.f0_ceil,
                f0_estimation=spec.f0_estimation, refine_f0=spec.refine_f0)
            n_frames = len(sequence)
            phones, notes, _ = utterance.frame_labels(
                spec.frame_period, self.config.default_note, n_frames=n_frames)
            voiced = sequence.f0 > spec.voiced_threshold

            note_semitones = 12.0 * np.log2(
                labels_module.midi_to_hz(notes) / spec.f0_ref_hz)
            relative = note_relative_pitch(sequence.f0,
                                           labels_module.midi_to_hz(notes),
                                           voiced, spec.f0_ref_hz)
            static = spec.encode(sequence.f0, sequence.sp, sequence.ap)
            # dimension 0 carries the note-relative pitch instead of the
            # absolute one: this is the "note conditions F0" mechanism
            static[:, 0] = relative
            features = add_dynamic_features(static, spec.use_delta,
                                            spec.use_delta2,
                                            window=spec.delta_window)

            spans = [(phone, lo, hi) for phone, lo, hi, _note
                     in labels_module.segment_boundaries(utterance,
                                                         spec.frame_period,
                                                         n_frames=n_frames)]
            out.append(UtteranceData(
                name=utterance.name, f0=sequence.f0, sp=sequence.sp,
                ap=sequence.ap, phones=phones, notes=notes, voiced=voiced,
                note_semitones=note_semitones, relative_pitch=relative,
                features=features, phoneme_spans=spans,
                diagnostics=[d for d in corpus.score.diagnostics
                              if d.startswith(utterance.name)]))
            self.log(f"  analysed {utterance.name}: {n_frames} frames "
                     f"({n_frames * hop_ms:.2f} s)")
        if not out:
            raise RuntimeError("no training utterances were analysed")
        return out

    # -- stage 2: normalise -------------------------------------------------

    def compute_normalization(self, utterances: Sequence[UtteranceData]
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Feature offset (mean) and scale (1/std) over the whole corpus."""
        static_dim = self.spec.static_dim
        total = np.zeros(static_dim, dtype=np.float64)
        total_sq = np.zeros(static_dim, dtype=np.float64)
        count = 0
        for data in utterances:
            static = data.features[:, :static_dim]
            total += static.sum(axis=0)
            total_sq += (static ** 2).sum(axis=0)
            count += len(static)
        if count == 0:
            return np.zeros(static_dim), np.ones(static_dim)
        mean = total / count
        variance = np.maximum(total_sq / count - mean ** 2, 1e-6)
        scale = 1.0 / np.sqrt(variance) if self.config.normalize_scale \
            else np.ones(static_dim)
        scale = np.minimum(scale, 100.0)          # guard against flat dimensions
        return mean, scale

    def normalize_features(self, features: np.ndarray, offset: np.ndarray,
                           scale: np.ndarray) -> np.ndarray:
        """Apply (x - offset) * scale to the static block, keep deltas linear.

        Because the dynamic streams are linear in the static stream, scaling the
        static block and re-deriving the deltas is exactly equivalent to scaling
        the stacked features -- but doing it in this order keeps
        `hms.core.generation` (which assumes the same window) consistent.
        """
        static_dim = self.spec.static_dim
        static = features[:, :static_dim]
        normalized = (static - offset) * scale
        return add_dynamic_features(normalized, self.spec.use_delta,
                                    self.spec.use_delta2,
                                    window=self.spec.delta_window)

    # -- stage 3: HMMs -----------------------------------------------------

    def collect_phoneme_data(self, utterances: Sequence[UtteranceData],
                             offset: np.ndarray, scale: np.ndarray
                             ) -> Tuple[Dict[str, List[np.ndarray]],
                                        Dict[str, List[np.ndarray]],
                                        Dict[str, List[np.ndarray]],
                                        Dict[str, List[float]]]:
        """Group frames by phoneme: features, voicing, relative pitch, durations."""
        features: Dict[str, List[np.ndarray]] = {}
        voiced: Dict[str, List[np.ndarray]] = {}
        pitch: Dict[str, List[np.ndarray]] = {}
        durations: Dict[str, List[float]] = {}

        for data in utterances:
            normalized = self.normalize_features(data.features, offset, scale)
            for phone, lo, hi in data.phoneme_spans:
                if hi <= lo:
                    continue
                canonical = self.phoneme_set.canonical(phone)
                features.setdefault(canonical, []).append(normalized[lo:hi])
                voiced.setdefault(canonical, []).append(data.voiced[lo:hi])
                pitch.setdefault(canonical, []).append(data.relative_pitch[lo:hi])
                durations.setdefault(canonical, []).append(float(hi - lo))
        return features, voiced, pitch, durations

    def train_hmms(self, features: Dict[str, List[np.ndarray]],
                   voiced: Dict[str, List[np.ndarray]]
                   ) -> Dict[str, LeftToRightHMM]:
        config = self.config
        hmms: Dict[str, LeftToRightHMM] = {}
        for phone in sorted(features):
            sequences = features[phone]
            n_frames = int(sum(len(s) for s in sequences))
            definition = self.phoneme_set.resolve(phone)
            if definition is None:
                self.log(f"  ! phoneme {phone!r} is not in phonemes.yaml; "
                         f"skipped (backoff will cover it)")
                continue
            if n_frames < config.min_phoneme_frames:
                self.log(f"  ! phoneme {phone!r} has only {n_frames} frames "
                         f"(< {config.min_phoneme_frames}); skipped")
                continue
            n_states = min(definition.n_states, max(1, n_frames // 2))
            hmm = LeftToRightHMM(n_states=n_states, allow_skip=config.allow_skip,
                                 covariance_type=config.covariance_type)
            hmm.train(sequences, n_components=definition.n_components,
                      covariance_type=config.covariance_type,
                      n_iterations=config.n_iterations,
                      var_floor_ratio=config.var_floor_ratio,
                      seed=config.seed, method=config.training_method,
                      voiced=voiced.get(phone))
            hmms[phone] = hmm
            self.log(f"  {phone:8s} {n_states} states x "
                     f"{definition.n_components} comp, {n_frames} frames, "
                     f"{hmm.n_free_params} params")
        if not hmms:
            raise RuntimeError("no phoneme had enough data to train an HMM")
        return hmms

    @staticmethod
    def _resample_rows(matrix: np.ndarray, target: int) -> np.ndarray:
        """Stretch or shrink the first axis of an array by index interpolation."""
        matrix = np.asarray(matrix)
        n = matrix.shape[0] if matrix.ndim else 1
        if n == target:
            return matrix.copy()
        if n == 1:
            return np.repeat(matrix, target, axis=0)
        index = np.round(np.linspace(0, n - 1, target)).astype(int)
        return matrix[index]

    def build_backoff(self, hmms: Dict[str, LeftToRightHMM]
                      ) -> Dict[str, LeftToRightHMM]:
        """One pooled model per phoneme class, for symbols with no model.

        Pooling is deliberately crude -- states are aligned by relative
        position, components by index, then averaged -- because this model only
        has to produce a *plausible* sound for a phoneme that was never trained
        (the alternative is a crash or silence).  It costs no extra training.
        """
        from hms.core.gmm import DiagGMM
        from hms.core.hmm import HMMState, StateDurationStats

        pooled: Dict[str, List[LeftToRightHMM]] = {}
        for phone, hmm in hmms.items():
            definition = self.phoneme_set.resolve(phone)
            klass = definition.type if definition else "unvoiced_consonant"
            pooled.setdefault(klass, []).append(hmm)

        backoff: Dict[str, LeftToRightHMM] = {}
        for klass, members in sorted(pooled.items()):
            reference = members[0]
            n_states = max(1, int(round(np.mean([m.n_states for m in members]))))
            n_components = max(1, int(round(np.mean(
                [m.states[0].gmm.n_components for m in members]))))
            template = LeftToRightHMM(
                n_states=n_states, covariance_type=reference.covariance_type)
            arrays = [m.to_arrays() for m in members]

            for state in range(n_states):
                weights, means, variances = [], [], []
                self_loops, duration_mean, duration_var, voiced = [], [], [], []
                for member, array in zip(members, arrays):
                    state_index = min(
                        int(round(state / max(n_states - 1, 1)
                                  * (member.n_states - 1))), member.n_states - 1)
                    component_index = np.round(np.linspace(
                        0, array["weights"].shape[1] - 1, n_components)).astype(int)
                    weights.append(array["weights"][state_index][component_index])
                    means.append(array["means"][state_index][component_index])
                    variances.append(array["variances"][state_index][component_index])
                    self_loops.append(array["self_loops"][state_index])
                    duration_mean.append(array["duration_mean"][state_index])
                    duration_var.append(array["duration_variance"][state_index])
                    voiced.append(array["voiced_prob"][state_index])

                weight = np.mean(weights, axis=0)
                weight = np.maximum(weight, 1e-4)
                weight /= weight.sum()
                template.states[state] = HMMState(
                    gmm=DiagGMM(weight, np.mean(means, axis=0),
                                np.mean(variances, axis=0),
                                reference.covariance_type),
                    duration=StateDurationStats(float(np.mean(duration_mean)),
                                                float(np.mean(duration_var)), 0),
                    voiced_prob=float(np.mean(voiced)))
                template.self_loops[state] = float(np.mean(self_loops))
            template.dim = reference.dim
            backoff[klass] = template
        return backoff

    # -- stage 4: duration and pitch ---------------------------------------

    def build_duration_model(self, durations: Dict[str, List[float]]
                             ) -> DurationModel:
        model = DurationModel()
        model.fit({phone: values for phone, values in durations.items()})
        return model

    def build_pitch_model(self, utterances: Sequence[UtteranceData],
                          offset: np.ndarray, scale: np.ndarray,
                          hmms: Dict[str, LeftToRightHMM]) -> PitchModel:
        """Per-state relative-pitch statistics + voicing priors + vibrato."""
        model = PitchModel(pitch_variation=self.config.pitch_variation)
        per_state_pitch: Dict[str, List[List[np.ndarray]]] = {}
        per_state_voiced: Dict[str, List[List[np.ndarray]]] = {}

        # Segment every occurrence with the *trained* HMM on the acoustic
        # features (not on pitch!), so these statistics line up with the states
        # that synthesis will actually select.
        for data in utterances:
            normalized = self.normalize_features(data.features, offset, scale)
            for phone, lo, hi in data.phoneme_spans:
                canonical = self.phoneme_set.canonical(phone)
                hmm = hmms.get(canonical)
                if hmm is None or hi <= lo:
                    continue
                state_path = hmm.segment(normalized[lo:hi])
                pitch_lists = per_state_pitch.setdefault(
                    canonical, [[] for _ in range(hmm.n_states)])
                voiced_lists = per_state_voiced.setdefault(
                    canonical, [[] for _ in range(hmm.n_states)])
                for state in range(hmm.n_states):
                    where = np.where(state_path == state)[0]
                    if len(where) == 0:
                        continue
                    pitch_lists[state].append(data.relative_pitch[lo:hi][where])
                    voiced_lists[state].append(data.voiced[lo:hi][where])

        for phone, pitch_lists in per_state_pitch.items():
            model.add_state_statistics(phone, pitch_lists,
                                       per_state_voiced[phone])

        if self.config.vibrato_enabled:
            model.vibrato.enabled = True
            if self.config.vibrato_estimate_from_data:
                estimate = self._estimate_vibrato(utterances)
                if estimate:
                    self.log(f"  vibrato from data: {estimate['rate_hz']:.2f} Hz, "
                             f"{estimate['depth_semitones']:.2f} semitones")
        return model

    def _estimate_vibrato(self, utterances: Sequence[UtteranceData]
                          ) -> Optional[dict]:
        from hms.core.pitch import estimate_vibrato

        candidates = []
        for data in utterances:
            long_notes = set()
            for _phone, lo, hi in data.phoneme_spans:
                if hi - lo >= 80:              # >= 400 ms sustained
                    long_notes.add((lo, hi))
            for lo, hi in long_notes:
                estimate = estimate_vibrato(data.f0[lo:hi], data.voiced[lo:hi],
                                            self.spec.fs, self.spec.frame_period)
                if estimate:
                    candidates.append(estimate)
        if not candidates:
            return None
        rates = np.array([c[0] for c in candidates])
        depths = np.array([c[1] for c in candidates])
        return {"rate_hz": float(np.median(rates)),
                "depth_semitones": float(np.median(depths))}

    # -- driver ------------------------------------------------------------

    def train(self) -> HMSModel:
        config = self.config
        if not config.label_file:
            raise ValueError("training needs a label file")

        self.log("1/5  reading corpus")
        score = labels_module.load(config.label_file, time_unit=config.time_unit)
        for diagnostic in score.diagnostics:
            self.log(f"  ! {diagnostic}")
        corpus = Corpus(score, Path(config.wav_dir or "."),
                        config.audio_extensions)
        missing = corpus.missing_audio()
        if missing:
            self.log(f"  ! missing audio for {len(missing)} utterance(s): "
                     f"{', '.join(missing[:5])}"
                     f"{' ...' if len(missing) > 5 else ''}")
        self.log(f"  {len(score)} utterances, "
                 f"{corpus.total_seconds:.1f} s of labelled audio")

        self.log("2/5  analysing (WORLD) and aligning")
        utterances = self.analyse_corpus(corpus)

        self.log("3/5  building features")
        offset, scale = self.compute_normalization(utterances)
        phoneme_features, phoneme_voiced, _pitch, durations = \
            self.collect_phoneme_data(utterances, offset, scale)
        n_occurrences = sum(len(v) for v in phoneme_features.values())
        covered_frames = sum(int(sum(len(s) for s in v))
                             for v in phoneme_features.values())
        total_frames = sum(len(u.features) for u in utterances)
        self.log(f"  {len(phoneme_features)} distinct phonemes, "
                 f"{n_occurrences} occurrences, {covered_frames}/{total_frames} "
                 f"frames covered by labels")

        self.log("4/5  training HMMs")
        hmms = self.train_hmms(phoneme_features, phoneme_voiced)
        backoff = self.build_backoff(hmms)
        self.log(f"  {len(hmms)} HMMs, {len(backoff)} backoff model(s), "
                 f"{sum(h.n_free_params for h in hmms.values()):,} free params")

        self.log("5/5  duration, pitch and voicing models")
        duration_model = self.build_duration_model(durations)
        pitch_model = self.build_pitch_model(utterances, offset, scale, hmms)

        stats = ModelStats(
            utterances=len(utterances),
            frames=int(total_frames),
            phoneme_occurrences=n_occurrences,
            duration_seconds=float(total_frames * self.spec.frame_period / 1000.0),
            training_method=config.training_method,
            n_iterations=config.n_iterations,
            created=_dt.datetime.now().isoformat(timespec="seconds"),
            notes=list(score.diagnostics),
        )
        return HMSModel(
            name=str(config.label_file),
            spec=self.spec,
            phoneme_set=self.phoneme_set,
            hmms=hmms,
            duration_model=duration_model,
            pitch_model=pitch_model,
            normalization={"offset": offset, "scale": scale},
            stats=stats,
            backoff=backoff,
            metadata={
                "hms_version": __version__,
                "vocoder": getattr(self.vocoder, "name", "unknown"),
                "label_file": str(config.label_file),
                "wav_dir": str(config.wav_dir),
                "training_config": {k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in vars(config).items()},
            },
        )


def train(config: TrainingConfig, phoneme_set: Optional[PhonemeSet] = None,
          log=None) -> HMSModel:
    """Functional entry point used by the CLI."""
    return Trainer(config, phoneme_set, log).train()
