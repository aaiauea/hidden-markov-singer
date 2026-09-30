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

Nothing here is a neural network and nothing is a black box: the acoustic
model is a compact set of learned parameters (about twenty thousand in the
default demo voice) that you can read out of ``model.yaml`` / ``hmm.npz`` and
reason about.

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

Optional tiers
--------------
Two opt-in tiers sit on top of the per-phoneme models and reuse this pipeline
rather than adding a second one:

* sparse phoneme contexts (``context_enabled``, see `hms.core.context`): extra
  HMMs for the phone contexts the corpus actually shows;
* pitch conditioning (``pitch_conditioning_enabled``, see
  `hms.core.pitch_condition`): the same units additionally split by the pitch
  bin of the note each observation was sung on.  It changes *which pool* an
  observation trains, never how a pool is fitted, and it leaves F0 generation
  alone -- the score still supplies the pitch.

Both are off by default, both keep the corpus disk-backed (buckets hold views
into the temporary feature cache, never copies of it), and both leave every
other tier bit-identical when they are off.
"""

from __future__ import annotations

import datetime as _dt
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hms import __version__
from hms.core import labels as labels_module
from hms.core.context import (KIND_LEFT, KIND_RIGHT, context_keys,
                              context_wildcard)
from hms.core.duration import DurationModel
from hms.core.features import (AcousticFrameSequence, FeatureSpec,
                               add_dynamic_features)
from hms.core.gv import estimate_global_variance
from hms.core.hmm import LeftToRightHMM
from hms.core.model import HMSModel, ModelStats
from hms.core.phonemes import PhonemeSet
from hms.core.pitch import PitchModel, PitchStats, note_relative_pitch
from hms.core.pitch_condition import (DEFAULT_BIN_SIZE, KIND_PHONE,
                                      PitchConditioning, bin_note_bounds,
                                      segment_pitch_bin, validate_bin_size)
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
    duration_variance_scale: float = 1.0
    vibrato_enabled: bool = False
    vibrato_estimate_from_data: bool = True
    pitch_variation: float = 1.0

    # -- optional sparse phoneme contexts (off by default) -----------------
    #: learn HMMs for observed phone contexts (see `hms.core.context`)
    context_enabled: bool = False
    #: minimum pooled frames before a context gets its own HMM
    context_min_frames: int = 100
    #: minimum occurrences before a context gets its own HMM
    context_min_occurrences: int = 3
    #: cap on the number of context HMMs (the best-supported are kept)
    context_max_models: int = 64
    #: also train one-sided diphone contexts (left and right)
    context_partial: bool = True
    #: train a pooled catch-all backoff HMM as the last fallback
    context_global_backoff: bool = False

    # -- optional pitch-conditioned acoustic models (off by default) -------
    #: EXPERIMENTAL: additionally train/resolve acoustic HMMs per pitch bin of
    #: the scored note (see `hms.core.pitch_condition`).  Off by default; it
    #: changes nothing about F0 generation, which stays score-driven.
    pitch_conditioning_enabled: bool = False
    #: width of a pitch bin, in semitones (6 = a tritone, two bins per octave)
    pitch_conditioning_bin_size: int = DEFAULT_BIN_SIZE

    def __post_init__(self) -> None:
        numeric_values = (self.n_iterations, self.min_phoneme_frames,
                          self.var_floor_ratio, self.fs, self.frame_period,
                          self.n_mcep, self.n_band, self.default_note,
                          self.pitch_variation, self.duration_variance_scale)
        try:
            if not all(np.isfinite(value) for value in numeric_values):
                raise ValueError("training configuration values must be finite")
        except TypeError as exc:
            raise ValueError("training configuration values must be numeric") from exc
        if self.fft_size is not None:
            try:
                if not np.isfinite(self.fft_size):
                    raise ValueError("fft_size must be finite")
            except TypeError as exc:
                raise ValueError("fft_size must be numeric") from exc
        if self.covariance_type not in ("diag", "tied"):
            raise ValueError("covariance_type must be 'diag' or 'tied'")
        if self.training_method not in ("viterbi", "baum_welch"):
            raise ValueError("training_method must be 'viterbi' or 'baum_welch'")
        if self.time_unit not in ("seconds", "frames"):
            raise ValueError("time_unit must be 'seconds' or 'frames'")
        if self.f0_estimation not in ("dio", "harvest"):
            raise ValueError("f0_estimation must be 'dio' or 'harvest'")
        if self.vocoder not in ("auto", "native", "pyworld", "builtin"):
            raise ValueError("vocoder must be auto, native, pyworld or builtin")
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")
        if self.min_phoneme_frames < 1:
            raise ValueError("min_phoneme_frames must be at least 1")
        if self.var_floor_ratio < 0 or not np.isfinite(self.var_floor_ratio):
            raise ValueError("var_floor_ratio must be finite and non-negative")
        if self.fs <= 0 or not np.isfinite(self.frame_period) \
                or self.frame_period <= 0:
            raise ValueError("fs and frame_period must be positive")
        if self.fft_size is not None and self.fft_size < 2:
            raise ValueError("fft_size must be >= 2 when specified")
        if self.n_mcep < 2 or self.n_band < 2:
            raise ValueError("n_mcep must be >= 2 and n_band must be >= 2")
        if not np.isfinite(self.default_note):
            raise ValueError("default_note must be finite")
        if not (labels_module.MIDI_NOTE_MIN <= self.default_note
                <= labels_module.MIDI_NOTE_MAX):
            raise ValueError(
                "default_note must be a MIDI note in "
                f"[{labels_module.MIDI_NOTE_MIN:g}, "
                f"{labels_module.MIDI_NOTE_MAX:g}], "
                f"got {self.default_note!r}")
        if not np.isfinite(self.pitch_variation) or self.pitch_variation < 0:
            raise ValueError("pitch_variation must be finite and non-negative")
        if self.duration_variance_scale < 0:
            raise ValueError("duration_variance_scale must be non-negative")
        if self.context_min_frames < 1:
            raise ValueError("context_min_frames must be at least 1")
        if self.context_min_occurrences < 1:
            raise ValueError("context_min_occurrences must be at least 1")
        if self.context_max_models < 0:
            raise ValueError("context_max_models must be non-negative")
        # normalises e.g. 6.0 -> 6 and rejects anything that is not a whole
        # number of semitones inside the supported range
        self.pitch_conditioning_bin_size = validate_bin_size(
            self.pitch_conditioning_bin_size)

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
    """One analysed utterance, ready to be cut into phoneme segments.

    Trainer-created instances retain only values consumed after feature
    encoding. WORLD's full spectral envelope and aperiodicity arrays are much
    larger than the model features and are left unset for compatibility with
    older callers that used these optional fields.
    """

    name: str
    f0: np.ndarray
    voiced: np.ndarray                    # per frame, bool
    relative_pitch: np.ndarray            # per frame, semitones re. the note
    features: np.ndarray                  # (T, dim) with dynamic streams
    phoneme_spans: List[Tuple[str, int, int]]
    diagnostics: List[str] = field(default_factory=list)
    # Kept as optional constructor fields for older callers; the trainer no
    # longer populates them because none is used after feature extraction.
    sp: Optional[np.ndarray] = None
    ap: Optional[np.ndarray] = None
    phones: Optional[List[str]] = None
    notes: Optional[np.ndarray] = None
    note_semitones: Optional[np.ndarray] = None
    #: Scored note per entry of ``phoneme_spans`` (``None`` where the label has
    #: none).  One float per segment, so it costs nothing next to the frames;
    #: it is what optional pitch conditioning groups observations by.  Optional
    #: for older callers, which then simply carry no pitch condition.
    span_notes: Optional[List[Optional[float]]] = None


@dataclass
class _UtteranceAnalysis:
    """Temporary arrays for one utterance during feature extraction only."""

    f0: np.ndarray
    static_features: np.ndarray
    voiced: np.ndarray
    relative_pitch: np.ndarray
    phoneme_spans: List[Tuple[str, int, int]]
    span_notes: List[Optional[float]] = field(default_factory=list)


@dataclass
class _CachedUtterance:
    """Small metadata record for training data held in temporary .npy files.

    The per-frame arrays are memory-mapped only while a stage consumes them;
    this record deliberately contains paths and spans, never feature arrays.
    """

    name: str
    features_path: Path
    relative_pitch_path: Path
    voiced_path: Path
    phoneme_spans: List[Tuple[str, int, int]]
    n_frames: int
    #: Scored note per entry of ``phoneme_spans`` (see `UtteranceData`).
    span_notes: List[Optional[float]] = field(default_factory=list)


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

    def _analyse_utterance(self, utterance: labels_module.Utterance,
                           path: Path) -> _UtteranceAnalysis:
        """Analyse one clip and return only data needed by later stages.

        The returned object intentionally excludes ``sp`` and ``ap``.  Those
        WORLD arrays are consumed by ``FeatureSpec.encode`` and are released as
        soon as this method returns, rather than being retained for every
        utterance in the corpus.
        """
        self.spec = self.spec or self.build_spec()
        spec = self.spec
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
        # WORLD does not retain the waveform. Drop it before the relatively
        # large spectral arrays are transformed into the compact feature space.
        del signal

        n_frames = len(sequence)
        phones, notes, _ = utterance.frame_labels(
            spec.frame_period, self.config.default_note, n_frames=n_frames)
        del phones
        voiced = sequence.f0 > spec.voiced_threshold
        relative = note_relative_pitch(
            sequence.f0, labels_module.midi_to_hz(notes), voiced,
            spec.f0_ref_hz)
        del notes
        static = spec.encode(sequence.f0, sequence.sp, sequence.ap)
        # Dimension 0 carries note-relative pitch instead of absolute pitch.
        static[:, 0] = relative
        boundaries = labels_module.segment_boundaries(
            utterance, spec.frame_period, n_frames=n_frames)
        spans = [(phone, lo, hi) for phone, lo, hi, _note in boundaries]
        # The scored note of each segment travels with its span.  It is the
        # only pitch condition HMS uses -- the stable musical note, never a
        # measured per-frame F0 -- and optional pitch conditioning is the only
        # consumer, so it costs one float per segment.
        span_notes = [note for _phone, _lo, _hi, note in boundaries]
        return _UtteranceAnalysis(
            f0=sequence.f0, static_features=static, voiced=voiced,
            relative_pitch=relative, phoneme_spans=spans,
            span_notes=span_notes)

    def analyse_corpus(self, corpus: Corpus) -> List[UtteranceData]:
        """WORLD analysis + label alignment for every utterance.

        This convenience path is used by corpus evaluation. Training itself
        uses :meth:`_prepare_training_cache`, which writes compact features to a
        temporary disk-backed cache instead of retaining this list.
        """
        self.spec = self.spec or self.build_spec()
        spec = self.spec
        hop_ms = spec.frame_period / 1000.0
        out: List[UtteranceData] = []

        for utterance in corpus.score:
            path = corpus.audio_path(utterance.name)
            if path is None:
                self.log(f"  ! {utterance.name}: no audio found, skipped")
                continue
            analysis = self._analyse_utterance(utterance, path)
            features = add_dynamic_features(analysis.static_features,
                                            spec.use_delta, spec.use_delta2,
                                            window=spec.delta_window)
            n_frames = len(analysis.f0)
            out.append(UtteranceData(
                name=utterance.name, f0=analysis.f0,
                voiced=analysis.voiced,
                relative_pitch=analysis.relative_pitch,
                features=features,
                phoneme_spans=analysis.phoneme_spans,
                span_notes=analysis.span_notes,
                diagnostics=[d for d in corpus.score.diagnostics
                              if d.startswith(utterance.name)]))
            self.log(f"  analysed {utterance.name}: {n_frames} frames "
                     f"({n_frames * hop_ms:.2f} s)")
            del analysis, features
        if not out:
            raise RuntimeError("no training utterances were analysed")
        return out

    # -- stage 2: normalise -------------------------------------------------

    def _normalization_from_moments(self, total: np.ndarray,
                                    total_sq: np.ndarray, count: int
                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finish corpus normalization from per-dimension running moments."""
        static_dim = self.spec.static_dim
        if count == 0:
            return np.zeros(static_dim), np.ones(static_dim)
        mean = total / count
        variance = np.maximum(total_sq / count - mean ** 2, 1e-6)
        scale = 1.0 / np.sqrt(variance) if self.config.normalize_scale \
            else np.ones(static_dim)
        scale = np.minimum(scale, 100.0)          # guard against flat dimensions
        return mean, scale

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
        return self._normalization_from_moments(total, total_sq, count)

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

    def _cache_static_utterance(
            self, utterance: labels_module.Utterance, path: Path,
            cache_dir: Path, index: int
            ) -> Tuple[_CachedUtterance, Path, np.ndarray, np.ndarray,
                       List[Tuple[float, float]]]:
        """Analyse one utterance, persist only its compact static features.

        The waveform, WORLD spectra, and all frame arrays stay local to this
        call.  The static matrix is written to a temporary ``.npy`` file so
        corpus normalization can be computed without keeping matrices from
        previously analysed utterances in Python memory.
        """
        analysis = self._analyse_utterance(utterance, path)
        static = analysis.static_features
        n_frames = len(static)
        stem = f"utterance_{index:05d}"
        static_path = cache_dir / f"{stem}.static.npy"
        record = _CachedUtterance(
            name=utterance.name,
            features_path=cache_dir / f"{stem}.features.npy",
            relative_pitch_path=cache_dir / f"{stem}.pitch.npy",
            voiced_path=cache_dir / f"{stem}.voiced.npy",
            phoneme_spans=analysis.phoneme_spans,
            n_frames=n_frames,
            span_notes=analysis.span_notes)

        np.save(static_path, static)
        np.save(record.voiced_path, analysis.voiced)
        total = static.sum(axis=0)
        total_sq = (static ** 2).sum(axis=0)

        vibrato_candidates: List[Tuple[float, float]] = []
        if self.config.vibrato_enabled and self.config.vibrato_estimate_from_data:
            from hms.core.pitch import estimate_vibrato

            long_spans = {(lo, hi) for _phone, lo, hi in analysis.phoneme_spans
                          if hi - lo >= 80}
            for lo, hi in long_spans:
                estimate = estimate_vibrato(
                    analysis.f0[lo:hi], analysis.voiced[lo:hi],
                    self.spec.fs, self.spec.frame_period)
                if estimate:
                    vibrato_candidates.append(estimate)

        return record, static_path, total, total_sq, vibrato_candidates

    def _prepare_training_cache(
            self, corpus: Corpus, cache_dir: Path
            ) -> Tuple[List[_CachedUtterance], np.ndarray, np.ndarray,
                       int, List[Tuple[float, float]]]:
        """Analyse incrementally, normalize, and spool training features.

        Pass one retains only corpus-wide scalar moments and writes each
        utterance's static features to disk.  Once the final normalization is
        known, pass two turns one cached utterance at a time into normalized
        static+delta features.  The cache contains the per-frame sequences that
        the multi-utterance HMM estimators genuinely need, but they remain
        disk-backed until a particular phone/model consumes them.
        """
        self.spec = self.spec or self.build_spec()
        spec = self.spec
        static_dim = spec.static_dim
        total = np.zeros(static_dim, dtype=np.float64)
        total_sq = np.zeros(static_dim, dtype=np.float64)
        frame_count = 0
        records: List[_CachedUtterance] = []
        static_paths: List[Path] = []
        vibrato_candidates: List[Tuple[float, float]] = []

        for utterance in corpus.score:
            path = corpus.audio_path(utterance.name)
            if path is None:
                self.log(f"  ! {utterance.name}: no audio found, skipped")
                continue
            record, static_path, utt_total, utt_total_sq, candidates = \
                self._cache_static_utterance(
                    utterance, path, cache_dir, len(records))
            records.append(record)
            static_paths.append(static_path)
            total += utt_total
            total_sq += utt_total_sq
            frame_count += record.n_frames
            vibrato_candidates.extend(candidates)
            self.log(f"  analysed {utterance.name}: {record.n_frames} frames "
                     f"({record.n_frames * spec.frame_period / 1000.0:.2f} s)")

        if not records:
            raise RuntimeError("no training utterances were analysed")
        offset, scale = self._normalization_from_moments(
            total, total_sq, frame_count)

        # Normalize/rebuild dynamics one utterance at a time. The temporary
        # static file is removed immediately after its compact training feature
        # file and pitch stream have been written.
        for record, static_path in zip(records, static_paths):
            # NumPy cannot mmap a zero-byte data payload; empty utterances are
            # harmless and keep the in-memory path bounded to an empty array.
            static = np.load(static_path, mmap_mode="r" if record.n_frames else None)
            np.save(record.relative_pitch_path, static[:, 0])
            normalized_static = np.empty(static.shape, dtype=np.float64)
            np.subtract(static, offset, out=normalized_static)
            np.multiply(normalized_static, scale, out=normalized_static)
            features = add_dynamic_features(
                normalized_static, spec.use_delta, spec.use_delta2,
                window=spec.delta_window)
            np.save(record.features_path, features)
            del static, normalized_static, features
            static_path.unlink()

        return records, offset, scale, frame_count, vibrato_candidates

    def _collect_cached_phoneme_data(
            self, utterances: Sequence[_CachedUtterance]
            ) -> Tuple[Dict[str, List[np.ndarray]],
                       Dict[str, List[np.ndarray]], Dict[str, List[float]]]:
        """Build HMM sequence indexes over memory-mapped utterance features."""
        features: Dict[str, List[np.ndarray]] = {}
        voiced: Dict[str, List[np.ndarray]] = {}
        durations: Dict[str, List[float]] = {}
        for data in utterances:
            if not data.n_frames:
                continue
            feature_matrix = np.load(data.features_path, mmap_mode="r")
            voiced_frames = np.load(data.voiced_path, mmap_mode="r")
            for phone, lo, hi in data.phoneme_spans:
                if hi <= lo:
                    continue
                canonical = self.phoneme_set.canonical(phone)
                # Slices remain views on the .npy mmap; no utterance-sized or
                # corpus-sized feature copy is made by this index.
                features.setdefault(canonical, []).append(
                    feature_matrix[lo:hi])
                voiced.setdefault(canonical, []).append(voiced_frames[lo:hi])
                durations.setdefault(canonical, []).append(float(hi - lo))
            del feature_matrix, voiced_frames
        return features, voiced, durations

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
        return hmms

    def build_backoff(self, features: Dict[str, List[np.ndarray]],
                      voiced: Dict[str, List[np.ndarray]]
                      ) -> Dict[str, LeftToRightHMM]:
        """Train one class backoff HMM from pooled phoneme frame sequences.

        The GMMs are fitted directly to the class's training frames. This keeps
        component meanings grounded in the pooled data and makes every frame,
        rather than every phoneme model, contribute to the estimates. Rare
        phones skipped by :meth:`train_hmms` are deliberately included here.
        """
        pooled_features: Dict[str, List[np.ndarray]] = {}
        pooled_voiced: Dict[str, List[np.ndarray]] = {}
        for phone in sorted(features):
            definition = self.phoneme_set.resolve(phone)
            if definition is None:
                continue
            sequences = features[phone]
            phone_voiced = voiced.get(phone)
            if phone_voiced is not None and len(phone_voiced) != len(sequences):
                raise ValueError(f"phoneme {phone!r} has {len(sequences)} feature "
                                 f"sequences but {len(phone_voiced)} voicing sequences")
            for index, sequence in enumerate(sequences):
                sequence = np.asarray(sequence, dtype=np.float64)
                pooled_features.setdefault(definition.type, []).append(sequence)
                if phone_voiced is None:
                    pooled_voiced.setdefault(definition.type, []).append(
                        np.zeros(len(sequence), dtype=bool))
                else:
                    flags = np.asarray(phone_voiced[index], dtype=bool).reshape(-1)
                    if len(flags) != len(sequence):
                        raise ValueError(f"phoneme {phone!r} feature and voicing "
                                         "sequence lengths differ")
                    pooled_voiced.setdefault(definition.type, []).append(flags)

        backoff: Dict[str, LeftToRightHMM] = {}
        for class_index, (klass, sequences) in enumerate(
                sorted(pooled_features.items())):
            total_frames = sum(len(sequence) for sequence in sequences)
            defaults = self.phoneme_set.defaults.get(klass, {})
            requested_states = int(defaults.get("n_states", 1))
            n_components = int(defaults.get("n_components", 1))
            n_states = min(requested_states, max(1, total_frames // 2))
            hmm = LeftToRightHMM(
                n_states=n_states, allow_skip=self.config.allow_skip,
                covariance_type=self.config.covariance_type)
            hmm.train(
                sequences, n_components=n_components,
                covariance_type=self.config.covariance_type,
                n_iterations=self.config.n_iterations,
                var_floor_ratio=self.config.var_floor_ratio,
                seed=self.config.seed + 10_000 * class_index,
                method=self.config.training_method,
                voiced=pooled_voiced[klass])
            backoff[klass] = hmm
            self.log(f"  backoff {klass:20s} {n_states} states x "
                     f"{hmm.states[0].gmm.n_components} comp, "
                     f"{total_frames} pooled frames")
        return backoff

    # -- stage 3b: optional sparse phoneme contexts ------------------------

    def _collect_cached_context_data(
            self, utterances: Sequence[_CachedUtterance]
            ) -> Dict[str, Dict[str, object]]:
        """Group context occurrences using views into the disk-backed cache."""
        config = self.config
        silence = self.phoneme_set.silence
        wildcard = context_wildcard(self.phoneme_set.phonemes)
        occurrences: Dict[str, Dict[str, object]] = {}
        for data in utterances:
            if not data.n_frames:
                continue
            feature_matrix = np.load(data.features_path, mmap_mode="r")
            voiced_frames = np.load(data.voiced_path, mmap_mode="r")
            spans = [(self.phoneme_set.canonical(phone), lo, hi)
                     for phone, lo, hi in data.phoneme_spans if hi > lo]
            for index, (curr, lo, hi) in enumerate(spans):
                pre = spans[index - 1][0] if index > 0 else silence
                post = spans[index + 1][0] if index < len(spans) - 1 \
                    else silence
                for key, kind in context_keys(pre, curr, post, wildcard,
                                              partial=config.context_partial):
                    entry = occurrences.get(key)
                    if entry is None:
                        entry = {
                            "kind": kind,
                            "pre": pre if kind != KIND_RIGHT else None,
                            "curr": curr,
                            "post": post if kind != KIND_LEFT else None,
                            "features": [],
                            "voiced": [],
                        }
                        occurrences[key] = entry
                    entry["features"].append(feature_matrix[lo:hi])
                    entry["voiced"].append(voiced_frames[lo:hi])
            del feature_matrix, voiced_frames
        return occurrences

    def collect_context_data(self, utterances: Sequence[UtteranceData],
                             offset: np.ndarray, scale: np.ndarray
                             ) -> Dict[str, Dict[str, object]]:
        """Group frames by observed phone context.

        Returns a mapping ``key -> entry`` where ``entry`` holds the context
        ``kind`` (triphone / left / right diphone), the ``pre``/``curr``/``post``
        phones (``None`` on the missing side of a diphone) and the pooled raw
        feature and voicing sequences of every occurrence.  Utterance
        boundaries use the inventory's silence symbol; there are no BOS/EOS
        tokens.
        """
        config = self.config
        silence = self.phoneme_set.silence
        wildcard = context_wildcard(self.phoneme_set.phonemes)
        occurrences: Dict[str, Dict[str, object]] = {}
        for data in utterances:
            normalized = self.normalize_features(data.features, offset, scale)
            spans = [(self.phoneme_set.canonical(phone), lo, hi)
                     for phone, lo, hi in data.phoneme_spans if hi > lo]
            for index, (curr, lo, hi) in enumerate(spans):
                pre = spans[index - 1][0] if index > 0 else silence
                post = spans[index + 1][0] if index < len(spans) - 1 \
                    else silence
                for key, kind in context_keys(pre, curr, post, wildcard,
                                              partial=config.context_partial):
                    entry = occurrences.get(key)
                    if entry is None:
                        entry = {
                            "kind": kind,
                            "pre": pre if kind != KIND_RIGHT else None,
                            "curr": curr,
                            "post": post if kind != KIND_LEFT else None,
                            "features": [],
                            "voiced": [],
                        }
                        occurrences[key] = entry
                    entry["features"].append(normalized[lo:hi])
                    entry["voiced"].append(data.voiced[lo:hi])
        return occurrences

    def select_context_models(self, occurrences: Dict[str, Dict[str, object]]
                              ) -> List[str]:
        """Pick which observed contexts get their own HMM.

        A context qualifies when it clears both support thresholds
        (`context_min_frames`, `context_min_occurrences`); the best-supported
        ones are kept up to the `context_max_models` cap.  Ordering is
        deterministic: frames descending, occurrences descending, key
        ascending.
        """
        config = self.config
        candidates: List[Tuple[int, int, str]] = []
        for key, entry in occurrences.items():
            sequences = entry["features"]
            frames = int(sum(len(s) for s in sequences))
            count = len(sequences)
            if frames < config.context_min_frames \
                    or count < config.context_min_occurrences:
                continue
            candidates.append((frames, count, key))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [key for _frames, _count, key
                in candidates[:max(0, config.context_max_models)]]

    def train_contexts(self, occurrences: Dict[str, Dict[str, object]],
                       selected: Sequence[str], seed_base: int
                       ) -> Tuple[Dict[str, LeftToRightHMM], Dict[str, dict]]:
        """Train one HMM per selected context, directly from pooled frames.

        Like the class backoffs, each context model is fitted to the raw
        feature sequences of its occurrences -- never by averaging separately
        trained phone models.  The state/component budget follows the current
        phone's definition, so a context of a vowel spends like a vowel.
        """
        config = self.config
        contexts: Dict[str, LeftToRightHMM] = {}
        index: Dict[str, dict] = {}
        for position, key in enumerate(selected):
            entry = occurrences[key]
            sequences = entry["features"]
            voiced = entry["voiced"]
            total_frames = int(sum(len(s) for s in sequences))
            curr = entry["curr"]
            definition = self.phoneme_set.resolve(curr)
            if definition is not None:
                requested_states = definition.n_states
                n_components = definition.n_components
            else:
                defaults = self.phoneme_set.defaults.get("unvoiced_consonant", {})
                requested_states = int(defaults.get("n_states", 2))
                n_components = int(defaults.get("n_components", 1))
            n_states = min(requested_states, max(1, total_frames // 2))
            hmm = LeftToRightHMM(
                n_states=n_states, allow_skip=config.allow_skip,
                covariance_type=config.covariance_type)
            hmm.train(sequences, n_components=n_components,
                      covariance_type=config.covariance_type,
                      n_iterations=config.n_iterations,
                      var_floor_ratio=config.var_floor_ratio,
                      seed=seed_base + position,
                      method=config.training_method, voiced=voiced)
            contexts[key] = hmm
            index[key] = {
                "kind": entry["kind"],
                "pre": entry["pre"],
                "curr": entry["curr"],
                "post": entry["post"],
                "frames": total_frames,
                "occurrences": len(sequences),
                "n_states": hmm.n_states,
                "n_components": hmm.states[0].gmm.n_components,
                "covariance": hmm.covariance_type,
                "allow_skip": bool(hmm.allow_skip),
                "n_free_params": hmm.n_free_params,
            }
            self.log(f"  context {key:18s} ({entry['kind']}) "
                     f"{hmm.n_states} states x "
                     f"{hmm.states[0].gmm.n_components} comp, "
                     f"{total_frames} frames / {len(sequences)} occ")
        return contexts, index

    def build_global_backoff(self, features: Dict[str, List[np.ndarray]],
                             voiced: Dict[str, List[np.ndarray]],
                             seed: int) -> LeftToRightHMM:
        """One pooled catch-all HMM over every training frame.

        The last rung of the context fallback hierarchy: it covers phones with
        neither a dedicated, context, nor class model.  Deliberately small
        (3 states, 1 component) -- it is a safety net, not an acoustic model.
        """
        pooled: List[np.ndarray] = []
        pooled_voiced: List[np.ndarray] = []
        for phone in sorted(features):
            sequences = features[phone]
            phone_voiced = voiced.get(phone)
            for position, sequence in enumerate(sequences):
                sequence = np.asarray(sequence, dtype=np.float64)
                pooled.append(sequence)
                if phone_voiced is None:
                    pooled_voiced.append(np.zeros(len(sequence), dtype=bool))
                else:
                    pooled_voiced.append(
                        np.asarray(phone_voiced[position], dtype=bool)
                        .reshape(-1))
        total_frames = sum(len(sequence) for sequence in pooled)
        n_states = min(3, max(1, total_frames // 2))
        hmm = LeftToRightHMM(
            n_states=n_states, allow_skip=self.config.allow_skip,
            covariance_type=self.config.covariance_type)
        hmm.train(pooled, n_components=1,
                  covariance_type=self.config.covariance_type,
                  n_iterations=self.config.n_iterations,
                  var_floor_ratio=self.config.var_floor_ratio,
                  seed=seed, method=self.config.training_method,
                  voiced=pooled_voiced)
        self.log(f"  global backoff         {hmm.n_states} states x "
                 f"{hmm.states[0].gmm.n_components} comp, "
                 f"{total_frames} pooled frames")
        return hmm

    # -- stage 3c: optional pitch-conditioned acoustic models --------------

    @staticmethod
    def _span_note(span_notes: Optional[Sequence[Optional[float]]],
                   index: int) -> Optional[float]:
        """The scored note of one span, or ``None`` when it carries none.

        Tolerates callers that build utterance data without notes (older code
        paths and hand-made test data), which then simply contribute no pitch
        condition instead of an invented one.
        """
        if not span_notes or index >= len(span_notes):
            return None
        return span_notes[index]

    def _pitch_span_units(self, spans: Sequence[Tuple[str, int, int]],
                          span_notes: Optional[Sequence[Optional[float]]],
                          features: np.ndarray, voiced: np.ndarray,
                          context_units: Optional[Sequence[str]] = None):
        """Yield ``(unit, kind, curr, bin, feature view, voiced view)``.

        One entry per (segment, unit) pair that carries a pitch condition, in
        label order.  ``features``/``voiced`` are whole-utterance arrays -- a
        memory map in the cached training path -- and what is yielded are
        *slices* of them, so grouping observations by pitch bin copies nothing.

        Units are the current phone plus, when ``context_units`` is given, the
        contexts of this occurrence that actually earned their own HMM: a
        pitch-conditioned context bucket is a subset of that context's data, so
        contexts the corpus did not support are not conditioned either.
        """
        config = self.config
        canonical = self.phoneme_set.canonical
        silence = self.phoneme_set.silence
        bin_size = config.pitch_conditioning_bin_size
        wildcard = context_wildcard(self.phoneme_set.phonemes) \
            if context_units else None
        indexed = [(canonical(phone), lo, hi, self._span_note(span_notes, i))
                   for i, (phone, lo, hi) in enumerate(spans) if hi > lo]
        for position, (curr, lo, hi, note) in enumerate(indexed):
            # The condition is the *scored note* of the segment, never its
            # measured F0: silence, unnoted segments and invalid notes carry no
            # condition at all (see `hms.core.pitch_condition`).
            pitch = segment_pitch_bin(curr, note, self.phoneme_set, bin_size)
            if pitch is None:
                continue
            units = [(curr, KIND_PHONE)]
            if context_units:
                pre = indexed[position - 1][0] if position > 0 else silence
                post = indexed[position + 1][0] \
                    if position < len(indexed) - 1 else silence
                for key, kind in context_keys(
                        pre, curr, post, wildcard,
                        partial=config.context_partial):
                    if key in context_units:
                        units.append((key, kind))
            for unit, kind in units:
                yield unit, kind, curr, pitch, features[lo:hi], voiced[lo:hi]

    def _accumulate_pitch_occurrences(
            self, occurrences: Dict[Tuple[str, int], Dict[str, object]],
            spans: Sequence[Tuple[str, int, int]],
            span_notes: Optional[Sequence[Optional[float]]],
            features: np.ndarray, voiced: np.ndarray,
            context_units: Optional[Sequence[str]] = None) -> None:
        """Pool one utterance's conditioned spans into ``occurrences``."""
        for unit, kind, curr, pitch, feature_view, voiced_view \
                in self._pitch_span_units(spans, span_notes, features, voiced,
                                          context_units):
            entry = occurrences.get((unit, pitch))
            if entry is None:
                entry = {"kind": kind, "unit": unit, "curr": curr,
                         "pitch_bin": pitch, "features": [], "voiced": []}
                occurrences[(unit, pitch)] = entry
            entry["features"].append(feature_view)
            entry["voiced"].append(voiced_view)

    def _collect_cached_pitch_data(
            self, utterances: Sequence[_CachedUtterance],
            context_units: Optional[Sequence[str]] = None
            ) -> Dict[Tuple[str, int], Dict[str, object]]:
        """Group pitch-conditioned buckets over the disk-backed cache.

        One utterance is mapped at a time and only its slices are kept, so a
        corpus is never resident in memory because it was split by pitch bin:
        the buckets hold views into the same temporary ``.npy`` files the phone
        and context tiers use.
        """
        occurrences: Dict[Tuple[str, int], Dict[str, object]] = {}
        for data in utterances:
            if not data.n_frames:
                continue
            feature_matrix = np.load(data.features_path, mmap_mode="r")
            voiced_frames = np.load(data.voiced_path, mmap_mode="r")
            self._accumulate_pitch_occurrences(
                occurrences, data.phoneme_spans, data.span_notes,
                feature_matrix, voiced_frames, context_units)
            del feature_matrix, voiced_frames
        return occurrences

    def collect_pitch_data(self, utterances: Sequence[UtteranceData],
                           offset: np.ndarray, scale: np.ndarray,
                           context_units: Optional[Sequence[str]] = None
                           ) -> Dict[Tuple[str, int], Dict[str, object]]:
        """Group frames by ``(unit, pitch bin)`` for in-memory utterance data.

        Returns a mapping ``(unit key, bin) -> entry`` where ``entry`` holds the
        unit ``kind`` (``phone``, ``triphone``, ``left``, ``right``), the unit
        key itself, the ``curr`` phone, the ``pitch_bin`` and the pooled raw
        feature/voicing sequences of every occurrence.  This is the analysis
        counterpart of `_collect_cached_pitch_data`, used wherever utterances
        are already in memory (evaluation, tests).
        """
        occurrences: Dict[Tuple[str, int], Dict[str, object]] = {}
        for data in utterances:
            normalized = self.normalize_features(data.features, offset, scale)
            self._accumulate_pitch_occurrences(
                occurrences, data.phoneme_spans, data.span_notes, normalized,
                np.asarray(data.voiced, dtype=bool), context_units)
        return occurrences

    def select_pitch_models(self, occurrences: Dict[Tuple[str, int],
                                                    Dict[str, object]]
                            ) -> List[Tuple[str, int]]:
        """Pick which pitch-conditioned buckets get their own HMM.

        The support philosophy of the tier each bucket conditions, unchanged: a
        phone-conditioned bucket must clear `min_phoneme_frames` (what a
        dedicated phone model must clear), and a context-conditioned bucket must
        clear `context_min_frames` and `context_min_occurrences`.  A pitch bin
        holds a *subset* of its unit's frames, so earning a conditioned model is
        never easier than earning the unconditioned one -- and a bucket that
        falls short is simply not created, leaving synthesis on the ordinary
        hierarchy instead of on a model fitted to a handful of frames.

        Ordering is deterministic: frames descending, occurrences descending,
        unit key ascending, bin ascending.
        """
        config = self.config
        candidates: List[Tuple[int, int, str, int]] = []
        for (unit, pitch), entry in occurrences.items():
            sequences = entry["features"]
            frames = int(sum(len(s) for s in sequences))
            count = len(sequences)
            if entry["kind"] == KIND_PHONE:
                if frames < config.min_phoneme_frames:
                    continue
            elif frames < config.context_min_frames \
                    or count < config.context_min_occurrences:
                continue
            candidates.append((-frames, -count, unit, int(pitch)))
        candidates.sort()
        return [(unit, pitch) for _frames, _count, unit, pitch in candidates]

    def train_pitch_models(self, occurrences: Dict[Tuple[str, int],
                                                   Dict[str, object]],
                           selected: Sequence[Tuple[str, int]], seed_base: int
                           ) -> Tuple[Dict[Tuple[str, int], LeftToRightHMM],
                                      Dict[Tuple[str, int], dict]]:
        """Train one HMM per selected ``(unit, pitch bin)`` bucket.

        This adds no estimator: each bucket is fitted with the same
        `LeftToRightHMM.train` call the phone, context and backoff tiers use, so
        iterations, variance floors, covariance tying, voicing statistics and
        seed handling are identical -- the pitch condition only decides *which
        pool* an observation lands in.  The state/component budget follows the
        current phone's definition, and feature normalisation stays the single
        corpus-wide one: a pitch bin is a model-selection condition, not a new
        feature space, so bins remain comparable and MLPG sees one geometry.
        """
        config = self.config
        models: Dict[Tuple[str, int], LeftToRightHMM] = {}
        index: Dict[Tuple[str, int], dict] = {}
        for position, key in enumerate(selected):
            entry = occurrences[key]
            sequences = entry["features"]
            voiced = entry["voiced"]
            total_frames = int(sum(len(s) for s in sequences))
            curr = entry["curr"]
            definition = self.phoneme_set.resolve(curr)
            if definition is not None:
                requested_states = definition.n_states
                n_components = definition.n_components
            else:
                defaults = self.phoneme_set.defaults.get("unvoiced_consonant", {})
                requested_states = int(defaults.get("n_states", 2))
                n_components = int(defaults.get("n_components", 1))
            n_states = min(requested_states, max(1, total_frames // 2))
            hmm = LeftToRightHMM(
                n_states=n_states, allow_skip=config.allow_skip,
                covariance_type=config.covariance_type)
            hmm.train(sequences, n_components=n_components,
                      covariance_type=config.covariance_type,
                      n_iterations=config.n_iterations,
                      var_floor_ratio=config.var_floor_ratio,
                      seed=seed_base + position,
                      method=config.training_method, voiced=voiced)
            unit, pitch = key
            low, high = bin_note_bounds(pitch,
                                        config.pitch_conditioning_bin_size)
            models[key] = hmm
            # The record carries the notes its bin stands for, so model.yaml
            # describes the conditioning on its own -- no training
            # configuration needed to read it back.
            index[key] = {
                "kind": entry["kind"],
                "unit": unit,
                "curr": curr,
                "pitch_bin": int(pitch),
                "note_min": low,
                "note_max": high,
                "frames": total_frames,
                "occurrences": len(sequences),
                "n_states": hmm.n_states,
                "n_components": hmm.states[0].gmm.n_components,
                "covariance": hmm.covariance_type,
                "allow_skip": bool(hmm.allow_skip),
                "n_free_params": hmm.n_free_params,
            }
            self.log(f"  pitch {unit:18s} bin {pitch:<3d} "
                     f"(MIDI {low:3d}-{high:3d}) {hmm.n_states} states x "
                     f"{hmm.states[0].gmm.n_components} comp, "
                     f"{total_frames} frames / {len(sequences)} occ")
        return models, index

    # -- stage 4: duration and pitch ---------------------------------------

    def build_duration_model(self, durations: Dict[str, List[float]]
                             ) -> DurationModel:
        model = DurationModel(variance_scale=self.config.duration_variance_scale)
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

    @staticmethod
    def _merge_moments(count: int, mean: float, m2: float,
                       values: np.ndarray) -> Tuple[int, float, float]:
        """Merge one frame batch into running population moments."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if not values.size:
            return count, mean, m2
        batch_count = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())
        if count == 0:
            return batch_count, batch_mean, batch_m2
        combined_count = count + batch_count
        delta = batch_mean - mean
        combined_mean = mean + delta * batch_count / combined_count
        combined_m2 = (m2 + batch_m2
                       + delta * delta * count * batch_count / combined_count)
        return combined_count, combined_mean, combined_m2

    def _build_pitch_model_from_cache(
            self, utterances: Sequence[_CachedUtterance],
            hmms: Dict[str, LeftToRightHMM],
            vibrato_candidates: Sequence[Tuple[float, float]]) -> PitchModel:
        """Accumulate state pitch/voicing statistics without retaining frames."""
        model = PitchModel(pitch_variation=self.config.pitch_variation)
        pitch_moments: Dict[str, List[Tuple[int, float, float]]] = {}
        voiced_counts: Dict[str, Tuple[int, int]] = {}

        # HMM state assignments are only available after acoustic training, so
        # revisit the disk-backed utterance files. Each frame contributes to a
        # few scalar accumulators and is released before the next occurrence.
        for data in utterances:
            if not data.n_frames:
                continue
            features = np.load(data.features_path, mmap_mode="r")
            relative_pitch = np.load(data.relative_pitch_path, mmap_mode="r")
            voiced = np.load(data.voiced_path, mmap_mode="r")
            for phone, lo, hi in data.phoneme_spans:
                canonical = self.phoneme_set.canonical(phone)
                hmm = hmms.get(canonical)
                if hmm is None or hi <= lo:
                    continue
                state_path = hmm.segment(features[lo:hi])
                moments = pitch_moments.setdefault(
                    canonical, [(0, 0.0, 0.0) for _ in range(hmm.n_states)])
                previous_voiced, previous_frames = voiced_counts.get(
                    canonical, (0, 0))
                phone_voiced = np.asarray(voiced[lo:hi], dtype=bool)
                voiced_counts[canonical] = (
                    previous_voiced + int(phone_voiced.sum()),
                    previous_frames + len(phone_voiced))

                pitch_segment = relative_pitch[lo:hi]
                for state in range(hmm.n_states):
                    indices = np.flatnonzero(state_path == state)
                    if not len(indices):
                        continue
                    state_values = np.asarray(pitch_segment[indices],
                                              dtype=np.float64)
                    moments[state] = self._merge_moments(
                        *moments[state], state_values)
            del features, relative_pitch, voiced

        for phone, moments in pitch_moments.items():
            model.stats[phone] = [
                PitchStats(
                    mean=mean if count else 0.0,
                    variance=max(m2 / count, 1e-4) if count else 0.35 ** 2,
                    count=count)
                for count, mean, m2 in moments]
            voiced_total, frame_total = voiced_counts.get(phone, (0, 0))
            if frame_total:
                model.voiced_prior[phone] = voiced_total / frame_total

        if self.config.vibrato_enabled:
            model.vibrato.enabled = True
            if self.config.vibrato_estimate_from_data and vibrato_candidates:
                rates = np.asarray([item[0] for item in vibrato_candidates])
                depths = np.asarray([item[1] for item in vibrato_candidates])
                estimate = {"rate_hz": float(np.median(rates)),
                            "depth_semitones": float(np.median(depths))}
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

    # -- input validation --------------------------------------------------

    def note_range_warnings(self, score: labels_module.Score) -> List[str]:
        """Diagnostics for score notes outside the analysis F0 range.

        F0 is only ever *estimated* within ``f0_floor``-``f0_ceil`` (Hz), so a
        label note beyond that boundary can never match any extracted F0 and
        the note-relative pitch learned on that segment is wrong by the whole
        difference.  The notes are kept -- they may be valid MIDI numbers the
        singer or the extractor simply cannot produce; the point is to say so
        instead of training on silently garbage features.  Unnoted segments
        are checked against ``default_note``.
        """
        low_hz, high_hz = self.config.f0_floor, self.config.f0_ceiling
        low_midi = float(labels_module.hz_to_midi(low_hz))
        high_midi = float(labels_module.hz_to_midi(high_hz))
        warnings: List[str] = []
        for utterance in score:
            out_of_range: set = set()
            for segment in utterance.segments:
                note = (segment.note if segment.note is not None
                        else self.config.default_note)
                if note < low_midi or note > high_midi:
                    out_of_range.add(round(float(note), 4))
            if out_of_range:
                listing = ", ".join(
                    f"{value:g} ({labels_module.midi_to_hz(value):.0f} Hz)"
                    for value in sorted(out_of_range)[:5])
                if len(out_of_range) > 5:
                    listing += f", ... ({len(out_of_range)} notes)"
                warnings.append(
                    f"{utterance.name}: note(s) {listing} outside the "
                    f"analysis F0 range ({low_hz:g}-{high_hz:g} Hz, "
                    f"MIDI {low_midi:.0f}-{high_midi:.0f}); F0 estimation "
                    f"is bounded by that range, so the note-relative pitch "
                    f"learned for those segments will be off")
        return warnings

    # -- driver ------------------------------------------------------------

    def train(self) -> HMSModel:
        config = self.config
        if not config.label_file:
            raise ValueError("training needs a label file")

        self.log("1/5  reading corpus")
        score = labels_module.load(config.label_file, time_unit=config.time_unit,
                                   frame_period=config.frame_period)
        for diagnostic in score.diagnostics:
            self.log(f"  ! {diagnostic}")
        for warning in self.note_range_warnings(score):
            self.log(f"  ! {warning}")
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
        # HMM re-estimation genuinely needs sequences from multiple utterances.
        # Keep their compact, normalized feature representation in a temporary
        # mmap cache; waveform, WORLD spectra, and analysis intermediates never
        # accumulate in the training process.
        with tempfile.TemporaryDirectory(prefix="hms-training-") as tmp:
            cache_dir = Path(tmp)
            (utterances, offset, scale, total_frames,
             vibrato_candidates) = self._prepare_training_cache(corpus, cache_dir)
            return self._train_cached(
                score, utterances, offset, scale, total_frames,
                vibrato_candidates)

    def _train_cached(
            self, score: labels_module.Score,
            utterances: Sequence[_CachedUtterance], offset: np.ndarray,
            scale: np.ndarray, total_frames: int,
            vibrato_candidates: Sequence[Tuple[float, float]]) -> HMSModel:
        config = self.config
        self.log("3/5  building features")
        # The cache already contains normalized static+dynamic features. Read
        # one utterance at a time, using only the first (static) block; the
        # target is the mean of within-utterance variances, not the variance
        # of pooled frames around a corpus-wide mean. No new corpus-sized
        # arrays or HMM training changes are needed.
        gv_stats = estimate_global_variance(
            (np.load(data.features_path, mmap_mode="r") for data in utterances
             if data.n_frames >= 2), self.spec.static_dim)
        phoneme_features, phoneme_voiced, durations = \
            self._collect_cached_phoneme_data(utterances)
        n_occurrences = sum(len(v) for v in phoneme_features.values())
        covered_frames = sum(int(sum(len(s) for s in v))
                             for v in phoneme_features.values())
        self.log(f"  {len(phoneme_features)} distinct phonemes, "
                 f"{n_occurrences} occurrences, {covered_frames}/{total_frames} "
                 f"frames covered by labels")

        self.log("4/5  training HMMs")
        hmms = self.train_hmms(phoneme_features, phoneme_voiced)
        backoff = self.build_backoff(phoneme_features, phoneme_voiced)
        if not hmms and not backoff:
            raise RuntimeError("no known phoneme data was available for dedicated "
                               "or class backoff training")

        # Optional sparse phoneme contexts (off by default). Their seeds are
        # offset past the class backoffs, so enabling them never perturbs the
        # dedicated/backoff models.
        contexts: Dict[str, LeftToRightHMM] = {}
        context_index: Dict[str, dict] = {}
        global_backoff: Optional[LeftToRightHMM] = None
        context_occurrences: Optional[Dict[str, Dict[str, object]]] = None
        if config.context_enabled:
            seed_base = config.seed + 10_000 * len(backoff)
            context_occurrences = self._collect_cached_context_data(utterances)
            selected = self.select_context_models(context_occurrences)
            self.log(f"  contexts: {len(context_occurrences)} observed, "
                     f"{len(selected)} selected "
                     f"(>= {config.context_min_frames} frames, "
                     f">= {config.context_min_occurrences} occurrences, "
                     f"cap {config.context_max_models})")
            contexts, context_index = self.train_contexts(
                context_occurrences, selected, seed_base)
            if not contexts:
                self.log(
                    "  ! context modelling is enabled but 0 context models "
                    "were created: no observed context met the support "
                    f"thresholds (>= {config.context_min_frames} pooled "
                    f"frames, >= {config.context_min_occurrences} "
                    f"occurrences, cap {config.context_max_models}); "
                    "synthesis will rely on the normal phone/backoff "
                    "hierarchy")
            if config.context_global_backoff:
                global_backoff = self.build_global_backoff(
                    phoneme_features, phoneme_voiced,
                    seed=seed_base + len(contexts))
            # Only the trained context keys are needed from here on, so the
            # pooled views go before the next (optional) pass maps the cache.
            del context_occurrences
            context_occurrences = None

        # The next stages reopen one utterance at a time, so discard all the
        # per-phone mmap views before anything else accumulates.
        del phoneme_features, phoneme_voiced, context_occurrences

        # Optional pitch-conditioned acoustic models (off by default).  They
        # form a tier *above* the ones trained so far -- a bucket is a subset of
        # its unit's frames -- and their seeds sit past the context block, so
        # enabling the feature never perturbs the other tiers.
        pitch_models: Dict[Tuple[str, int], LeftToRightHMM] = {}
        pitch_index: Dict[Tuple[str, int], dict] = {}
        pitch_conditioning = PitchConditioning(
            enabled=bool(config.pitch_conditioning_enabled),
            bin_size=config.pitch_conditioning_bin_size)
        if config.pitch_conditioning_enabled:
            pitch_seed_base = config.seed + 10_000 * (len(backoff) + 1)
            # Contexts that never earned their own HMM are not conditioned
            # either: their pitch buckets would hold even less data.
            conditioned_contexts = tuple(sorted(contexts)) or None
            pitch_occurrences = self._collect_cached_pitch_data(
                utterances, conditioned_contexts)
            selected_pitch = self.select_pitch_models(pitch_occurrences)
            self.log(f"  {pitch_conditioning.describe()}; "
                     f"{len(pitch_occurrences)} (unit, bin) buckets observed, "
                     f"{len(selected_pitch)} selected "
                     f"(phone buckets >= {config.min_phoneme_frames} frames, "
                     f"context buckets >= {config.context_min_frames} frames "
                     f"and >= {config.context_min_occurrences} occurrences)")
            pitch_models, pitch_index = self.train_pitch_models(
                pitch_occurrences, selected_pitch, pitch_seed_base)
            del pitch_occurrences
            if not pitch_models:
                self.log(
                    "  ! pitch conditioning is enabled but 0 pitch-conditioned "
                    "models were created: no (phone or trained context, pitch "
                    "bin) bucket met its support threshold, or no segment "
                    "carried a scored note; synthesis will rely on the normal "
                    "phone/context/backoff hierarchy")

        total_params = sum(h.n_free_params for h in hmms.values()) \
            + sum(h.n_free_params for h in backoff.values()) \
            + sum(h.n_free_params for h in contexts.values()) \
            + sum(h.n_free_params for h in pitch_models.values()) \
            + (global_backoff.n_free_params if global_backoff else 0)
        self.log(f"  {len(hmms)} HMMs, {len(backoff)} backoff model(s), "
                 f"{len(contexts)} context model(s), "
                 f"{len(pitch_models)} pitch-conditioned model(s), "
                 f"{total_params:,} total free params")

        self.log("5/5  duration, pitch and voicing models")
        duration_model = self.build_duration_model(durations)
        pitch_model = self._build_pitch_model_from_cache(
            utterances, hmms, vibrato_candidates)

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
            contexts=contexts,
            context_index=context_index,
            global_backoff=global_backoff,
            pitch_models=pitch_models,
            pitch_index=pitch_index,
            pitch_conditioning=pitch_conditioning,
            gv_stats=gv_stats,
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
