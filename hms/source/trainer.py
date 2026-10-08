"""Training path for the Phase-2 context/F0-conditioned source HMM/GMM.

The trainer consumes Phase-1 ``SourceSequence`` objects, aligns their existing
PCA unit coefficients to HMS's analysis-frame grid, adds the same static/delta
features used by the acoustic HMMs, and trains the shared left-to-right
HMM/GMM machinery.  It never reimplements source analysis or PCA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hms.core.context import (KIND_LEFT, KIND_RIGHT, context_keys,
                              context_wildcard)
from hms.core.features import FeatureSpec, add_dynamic_features, hz_to_semitone
from hms.core.hmm import LeftToRightHMM
from hms.core.labels import Utterance, segment_boundaries
from hms.core.phonemes import PhonemeSet
from hms.source.base import SourceModel, SourceSequence
from hms.source.hmm import (SourceF0Regressor,
                            SourceHMMModel, _true_runs, _validate_f0_track,
                            align_source_coefficients, f0_condition_features)
from hms.source.pca import DEFAULT_N_COMPONENTS, SourcePCA


@dataclass
class SourceTrainingConfig:
    """CPU-friendly source-model training options.

    PCA dimensionality is configurable and is only used when the caller does
    not supply an already-fitted Phase-1 ``SourcePCA``.  GMM state/component
    budgets follow the existing ``PhonemeSet`` definitions.
    """

    pca_components: int = DEFAULT_N_COMPONENTS
    covariance_type: str = "diag"
    training_method: str = "viterbi"
    n_iterations: int = 5
    var_floor_ratio: float = 1e-3
    seed: int = 0
    allow_skip: bool = False
    min_phone_frames: int = 20
    normalize_coefficients: bool = True

    # Sparse phone-neighbour HMM tier, matching hms.core.context.
    context_enabled: bool = False
    context_min_frames: int = 100
    context_min_occurrences: int = 3
    context_max_models: int = 64
    context_partial: bool = True
    global_backoff: bool = False

    # None inherits the corresponding existing HMS FeatureSpec convention.
    use_delta: Optional[bool] = None
    use_delta2: Optional[bool] = None
    delta_window: Optional[int] = None

    # Regularization for the per-component F0 -> source linear regressions.
    f0_regression_ridge: float = 1e-3
    f0_regression_variance_floor: float = 1e-5

    # Used only for pitch-cycle geometry at generation. F0 conditioning itself
    # remains unclamped, matching HMS's explicit-F0 convention.
    source_f0_floor: float = 50.0
    source_f0_ceil: float = 2000.0

    def __post_init__(self) -> None:
        self.pca_components = int(self.pca_components)
        self.n_iterations = int(self.n_iterations)
        self.min_phone_frames = int(self.min_phone_frames)
        self.context_min_frames = int(self.context_min_frames)
        self.context_min_occurrences = int(self.context_min_occurrences)
        self.context_max_models = int(self.context_max_models)
        self.covariance_type = str(self.covariance_type)
        self.training_method = str(self.training_method)
        if self.pca_components < 1:
            raise ValueError("pca_components must be positive")
        if self.covariance_type not in ("diag", "tied"):
            raise ValueError("covariance_type must be 'diag' or 'tied'")
        if self.training_method not in ("viterbi", "baum_welch"):
            raise ValueError("training_method must be 'viterbi' or 'baum_welch'")
        if self.n_iterations < 1 or self.min_phone_frames < 1:
            raise ValueError("n_iterations and min_phone_frames must be positive")
        if self.context_min_frames < 1 or self.context_min_occurrences < 1 \
                or self.context_max_models < 0:
            raise ValueError("context support thresholds must be positive and "
                             "context_max_models non-negative")
        if self.use_delta2 and self.use_delta is False:
            raise ValueError("use_delta2 requires use_delta")
        if self.delta_window is not None and int(self.delta_window) < 1:
            raise ValueError("delta_window must be at least 1")
        self.delta_window = (None if self.delta_window is None
                             else int(self.delta_window))
        numeric = (self.var_floor_ratio, self.f0_regression_ridge,
                   self.f0_regression_variance_floor, self.source_f0_floor,
                   self.source_f0_ceil)
        try:
            if not all(np.isfinite(value) for value in numeric):
                raise ValueError("source training parameters must be finite")
        except TypeError as exc:
            raise ValueError("source training parameters must be numeric") from exc
        if self.var_floor_ratio < 0:
            raise ValueError("var_floor_ratio must be non-negative")
        if self.f0_regression_ridge < 0 \
                or self.f0_regression_variance_floor <= 0:
            raise ValueError("F0 regression regularization must be non-negative "
                             "with a positive variance floor")
        if self.source_f0_floor <= 0 \
                or self.source_f0_ceil <= self.source_f0_floor:
            raise ValueError("source F0 geometry must satisfy 0 < floor < ceiling")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SourceTrainingExample:
    """One Phase-1 source analysis aligned with one labelled utterance.

    ``f0_hz`` is the exact HMS-frame-grid trajectory to condition on.  If it is
    omitted, the track already stored in ``source.f0`` is used; that may be the
    Phase-1 backend's estimate when no external F0 was supplied during
    analysis.  An explicit track always wins and is never resized.
    """

    utterance: Utterance
    source: SourceSequence
    f0_hz: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.f0_hz is None:
            self.f0_hz = np.asarray(self.source.f0, dtype=np.float64).copy()
        else:
            self.f0_hz = _validate_f0_track(self.f0_hz, self.source.n_frames)

    @classmethod
    def from_audio(cls, utterance: Utterance, audio: np.ndarray,
                   analyzer: SourceModel, fs: Optional[int] = None,
                   frame_period: Optional[float] = None,
                   f0_hz: Optional[np.ndarray] = None,
                   **analysis_options) -> "SourceTrainingExample":
        """Run the existing Phase-1 analyzer with the supplied geometry/F0.

        This is a convenience wrapper only: source extraction remains entirely
        in the Phase-1 ``SourceModel.analyze`` implementation.
        """
        signal = np.asarray(audio, dtype=np.float64).reshape(-1)
        sample_rate = int(fs or analyzer.default_fs)
        period = float(frame_period or analyzer.default_frame_period)
        if f0_hz is not None:
            # Validate values up front, but let the Phase-1 analyzer define its
            # own frame count.  The exact-length check below catches analyzers
            # (such as the frame-residual backend) that would otherwise clip an
            # external track instead of preserving it.
            _validate_f0_track(f0_hz)
        source = analyzer.analyze(signal, fs=sample_rate,
                                  frame_period=period, f0=f0_hz,
                                  **analysis_options)
        supplied = source.f0 if f0_hz is None else _validate_f0_track(
            f0_hz, source.n_frames)
        return cls(utterance=utterance, source=source, f0_hz=supplied)


@dataclass
class _SequenceBucket:
    observations: List[np.ndarray]
    conditions: List[np.ndarray]
    voiced: List[np.ndarray]

    @classmethod
    def empty(cls) -> "_SequenceBucket":
        return cls([], [], [])

    def append(self, observations: np.ndarray, conditions: np.ndarray,
               voiced: np.ndarray) -> None:
        if len(observations):
            self.observations.append(observations)
            self.conditions.append(conditions)
            self.voiced.append(voiced)

    @property
    def frames(self) -> int:
        return int(sum(len(sequence) for sequence in self.observations))


@dataclass
class _PreparedExample:
    example: SourceTrainingExample
    coefficients: np.ndarray
    valid: np.ndarray
    f0: np.ndarray
    spans: List[Tuple[str, int, int]]
    conditions: Optional[np.ndarray] = None
    observations: Optional[np.ndarray] = None
    voiced: Optional[np.ndarray] = None


class SourceTrainer:
    """Train a context/F0-conditioned HMM/GMM over existing PCA coefficients."""

    def __init__(self, spec: Optional[FeatureSpec] = None,
                 phoneme_set: Optional[PhonemeSet] = None,
                 config: Optional[SourceTrainingConfig] = None,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.spec = spec or FeatureSpec()
        self.phoneme_set = phoneme_set or PhonemeSet.default()
        self.config = config or SourceTrainingConfig()
        self.log = log or (lambda _message: None)

    def fit(self, examples: Iterable[SourceTrainingExample],
            pca: Optional[SourcePCA] = None,
            name: str = "source") -> SourceHMMModel:
        """Fit and return a standalone Phase-2 source model.

        When ``pca`` is omitted, the existing Phase-1 ``SourcePCA.fit`` is
        applied to the source vectors in these examples.  Supplying ``pca``
        reuses a basis trained/loaded by the Phase-1 pipeline unchanged.
        """
        examples = list(examples)
        if not examples:
            raise ValueError("source training needs at least one example")
        self._validate_examples(examples)
        source_backend = examples[0].source.backend
        if source_backend not in ("voice", "residual"):
            raise ValueError("source training currently supports the Phase-1 "
                             "'voice' and 'residual' backends")

        if pca is None:
            vectors = [example.source.excitation for example in examples
                       if example.source.n_units]
            if not vectors:
                raise ValueError("Phase-1 examples contain no source units to fit PCA")
            source_vectors = np.concatenate(vectors, axis=0)
            pca = SourcePCA.fit(source_vectors,
                                n_components=self.config.pca_components)
        if pca.cycle_length != examples[0].source.cycle_length:
            raise ValueError("Phase-1 PCA cycle_length does not match source units")

        self.log("1/4  encoding Phase-1 source units and aligning to HMS frames")
        prepared: List[_PreparedExample] = []
        for example in examples:
            unit_coefficients = pca.encode(example.source.excitation)
            frame_coefficients, valid = align_source_coefficients(
                example.source, unit_coefficients, example.f0_hz,
                voiced_threshold=self.spec.voiced_threshold)
            spans = [(self.phoneme_set.canonical(phone), int(lo), int(hi))
                     for phone, lo, hi, _note in segment_boundaries(
                         example.utterance, self.spec.frame_period,
                         n_frames=example.source.n_frames) if hi > lo]
            labelled = np.zeros(example.source.n_frames, dtype=bool)
            for _phone, lo, hi in spans:
                labelled[lo:hi] = True
            valid &= labelled
            prepared.append(_PreparedExample(
                example=example, coefficients=frame_coefficients, valid=valid,
                f0=np.asarray(example.f0_hz, dtype=np.float64), spans=spans))

        valid_frames = int(sum(int(item.valid.sum()) for item in prepared))
        if valid_frames == 0:
            raise ValueError("no labelled Phase-1 source units align to training frames")
        self.log(f"  aligned {valid_frames} source-bearing frame(s)")

        use_delta = (self.spec.use_delta if self.config.use_delta is None
                     else bool(self.config.use_delta))
        use_delta2 = (self.spec.use_delta2 if self.config.use_delta2 is None
                      else bool(self.config.use_delta2))
        delta_window = int(self.config.delta_window or self.spec.delta_window)
        if use_delta2 and not use_delta:
            raise ValueError("use_delta2 requires use_delta")
        f0_mean, f0_scale, f0_delta_scale = self._fit_f0_scaling(
            prepared, delta_window)
        coefficient_offset, coefficient_scale = self._fit_coefficient_scaling(
            prepared, pca.n_components)

        for item in prepared:
            item.conditions = f0_condition_features(
                item.f0, self.spec, f0_mean, f0_scale, f0_delta_scale,
                window=delta_window)
            item.voiced = item.f0 > self.spec.voiced_threshold
            normalized = (item.coefficients - coefficient_offset) * coefficient_scale
            observations = np.zeros(
                (len(normalized), pca.n_components
                 * (1 + int(use_delta) + int(use_delta2))), dtype=np.float64)
            for lo, hi in _true_runs(item.valid):
                observations[lo:hi] = add_dynamic_features(
                    normalized[lo:hi], use_delta=use_delta,
                    use_delta2=use_delta2, window=delta_window)
            item.observations = observations

        self.log("2/4  collecting phone and sparse context sequences")
        phone_data, context_data = self._collect_sequences(prepared)
        if not phone_data:
            raise ValueError("no labelled source coefficient sequences were collected")

        self.log("3/4  training source HMM/GMM units and F0 regressions")
        hmms: Dict[str, LeftToRightHMM] = {}
        conditioners: Dict[Tuple[str, str], List[SourceF0Regressor]] = {}
        for index, phone in enumerate(sorted(phone_data)):
            bucket = phone_data[phone]
            if bucket.frames < self.config.min_phone_frames \
                    or self.phoneme_set.resolve(phone) is None:
                continue
            hmm, regressor = self._train_hmm(
                phone, bucket, seed=self.config.seed + 1000 + index * 31)
            hmms[phone] = hmm
            conditioners[("phone", phone)] = regressor

        backoff_data = self._collect_class_backoff(phone_data)
        backoff: Dict[str, LeftToRightHMM] = {}
        for index, klass in enumerate(sorted(backoff_data)):
            bucket = backoff_data[klass]
            if not bucket.frames:
                continue
            hmm, regressor = self._train_hmm(
                klass, bucket, seed=self.config.seed + 10_000 + index * 101,
                class_backoff=True)
            backoff[klass] = hmm
            conditioners[("backoff", klass)] = regressor

        contexts: Dict[str, LeftToRightHMM] = {}
        context_index: Dict[str, dict] = {}
        selected_contexts = self._select_contexts(context_data)
        for index, key in enumerate(selected_contexts):
            entry = context_data[key]
            bucket = entry["bucket"]
            assert isinstance(bucket, _SequenceBucket)
            phone = str(entry["curr"])
            hmm, regressor = self._train_hmm(
                phone, bucket, seed=self.config.seed + 20_000 + index * 37)
            contexts[key] = hmm
            conditioners[("context", key)] = regressor
            context_index[key] = {
                "kind": entry["kind"],
                "pre": entry["pre"],
                "curr": entry["curr"],
                "post": entry["post"],
                "frames": int(bucket.frames),
                "occurrences": int(entry["occurrences"]),
                "n_states": int(hmm.n_states),
                "n_components": int(hmm.states[0].gmm.n_components),
                "covariance": hmm.covariance_type,
                "allow_skip": bool(hmm.allow_skip),
            }

        global_backoff = None
        if self.config.global_backoff:
            pooled = _SequenceBucket.empty()
            for phone in sorted(phone_data):
                self._extend_bucket(pooled, phone_data[phone])
            if pooled.frames:
                global_backoff, regressors = self._train_hmm(
                    "global", pooled, seed=self.config.seed + 30_000,
                    global_backoff=True)
                conditioners[("global", "__global__")] = regressors

        if not hmms and not backoff and global_backoff is None and not contexts:
            raise ValueError("source training did not produce any usable HMM/GMM")
        if source_backend == "residual":
            unit_frames = self._infer_unit_frames(examples)
        else:
            unit_frames = 1
        training = self.config.to_dict()
        training.update({
            "examples": len(examples),
            "source_frames": valid_frames,
            "source_units": int(sum(example.source.n_units for example in examples)),
            "pca_components": pca.n_components,
            "source_backend": source_backend,
        })
        self.log(f"  {len(hmms)} phone, {len(contexts)} context, "
                 f"{len(backoff)} class-backoff HMM(s)")
        self.log("4/4  source model ready")
        return SourceHMMModel(
            pca=pca, spec=self.spec, phoneme_set=self.phoneme_set,
            hmms=hmms, contexts=contexts, context_index=context_index,
            backoff=backoff, global_backoff=global_backoff,
            conditioners=conditioners,
            coefficient_offset=coefficient_offset,
            coefficient_scale=coefficient_scale,
            f0_mean=f0_mean, f0_scale=f0_scale,
            f0_delta_scale=f0_delta_scale,
            use_delta=use_delta, use_delta2=use_delta2,
            delta_window=delta_window,
            source_backend=source_backend, unit_frames=unit_frames,
            source_f0_floor=self.config.source_f0_floor,
            source_f0_ceil=self.config.source_f0_ceil,
            name=name, training=training)

    def train(self, examples: Iterable[SourceTrainingExample],
              pca: Optional[SourcePCA] = None,
              name: str = "source") -> SourceHMMModel:
        """Alias for :meth:`fit`, for users who prefer a trainer verb."""
        return self.fit(examples, pca=pca, name=name)

    def _validate_examples(self, examples: Sequence[SourceTrainingExample]) -> None:
        first = examples[0].source
        for index, example in enumerate(examples):
            source = example.source
            if source.backend != first.backend:
                raise ValueError("one source HMM cannot mix Phase-1 backend unit grids")
            if source.cycle_length != first.cycle_length:
                raise ValueError("all Phase-1 examples must share a cycle_length")
            if source.fs != self.spec.fs \
                    or not np.isclose(source.frame_period, self.spec.frame_period):
                raise ValueError(
                    f"example {index} uses {source.fs} Hz / {source.frame_period:g} ms, "
                    f"but the source model frame grid is {self.spec.fs} Hz / "
                    f"{self.spec.frame_period:g} ms; HMS does not resample implicitly")
            f0 = _validate_f0_track(example.f0_hz, source.n_frames)
            if source.excitation.size and not np.isfinite(source.excitation).all():
                raise ValueError(f"example {index} contains non-finite source vectors")
            # Store a sanitized, exact-length copy (not an interpolated track).
            example.f0_hz = f0

    def _fit_f0_scaling(self, examples: Sequence[_PreparedExample],
                        delta_window: int) -> Tuple[float, float, float]:
        values = []
        for item in examples:
            voiced = (item.f0 > self.spec.voiced_threshold) & item.valid
            if voiced.any():
                values.append(hz_to_semitone(item.f0[voiced], self.spec.f0_ref_hz))
        if not values:
            return 0.0, 1.0, 1.0
        pitch = np.concatenate(values)
        mean = float(pitch.mean())
        scale = float(max(pitch.std(), 1.0))
        delta_values = []
        for item in examples:
            features = f0_condition_features(item.f0, self.spec, mean, scale, 1.0,
                                             window=delta_window)
            mask = item.valid & (item.f0 > self.spec.voiced_threshold)
            if mask.any():
                delta_values.append(features[mask, 1])
        delta_scale = (float(max(np.concatenate(delta_values).std(), 1e-3))
                       if delta_values else 1.0)
        return mean, scale, delta_scale

    def _fit_coefficient_scaling(
            self, examples: Sequence[_PreparedExample], dimension: int
            ) -> Tuple[np.ndarray, np.ndarray]:
        chunks = [item.coefficients[item.valid] for item in examples if item.valid.any()]
        if not chunks:
            return np.zeros(dimension), np.ones(dimension)
        all_coefficients = np.concatenate(chunks, axis=0)
        if not self.config.normalize_coefficients:
            return np.zeros(dimension), np.ones(dimension)
        offset = all_coefficients.mean(axis=0)
        variance = np.maximum(all_coefficients.var(axis=0), 1e-6)
        scale = np.minimum(1.0 / np.sqrt(variance), 100.0)
        return offset, scale

    def _collect_sequences(
            self, examples: Sequence[_PreparedExample]
            ) -> Tuple[Dict[str, _SequenceBucket], Dict[str, dict]]:
        phone_data: Dict[str, _SequenceBucket] = {}
        context_data: Dict[str, dict] = {}
        wildcard = context_wildcard(self.phoneme_set.phonemes)
        silence = self.phoneme_set.silence
        for item in examples:
            assert item.observations is not None
            assert item.conditions is not None
            assert item.voiced is not None
            spans = [(phone, lo, hi) for phone, lo, hi in item.spans
                     if hi > lo]
            occurrence_chunks: List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
            for phone, lo, hi in spans:
                chunks = []
                for run_lo, run_hi in _true_runs(item.valid[lo:hi]):
                    start, stop = lo + run_lo, lo + run_hi
                    observations = item.observations[start:stop]
                    conditions = item.conditions[start:stop]
                    voiced = item.voiced[start:stop]
                    bucket = phone_data.setdefault(phone, _SequenceBucket.empty())
                    bucket.append(observations, conditions, voiced)
                    chunks.append((observations, conditions, voiced))
                occurrence_chunks.append(chunks)

            if not self.config.context_enabled:
                continue
            for index, ((curr, _lo, _hi), chunks) in enumerate(
                    zip(spans, occurrence_chunks)):
                if not chunks:
                    continue
                pre = spans[index - 1][0] if index > 0 else silence
                post = spans[index + 1][0] if index + 1 < len(spans) else silence
                for key, kind in context_keys(
                        pre, curr, post, wildcard,
                        partial=self.config.context_partial):
                    entry = context_data.get(key)
                    if entry is None:
                        entry = {
                            "kind": kind,
                            "pre": pre if kind != KIND_RIGHT else None,
                            "curr": curr,
                            "post": post if kind != KIND_LEFT else None,
                            "bucket": _SequenceBucket.empty(),
                            "occurrences": 0,
                        }
                        context_data[key] = entry
                    bucket = entry["bucket"]
                    assert isinstance(bucket, _SequenceBucket)
                    for observations, conditions, voiced in chunks:
                        bucket.append(observations, conditions, voiced)
                    entry["occurrences"] += 1
        return phone_data, context_data

    def _collect_class_backoff(
            self, phone_data: Dict[str, _SequenceBucket]
            ) -> Dict[str, _SequenceBucket]:
        pooled: Dict[str, _SequenceBucket] = {}
        for phone, bucket in sorted(phone_data.items()):
            definition = self.phoneme_set.resolve(phone)
            klass = definition.type if definition is not None else "unvoiced_consonant"
            target = pooled.setdefault(klass, _SequenceBucket.empty())
            self._extend_bucket(target, bucket)
        return pooled

    @staticmethod
    def _extend_bucket(destination: _SequenceBucket,
                       source: _SequenceBucket) -> None:
        destination.observations.extend(source.observations)
        destination.conditions.extend(source.conditions)
        destination.voiced.extend(source.voiced)

    def _select_contexts(self, context_data: Dict[str, dict]) -> List[str]:
        candidates = []
        for key, entry in context_data.items():
            bucket = entry["bucket"]
            assert isinstance(bucket, _SequenceBucket)
            occurrences = int(entry["occurrences"])
            if bucket.frames < self.config.context_min_frames \
                    or occurrences < self.config.context_min_occurrences:
                continue
            candidates.append((bucket.frames, occurrences, key))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [key for _frames, _occurrences, key in
                candidates[:self.config.context_max_models]]

    def _train_hmm(self, phone_or_class: str, bucket: _SequenceBucket,
                   seed: int, class_backoff: bool = False,
                   global_backoff: bool = False
                   ) -> Tuple[LeftToRightHMM, List[SourceF0Regressor]]:
        if not bucket.observations:
            raise ValueError(f"cannot train an empty source HMM for {phone_or_class!r}")
        definition = None if class_backoff or global_backoff \
            else self.phoneme_set.resolve(phone_or_class)
        if global_backoff:
            requested_states, n_components = 3, 1
        elif definition is not None:
            requested_states, n_components = definition.n_states, definition.n_components
        else:
            klass = phone_or_class if class_backoff else "unvoiced_consonant"
            defaults = self.phoneme_set.defaults.get(klass, {})
            requested_states = int(defaults.get("n_states", 2))
            n_components = int(defaults.get("n_components", 1))
        total_frames = bucket.frames
        min_sequence_frames = min(len(sequence) for sequence in bucket.observations)
        n_states = min(int(requested_states), max(1, total_frames // 2),
                       max(1, min_sequence_frames))
        hmm = LeftToRightHMM(
            n_states=n_states, allow_skip=self.config.allow_skip,
            covariance_type=self.config.covariance_type)
        hmm.train(
            bucket.observations, n_components=int(n_components),
            covariance_type=self.config.covariance_type,
            n_iterations=self.config.n_iterations,
            var_floor_ratio=self.config.var_floor_ratio,
            seed=int(seed), method=self.config.training_method,
            voiced=bucket.voiced)
        regressors = SourceF0Regressor.fit(
            hmm, bucket.observations, bucket.conditions,
            ridge=self.config.f0_regression_ridge,
            variance_floor=self.config.f0_regression_variance_floor)
        return hmm, regressors

    @staticmethod
    def _infer_unit_frames(examples: Sequence[SourceTrainingExample]) -> int:
        estimates = []
        for example in examples:
            source = example.source
            if source.n_units:
                ratios = source.periods.astype(np.float64) / float(source.hop)
                estimates.extend(np.maximum(1, np.rint(ratios)).astype(int).tolist())
        return max(1, int(round(float(np.median(estimates))))) if estimates else 1


__all__ = ["SourceTrainingConfig", "SourceTrainingExample", "SourceTrainer"]
