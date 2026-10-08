"""Frame-aligned, F0-conditioned source HMM/GMM prediction (Phase 2).

This layer predicts the existing :class:`SourcePCA` coefficients.  It does not
analyse audio or define another source representation: Phase 1 owns the source
vectors, their unit grids and the PCA decoder.  Linguistic context selects the
same sparse phone/triphone/diphone HMM units as ``hms.core.context``; each
unit's existing left-to-right HMM/GMM models coefficient static/delta streams.
A small Gaussian-mixture regression attached to each state conditions those
statistics on the caller's frame-level F0, and the existing banded MLPG solver
produces the smooth coefficient trajectory.

The model is deliberately separate from ``HMSModel`` and the synthesizer.  Its
standalone directory contains the trained source HMM/GMM, the Phase-1 PCA basis,
and the frame/pitch geometry needed to decode generated source units.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from hms.core.context import (GLOBAL_KEY, KIND_LEFT, KIND_RIGHT,
                              KIND_TRIPHONE, context_wildcard,
                              left_diphone_key, right_diphone_key,
                              triphone_key)
from hms.core.duration import DurationModel
from hms.core.features import (FeatureSpec, add_dynamic_features,
                               hz_to_semitone)
from hms.core.generation import mlpg, stack_streams
from hms.core.hmm import LeftToRightHMM
from hms.core.labels import Utterance, segment_boundaries
from hms.core.phonemes import PhonemeSet
from hms.source.base import SourceSequence
from hms.source.cycles import pick_epochs
from hms.source.pca import SourcePCA

SOURCE_MODEL_FORMAT = "hms.source.hmm"
SOURCE_MODEL_FORMAT_VERSION = 1
F0_CONDITION_DIM = 3  # log-F0, its HMS delta, explicit voiced flag
_SOURCE_YAML = "source.yaml"
_SOURCE_HMMS_NPZ = "source_hmms.npz"
_SOURCE_PCA_NPZ = "source_pca.npz"


@dataclass
class SourceF0Regressor:
    """GMM-gated linear regression of HMM/GMM outputs on explicit F0 features.

    The HMM state's existing diagonal GMM supplies the component weights,
    source-output distributions and component responsibilities during fitting.
    For each component, this compact regressor stores a diagonal Gaussian over
    the F0 condition and a weighted least-squares map from the condition to the
    source static/dynamic observation.  This lets a diagonal GMM express
    continuous pitch conditioning without changing HMS's shared GMM machinery
    or introducing full-covariance matrices.
    """

    input_means: np.ndarray            # (K, F)
    input_variances: np.ndarray       # (K, F)
    regression: np.ndarray            # (K, D, F + 1), intercept first
    residual_variances: np.ndarray    # (K, D)

    def __post_init__(self) -> None:
        self.input_means = np.asarray(self.input_means, dtype=np.float64)
        self.input_variances = np.asarray(self.input_variances, dtype=np.float64)
        self.regression = np.asarray(self.regression, dtype=np.float64)
        self.residual_variances = np.asarray(self.residual_variances,
                                             dtype=np.float64)
        if self.input_means.ndim != 2:
            raise ValueError("F0-regression input means must be (components, features)")
        k, n_input = self.input_means.shape
        if self.input_variances.shape != (k, n_input):
            raise ValueError("F0-regression input variances have the wrong shape")
        if self.regression.ndim != 3 or self.regression.shape[0] != k \
                or self.regression.shape[2] != n_input + 1:
            raise ValueError("F0-regression coefficients have the wrong shape")
        if self.residual_variances.shape != self.regression.shape[:2]:
            raise ValueError("F0-regression residual variances have the wrong shape")
        values = (self.input_means, self.input_variances, self.regression,
                  self.residual_variances)
        if any(value.size and not np.isfinite(value).all() for value in values):
            raise ValueError("F0-regression parameters must be finite")
        if (self.input_variances <= 0).any() or (self.residual_variances <= 0).any():
            raise ValueError("F0-regression variances must be positive")

    @classmethod
    def fit(cls, hmm: LeftToRightHMM,
            sequences: Sequence[np.ndarray],
            conditions: Sequence[np.ndarray],
            ridge: float = 1e-3,
            variance_floor: float = 1e-5) -> List["SourceF0Regressor"]:
        """Fit one component-gated regression for each HMM state.

        ``sequences`` are the same normalized coefficient static/delta
        observations used to fit ``hmm``.  ``conditions`` are aligned
        ``(frames, 3)`` arrays from :func:`f0_condition_features`.
        """
        if len(sequences) != len(conditions):
            raise ValueError("source observations and F0 conditions must align")
        ridge = float(ridge)
        variance_floor = float(variance_floor)
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and non-negative")
        if not np.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")

        state_y: List[List[np.ndarray]] = [[] for _ in range(hmm.n_states)]
        state_x: List[List[np.ndarray]] = [[] for _ in range(hmm.n_states)]
        global_x = []
        for observation, condition in zip(sequences, conditions):
            observation = np.atleast_2d(np.asarray(observation, dtype=np.float64))
            condition = np.atleast_2d(np.asarray(condition, dtype=np.float64))
            if len(observation) != len(condition):
                raise ValueError("source observations and F0 conditions differ in length")
            if len(observation) == 0:
                continue
            if not np.isfinite(observation).all() or not np.isfinite(condition).all():
                raise ValueError("source observations and F0 conditions must be finite")
            global_x.append(condition)
            path = hmm.segment(observation)
            for state in range(hmm.n_states):
                selected = path == state
                if selected.any():
                    state_y[state].append(observation[selected])
                    state_x[state].append(condition[selected])

        if global_x:
            all_x = np.concatenate(global_x, axis=0)
            global_mean = all_x.mean(axis=0)
            global_variance = np.maximum(all_x.var(axis=0), 1e-3)
        else:
            global_mean = np.zeros(F0_CONDITION_DIM)
            global_variance = np.ones(F0_CONDITION_DIM)

        fitted = []
        for state, hmm_state in enumerate(hmm.states):
            gmm = hmm_state.gmm
            y_chunks, x_chunks = state_y[state], state_x[state]
            if y_chunks:
                y = np.concatenate(y_chunks, axis=0)
                x = np.concatenate(x_chunks, axis=0)
                responsibilities = gmm.posterior(y)
            else:
                y = np.zeros((0, gmm.dim), dtype=np.float64)
                x = np.zeros((0, F0_CONDITION_DIM), dtype=np.float64)
                responsibilities = np.zeros((0, gmm.n_components), dtype=np.float64)

            input_means = np.repeat(global_mean[None, :], gmm.n_components, axis=0)
            input_variances = np.repeat(global_variance[None, :],
                                        gmm.n_components, axis=0)
            regression = np.zeros((gmm.n_components, gmm.dim,
                                   F0_CONDITION_DIM + 1), dtype=np.float64)
            residual_variances = np.empty((gmm.n_components, gmm.dim),
                                          dtype=np.float64)
            for component in range(gmm.n_components):
                regression[component, :, 0] = gmm.means[component]
                residual_variances[component] = np.maximum(
                    gmm.variances[component], variance_floor)
                if not len(y):
                    continue
                weights = responsibilities[:, component]
                mass = float(weights.sum())
                # Components with too little effective data retain their
                # ordinary GMM prediction; well-supported components learn a
                # regularized F0-to-source regression.
                if mass < 2.0:
                    continue

                normalized_weights = weights / mass
                mean_x = normalized_weights @ x
                centered_x = x - mean_x
                var_x = normalized_weights @ (centered_x ** 2)
                input_means[component] = mean_x
                input_variances[component] = np.maximum(var_x, 1e-3)

                design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
                gram = design.T @ (weights[:, None] * design)
                penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
                penalty[0, 0] = 0.0              # do not shrink the intercept
                rhs = design.T @ (weights[:, None] * y)
                try:
                    beta = np.linalg.solve(gram + penalty, rhs)
                except np.linalg.LinAlgError:     # defensive for a singular tiny fit
                    beta = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
                prediction = design @ beta
                residual = y - prediction
                fitted_variance = (weights[:, None] * residual ** 2).sum(axis=0) / mass
                regression[component] = beta.T
                residual_variances[component] = np.maximum(
                    np.maximum(fitted_variance, variance_floor),
                    0.02 * gmm.variances[component])

            fitted.append(cls(input_means, input_variances, regression,
                              residual_variances))
        return fitted

    @property
    def n_components(self) -> int:
        return int(self.input_means.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.regression.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.input_means.shape[1])

    def predict(self, gmm, conditions: np.ndarray,
                mixture: str = "dominant", batch_size: int = 8192
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Conditional output moments for frame-level F0 features."""
        conditions = np.atleast_2d(np.asarray(conditions, dtype=np.float64))
        if conditions.shape[1] != self.input_dim:
            raise ValueError(f"F0 conditions must have {self.input_dim} columns, "
                             f"got {conditions.shape}")
        if not np.isfinite(conditions).all():
            raise ValueError("F0 conditions must be finite")
        if mixture not in ("dominant", "mean"):
            raise ValueError("mixture must be 'dominant' or 'mean'")
        if gmm.n_components != self.n_components or gmm.dim != self.output_dim:
            raise ValueError("F0 regressor does not match its HMM-state GMM")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")

        n_frames = len(conditions)
        means = np.zeros((n_frames, self.output_dim), dtype=np.float64)
        variances = np.zeros_like(means)
        log_weights = np.log(np.maximum(gmm.weights, 1e-12))
        for lo in range(0, n_frames, int(batch_size)):
            hi = min(n_frames, lo + int(batch_size))
            x = conditions[lo:hi]
            delta = x[:, None, :] - self.input_means[None, :, :]
            log_gate = (log_weights[None, :]
                        - 0.5 * (np.log(2.0 * np.pi * self.input_variances)
                                 + delta ** 2 / self.input_variances).sum(axis=2))
            log_gate -= log_gate.max(axis=1, keepdims=True)
            gate = np.exp(log_gate)
            gate /= np.maximum(gate.sum(axis=1, keepdims=True), 1e-300)
            design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
            component_means = np.einsum("nf,kdf->nkd", design, self.regression,
                                        optimize=True)
            if mixture == "dominant":
                chosen = np.argmax(gate, axis=1)
                rows = np.arange(len(x))
                means[lo:hi] = component_means[rows, chosen]
                variances[lo:hi] = self.residual_variances[chosen]
            else:
                mean = np.einsum("nk,nkd->nd", gate, component_means,
                                 optimize=True)
                second = np.einsum(
                    "nk,nkd->nd", gate,
                    self.residual_variances[None, :, :] + component_means ** 2,
                    optimize=True)
                means[lo:hi] = mean
                variances[lo:hi] = np.maximum(second - mean ** 2, 1e-8)
        return means, np.maximum(variances, 1e-8)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "input_means": self.input_means,
            "input_variances": self.input_variances,
            "regression": self.regression,
            "residual_variances": self.residual_variances,
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray]) -> "SourceF0Regressor":
        required = ("input_means", "input_variances", "regression",
                    "residual_variances")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError("source F0 regressor is missing arrays: "
                             + ", ".join(missing))
        return cls(*(arrays[name] for name in required))


@dataclass
class SourcePrediction:
    """A Phase-2 prediction on the HMS frame grid plus decoded Phase-1 units."""

    frame_coefficients: np.ndarray      # (T, PCA dimensions), raw PCA scale
    f0_hz: np.ndarray                   # (T,), caller's authoritative track
    voiced: np.ndarray                  # (T,), HMS threshold convention
    active: np.ndarray                  # (T,), where the source model predicts
    state_ids: np.ndarray               # (T,), -1 where inactive
    phones: Tuple[str, ...]             # (T,)
    sequence: SourceSequence            # Phase-1 source-unit container/decoder input

    def __post_init__(self) -> None:
        self.frame_coefficients = np.atleast_2d(
            np.asarray(self.frame_coefficients, dtype=np.float64))
        self.f0_hz = np.asarray(self.f0_hz, dtype=np.float64).reshape(-1)
        self.voiced = np.asarray(self.voiced, dtype=bool).reshape(-1)
        self.active = np.asarray(self.active, dtype=bool).reshape(-1)
        self.state_ids = np.asarray(self.state_ids, dtype=np.int64).reshape(-1)
        self.phones = tuple(str(phone) for phone in self.phones)
        n_frames = len(self.f0_hz)
        if self.frame_coefficients.shape[0] != n_frames \
                or len(self.voiced) != n_frames or len(self.active) != n_frames \
                or len(self.state_ids) != n_frames or len(self.phones) != n_frames:
            raise ValueError("source prediction fields must share one frame grid")
        if not np.isfinite(self.frame_coefficients).all():
            raise ValueError("generated source coefficients must be finite")

    @property
    def coefficients(self) -> np.ndarray:
        """Alias for the frame-aligned PCA coefficient trajectory."""
        return self.frame_coefficients

    @property
    def source_vectors(self) -> np.ndarray:
        """Phase-1 PCA-decoded vectors, one row per generated source unit."""
        return self.sequence.excitation


class SourceHMMModel:
    """Singer-specific HMM/GMM predictor for a Phase-1 ``SourcePCA`` space.

    Phone context is handled by sparse HMM selection, not by appending an
    unbounded one-hot vector to the GMM observation.  The GMM dimension is
    therefore ``pca.n_components * number_of_dynamic_streams`` regardless of
    how many phone contexts the trainer retains.
    """

    def __init__(self, pca: SourcePCA, spec: FeatureSpec,
                 phoneme_set: PhonemeSet,
                 hmms: Optional[Dict[str, LeftToRightHMM]] = None,
                 contexts: Optional[Dict[str, LeftToRightHMM]] = None,
                 context_index: Optional[Dict[str, dict]] = None,
                 backoff: Optional[Dict[str, LeftToRightHMM]] = None,
                 global_backoff: Optional[LeftToRightHMM] = None,
                 conditioners: Optional[
                     Dict[Tuple[str, str], List[SourceF0Regressor]]] = None,
                 coefficient_offset: Optional[np.ndarray] = None,
                 coefficient_scale: Optional[np.ndarray] = None,
                 f0_mean: float = 0.0, f0_scale: float = 1.0,
                 f0_delta_scale: float = 1.0,
                 use_delta: bool = True, use_delta2: bool = False,
                 delta_window: Optional[int] = None,
                 source_backend: str = "voice", unit_frames: int = 1,
                 source_f0_floor: float = 50.0,
                 source_f0_ceil: float = 2000.0,
                 name: str = "source", training: Optional[Dict[str, object]] = None
                 ) -> None:
        if source_backend not in ("voice", "residual"):
            raise ValueError("source_backend must be 'voice' or 'residual'")
        if int(unit_frames) < 1:
            raise ValueError("unit_frames must be at least 1")
        try:
            numeric = (f0_mean, f0_scale, f0_delta_scale,
                       source_f0_floor, source_f0_ceil)
            if not all(np.isfinite(value) for value in numeric):
                raise ValueError("source-model pitch values must be finite")
        except TypeError as exc:
            raise ValueError("source-model pitch values must be numeric") from exc
        if f0_scale <= 0 or f0_delta_scale <= 0:
            raise ValueError("F0 conditioning scales must be positive")
        if source_f0_floor <= 0 or source_f0_ceil <= source_f0_floor:
            raise ValueError("source F0 range must satisfy 0 < floor < ceiling")

        self.pca = pca
        self.spec = spec
        self.phoneme_set = phoneme_set
        self.hmms = dict(hmms or {})
        self.contexts = dict(contexts or {})
        self.context_index = dict(context_index or {})
        self.backoff = dict(backoff or {})
        self.global_backoff = global_backoff
        self.conditioners = dict(conditioners or {})
        self.coefficient_offset = np.asarray(
            np.zeros(pca.n_components) if coefficient_offset is None
            else coefficient_offset, dtype=np.float64).reshape(-1)
        self.coefficient_scale = np.asarray(
            np.ones(pca.n_components) if coefficient_scale is None
            else coefficient_scale, dtype=np.float64).reshape(-1)
        if self.coefficient_offset.shape != (pca.n_components,) \
                or self.coefficient_scale.shape != (pca.n_components,):
            raise ValueError("source coefficient normalization does not match PCA")
        if not np.isfinite(self.coefficient_offset).all() \
                or not np.isfinite(self.coefficient_scale).all() \
                or (self.coefficient_scale <= 0).any():
            raise ValueError("source coefficient normalization must be finite "
                             "with positive scales")
        self.f0_mean = float(f0_mean)
        self.f0_scale = float(f0_scale)
        self.f0_delta_scale = float(f0_delta_scale)
        self.use_delta = bool(use_delta)
        self.use_delta2 = bool(use_delta2)
        self.delta_window = int(delta_window or spec.delta_window)
        if self.delta_window < 1:
            raise ValueError("delta_window must be at least 1")
        self.source_backend = str(source_backend)
        self.unit_frames = int(unit_frames)
        self.source_f0_floor = float(source_f0_floor)
        self.source_f0_ceil = float(source_f0_ceil)
        self.name = str(name)
        self.training = dict(training or {})
        self.stream_sizes = tuple(
            [pca.n_components] * (1 + int(self.use_delta) + int(self.use_delta2)))
        self.feature_dim = pca.n_components * len(self.stream_sizes)
        self.context_feature_dim = 3  # categorical (left, current, right) phones
        self.condition_feature_dim = F0_CONDITION_DIM

    @property
    def n_context_models(self) -> int:
        return len(self.contexts)

    @property
    def n_free_params(self) -> int:
        hmms = list(self.hmms.values()) + list(self.contexts.values()) \
            + list(self.backoff.values())
        if self.global_backoff is not None:
            hmms.append(self.global_backoff)
        hmm_parameters = sum(hmm.n_free_params for hmm in hmms)
        regression_parameters = sum(
            regressor.input_means.size + regressor.input_variances.size
            + regressor.regression.size + regressor.residual_variances.size
            for regressors in self.conditioners.values()
            for regressor in regressors)
        return int(hmm_parameters + regression_parameters)

    def resolve_unit(self, pre: str, curr: str, post: str
                     ) -> Tuple[Tuple[str, str], LeftToRightHMM, str]:
        """Resolve an occurrence with HMS's triphone/diphone/phone/class fallback."""
        canonical = self.phoneme_set.canonical
        pre_c, curr_c, post_c = canonical(pre), canonical(curr), canonical(post)
        if self.contexts:
            tri = triphone_key(pre_c, curr_c, post_c)
            if tri in self.contexts:
                return ("context", tri), self.contexts[tri], KIND_TRIPHONE
            wildcard = context_wildcard(self.phoneme_set.phonemes)
            left_key = left_diphone_key(pre_c, curr_c, wildcard)
            right_key = right_diphone_key(curr_c, post_c, wildcard)
            left_support = self._context_support(left_key)
            right_support = self._context_support(right_key)
            if left_support is not None \
                    and (right_support is None or left_support >= right_support):
                return ("context", left_key), self.contexts[left_key], KIND_LEFT
            if right_support is not None:
                return ("context", right_key), self.contexts[right_key], KIND_RIGHT
        if curr_c in self.hmms:
            return ("phone", curr_c), self.hmms[curr_c], "phone"
        definition = self.phoneme_set.resolve(curr_c)
        klass = definition.type if definition is not None else "unvoiced_consonant"
        for backoff_key in (klass, "unvoiced_consonant"):
            if backoff_key in self.backoff:
                return ("backoff", backoff_key), self.backoff[backoff_key], "class"
        if self.global_backoff is not None:
            return ("global", GLOBAL_KEY), self.global_backoff, "global"
        raise KeyError(f"no source HMM for phoneme {curr_c!r} and no class/global "
                       "backoff is available")

    def _context_support(self, key: str) -> Optional[int]:
        if key not in self.contexts:
            return None
        return int((self.context_index.get(key) or {}).get("frames", 0))

    def generate(self, utterance: Utterance, f0_hz: np.ndarray,
                 n_samples: Optional[int] = None, mixture: str = "dominant",
                 variance_scale: float = 1.0, seed: int = 0
                 ) -> SourcePrediction:
        """Predict source PCA coefficients from labels/context and explicit F0.

        ``f0_hz`` is mandatory and must have one value per HMS analysis frame.
        It is never estimated, smoothed, interpolated, clamped for conditioning,
        or silently resized. Zero and values at/below ``FeatureSpec``'s voicing
        threshold use the established unvoiced convention. ``seed`` is kept in
        the signature for parity with generation APIs; prediction itself is
        deterministic and does not sample.

        The returned ``frame_coefficients`` are on the fixed HMS frame grid.
        ``sequence`` is the Phase-1 ``SourceSequence`` filled on the backend's
        native source-unit grid (pitch cycles for ``voice``, frame units for
        ``residual``), with its vectors decoded by the saved ``SourcePCA``.
        """
        del seed  # no sampling in deterministic maximum-likelihood generation
        f0 = _validate_f0_track(f0_hz)
        n_frames = len(f0)
        if mixture not in ("dominant", "mean"):
            raise ValueError("mixture must be 'dominant' or 'mean'")
        if not np.isfinite(variance_scale) or variance_scale <= 0:
            raise ValueError("variance_scale must be finite and positive")
        hop = max(1, int(round(self.spec.fs * self.spec.frame_period / 1000.0)))
        if n_samples is None:
            n_samples = n_frames * hop
        n_samples = int(n_samples)
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if n_frames and n_samples <= 0:
            raise ValueError("n_samples must be positive for a non-empty trajectory")
        if n_samples > n_frames * hop:
            raise ValueError("n_samples cannot exceed the supplied F0 frame coverage")

        label_spans = _frame_spans(utterance, self.spec.frame_period, n_frames,
                                   self.phoneme_set)
        phones = np.full(n_frames, self.phoneme_set.silence, dtype=object)
        for _index, phone, lo, hi in label_spans:
            phones[lo:hi] = phone
        voiced = f0 > self.spec.voiced_threshold
        active = phones != self.phoneme_set.silence
        if self.source_backend == "voice":
            # Phase 1's voice backend has voiced cycles only.  Do not fabricate
            # cycle coefficients for unvoiced frames; the later renderer owns
            # their noise source, just as it did before Phase 2.
            active &= voiced

        conditions = f0_condition_features(
            f0, self.spec, self.f0_mean, self.f0_scale, self.f0_delta_scale,
            window=self.delta_window)
        state_ids = np.full(n_frames, -1, dtype=np.int64)
        frame_units: List[Optional[Tuple[str, str]]] = [None] * n_frames
        resolved: Dict[Tuple[str, str], LeftToRightHMM] = {}

        # The label/context sequence is segment-based, as it is in HMS.  Gaps
        # stay silent; neighboring phones are the actual labelled segments (the
        # same convention used by Trainer._collect_cached_context_data).
        for position, phone, lo, hi in label_spans:
            if hi <= lo or phone == self.phoneme_set.silence:
                continue
            pre = label_spans[position - 1][1] if position > 0 \
                else self.phoneme_set.silence
            post = label_spans[position + 1][1] \
                if position + 1 < len(label_spans) else self.phoneme_set.silence
            unit_id, hmm, _tier = self.resolve_unit(pre, phone, post)
            resolved[unit_id] = hmm
            for run_lo, run_hi in _true_runs(active[lo:hi]):
                start, stop = lo + run_lo, lo + run_hi
                counts = DurationModel.allocate(stop - start,
                                                hmm.duration_proportions())
                cursor = start
                for state, count in enumerate(counts):
                    count = int(count)
                    if count <= 0:
                        continue
                    end = min(stop, cursor + count)
                    state_ids[cursor:end] = state
                    frame_units[cursor:end] = [unit_id] * (end - cursor)
                    cursor = end

        observation_dim = self.feature_dim
        frame_means = np.zeros((n_frames, observation_dim), dtype=np.float64)
        frame_variances = np.ones_like(frame_means)
        grouped: Dict[Tuple[Tuple[str, str], int], List[int]] = {}
        for frame, unit_id in enumerate(frame_units):
            if unit_id is not None:
                grouped.setdefault((unit_id, int(state_ids[frame])), []).append(frame)
        for (unit_id, state), frames in grouped.items():
            hmm = resolved[unit_id]
            regressors = self.conditioners.get(unit_id)
            if regressors is None or state >= len(regressors):
                raise ValueError(f"source HMM {unit_id!r} has no F0 regressor "
                                 f"for state {state}")
            indices = np.asarray(frames, dtype=np.int64)
            mean, variance = regressors[state].predict(
                hmm.states[state].gmm, conditions[indices], mixture=mixture)
            frame_means[indices] = mean
            frame_variances[indices] = variance

        normalized = np.zeros((n_frames, self.pca.n_components), dtype=np.float64)
        # MLPG is solved independently on contiguous active runs: no derivative
        # window crosses an unvoiced gap or an unlabelled/silent span.
        for lo, hi in _true_runs(active):
            stacked_means = stack_streams(frame_means[lo:hi], self.stream_sizes)
            stacked_vars = stack_streams(frame_variances[lo:hi], self.stream_sizes)
            normalized[lo:hi] = mlpg(
                stacked_means, stacked_vars, self.stream_sizes,
                window=self.delta_window, variance_scale=float(variance_scale),
                smooth=True)
        coefficients = normalized / self.coefficient_scale + self.coefficient_offset
        coefficients[~active] = 0.0
        sequence = self._decode_to_sequence(
            coefficients, f0, voiced, active, n_samples)
        return SourcePrediction(
            frame_coefficients=coefficients, f0_hz=f0, voiced=voiced,
            active=active, state_ids=state_ids,
            phones=tuple(str(phone) for phone in phones), sequence=sequence)

    def _decode_to_sequence(self, frame_coefficients: np.ndarray, f0: np.ndarray,
                            voiced: np.ndarray, active: np.ndarray,
                            n_samples: int) -> SourceSequence:
        """Sample a frame trajectory onto the Phase-1 backend's unit grid."""
        hop = max(1, int(round(self.spec.fs * self.spec.frame_period / 1000.0)))
        if self.source_backend == "voice":
            effective_f0 = np.where(active, f0, 0.0)
            epochs, runs = pick_epochs(
                effective_f0, hop, n_samples, self.spec.fs,
                f0_floor=self.source_f0_floor, f0_ceil=self.source_f0_ceil,
                refine=False)
            # Match Phase 1 extraction: a cycle exists only when both boundary
            # epochs belong to this same voiced run.
            keep = np.flatnonzero(runs[:-1] == runs[1:]) if len(runs) > 1 \
                else np.zeros(0, dtype=np.int64)
            unit_epochs = epochs[keep]
            unit_periods = (epochs[keep + 1] - epochs[keep]).astype(np.int64) \
                if len(keep) else np.zeros(0, dtype=np.int64)
            centers = (unit_epochs.astype(np.float64)
                       + 0.5 * unit_periods.astype(np.float64)) / hop
            unit_coefficients = _sample_trajectory(frame_coefficients, centers)
        else:
            step = self.unit_frames * hop
            unit_epochs_list: List[int] = []
            unit_periods_list: List[int] = []
            unit_coefficients_list: List[np.ndarray] = []
            # Start a residual unit grid at each active run so a unit cannot
            # straddle a labelled-silence or unvoiced-source gap. The final unit
            # of a run may be shorter, as it is in Phase-1 frame residuals.
            for run_lo, run_hi in _true_runs(active):
                run_start = run_lo * hop
                run_stop = min(run_hi * hop, n_samples)
                for start in range(run_start, run_stop, step):
                    stop = min(start + step, run_stop)
                    average = _average_active_frames(
                        frame_coefficients, active, start, stop, hop, n_samples)
                    if average is None:
                        continue
                    unit_epochs_list.append(start)
                    unit_periods_list.append(stop - start)
                    unit_coefficients_list.append(average)
            unit_epochs = np.asarray(unit_epochs_list, dtype=np.int64)
            unit_periods = np.asarray(unit_periods_list, dtype=np.int64)
            unit_coefficients = (np.asarray(unit_coefficients_list, dtype=np.float64)
                                 .reshape(-1, self.pca.n_components))

        vectors = self.pca.decode(unit_coefficients)
        gains = np.ones(len(unit_epochs), dtype=np.float64)
        noise = np.where(voiced, 0.0, 1.0)
        return SourceSequence(
            f0=f0, voiced=voiced, noise_level=noise, excitation=vectors,
            gains=gains, epochs=unit_epochs, periods=unit_periods,
            cycle_length=self.pca.cycle_length, fs=self.spec.fs,
            frame_period=self.spec.frame_period, n_samples=n_samples,
            coefficients=unit_coefficients, backend=self.source_backend)

    def save(self, directory) -> Path:
        """Save this source tier as YAML + numeric HMMs + the Phase-1 PCA basis."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        indices = {"phones": {}, "contexts": {}, "backoff": {}}
        model_groups = (
            ("phone", self.hmms, "phones"),
            ("context", self.contexts, "contexts"),
            ("backoff", self.backoff, "backoff"),
        )
        for category, models, index_name in model_groups:
            for key, hmm in sorted(models.items()):
                prefix = _hmm_prefix(category, key)
                for name, value in hmm.to_arrays().items():
                    arrays[prefix + name] = np.asarray(value)
                indices[index_name][key] = _hmm_info(hmm)
                for state, regressor in enumerate(
                        self.conditioners.get((category, key), [])):
                    for name, value in regressor.to_arrays().items():
                        arrays[_regressor_prefix(category, key, state) + name] = value
        global_info = None
        if self.global_backoff is not None:
            prefix = _hmm_prefix("global", GLOBAL_KEY)
            for name, value in self.global_backoff.to_arrays().items():
                arrays[prefix + name] = np.asarray(value)
            global_info = _hmm_info(self.global_backoff)
            for state, regressor in enumerate(
                    self.conditioners.get(("global", GLOBAL_KEY), [])):
                for name, value in regressor.to_arrays().items():
                    arrays[_regressor_prefix("global", GLOBAL_KEY, state) + name] = value

        np.savez_compressed(directory / _SOURCE_HMMS_NPZ, **arrays)
        self.pca.save(directory / _SOURCE_PCA_NPZ)
        document = {
            "format": SOURCE_MODEL_FORMAT,
            "format_version": SOURCE_MODEL_FORMAT_VERSION,
            "name": self.name,
            "source_backend": self.source_backend,
            "source_geometry": {
                "fs": int(self.spec.fs),
                "frame_period": float(self.spec.frame_period),
                "cycle_length": int(self.pca.cycle_length),
                "unit_frames": int(self.unit_frames),
                "f0_floor": float(self.source_f0_floor),
                "f0_ceil": float(self.source_f0_ceil),
            },
            "feature_spec": self.spec.to_dict(),
            "phonemes": self.phoneme_set.to_dict(),
            "pca_file": _SOURCE_PCA_NPZ,
            "hmm_file": _SOURCE_HMMS_NPZ,
            "target": {
                "coefficient_offset": self.coefficient_offset.tolist(),
                "coefficient_scale": self.coefficient_scale.tolist(),
                "use_delta": self.use_delta,
                "use_delta2": self.use_delta2,
                "delta_window": self.delta_window,
            },
            "f0_conditioning": {
                "reference_hz": float(self.spec.f0_ref_hz),
                "voiced_threshold": float(self.spec.voiced_threshold),
                "mean_semitones": self.f0_mean,
                "scale_semitones": self.f0_scale,
                "delta_scale": self.f0_delta_scale,
                "feature_dim": F0_CONDITION_DIM,
                "features": ["log_f0_semitones", "delta_log_f0", "voiced_flag"],
            },
            "training": self.training,
            "hmm_index": indices,
            "context_index": self.context_index,
            "global_backoff_index": global_info,
        }
        with open(directory / _SOURCE_YAML, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
        return directory

    @classmethod
    def load(cls, directory) -> "SourceHMMModel":
        """Load a source tier saved by :meth:`save`, without pickle."""
        directory = Path(directory)
        if directory.is_file():
            directory = directory.parent
        yaml_path = directory / _SOURCE_YAML
        if not yaml_path.exists():
            raise FileNotFoundError(f"{directory} is not a source model "
                                    f"(missing {_SOURCE_YAML})")
        with open(yaml_path, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        if document.get("format") != SOURCE_MODEL_FORMAT:
            raise ValueError(f"{directory} is not an HMS source HMM model")
        version = int(document.get("format_version", 0))
        if version != SOURCE_MODEL_FORMAT_VERSION:
            raise ValueError(f"source model format version {version} is not "
                             f"supported (expected {SOURCE_MODEL_FORMAT_VERSION})")

        pca_path = directory / str(document.get("pca_file", _SOURCE_PCA_NPZ))
        if not pca_path.exists():
            raise ValueError(f"source model declares a PCA basis but {pca_path.name} "
                             "is missing")
        pca = SourcePCA.load(pca_path)
        spec = FeatureSpec.from_dict(document.get("feature_spec") or {})
        phoneme_set = PhonemeSet.from_dict(document.get("phonemes") or {})
        target = document.get("target") or {}
        conditioning = document.get("f0_conditioning") or {}
        if int(conditioning.get("feature_dim", F0_CONDITION_DIM)) != F0_CONDITION_DIM:
            raise ValueError("source model uses an unsupported F0 feature dimension")

        hmm_path = directory / str(document.get("hmm_file", _SOURCE_HMMS_NPZ))
        if not hmm_path.exists():
            raise ValueError(f"source model declares HMMs but {hmm_path.name} is missing")
        with np.load(hmm_path, allow_pickle=False) as handle:
            archive = {key: handle[key] for key in handle.files}

        indices = document.get("hmm_index") or {}
        hmms: Dict[str, LeftToRightHMM] = {}
        contexts: Dict[str, LeftToRightHMM] = {}
        backoff: Dict[str, LeftToRightHMM] = {}
        conditioners: Dict[Tuple[str, str], List[SourceF0Regressor]] = {}
        for category, index_name, destination in (
                ("phone", "phones", hmms),
                ("context", "contexts", contexts),
                ("backoff", "backoff", backoff)):
            for key, info in (indices.get(index_name) or {}).items():
                hmm = _load_hmm(archive, category, str(key), info)
                destination[str(key)] = hmm
                conditioners[(category, str(key))] = _load_regressors(
                    archive, category, str(key), hmm)
        global_backoff = None
        global_info = document.get("global_backoff_index")
        if global_info:
            global_backoff = _load_hmm(archive, "global", GLOBAL_KEY, global_info)
            conditioners[("global", GLOBAL_KEY)] = _load_regressors(
                archive, "global", GLOBAL_KEY, global_backoff)

        geometry = document.get("source_geometry") or {}
        model = cls(
            pca=pca, spec=spec, phoneme_set=phoneme_set,
            hmms=hmms, contexts=contexts,
            context_index=document.get("context_index") or {},
            backoff=backoff, global_backoff=global_backoff,
            conditioners=conditioners,
            coefficient_offset=target.get("coefficient_offset"),
            coefficient_scale=target.get("coefficient_scale"),
            f0_mean=float(conditioning.get("mean_semitones", 0.0)),
            f0_scale=float(conditioning.get("scale_semitones", 1.0)),
            f0_delta_scale=float(conditioning.get("delta_scale", 1.0)),
            use_delta=bool(target.get("use_delta", True)),
            use_delta2=bool(target.get("use_delta2", False)),
            delta_window=int(target.get("delta_window", spec.delta_window)),
            source_backend=str(document.get("source_backend", "voice")),
            unit_frames=int(geometry.get("unit_frames", 1)),
            source_f0_floor=float(geometry.get("f0_floor", 50.0)),
            source_f0_ceil=float(geometry.get("f0_ceil", 2000.0)),
            name=str(document.get("name", directory.name)),
            training=document.get("training") or {})
        if pca.cycle_length != int(geometry.get("cycle_length", pca.cycle_length)):
            raise ValueError("saved source geometry and PCA cycle length differ")
        if int(geometry.get("fs", spec.fs)) != spec.fs \
                or not np.isclose(float(geometry.get("frame_period", spec.frame_period)),
                                  spec.frame_period):
            raise ValueError("saved source geometry and feature frame grid differ")
        return model


def f0_condition_features(f0_hz: np.ndarray, spec: FeatureSpec,
                          mean_semitones: float = 0.0,
                          scale_semitones: float = 1.0,
                          delta_scale: float = 1.0,
                          window: Optional[int] = None) -> np.ndarray:
    """Convert the explicit HMS F0 track to the source model's fixed 3-D input.

    The columns are normalized absolute log-F0 in semitones relative to the
    existing ``FeatureSpec.f0_ref_hz``, its first dynamic feature computed with
    HMS's edge-replicated delta window (``window`` overrides the spec default),
    and a separate voiced flag.  Delta
    calculation is confined to voiced runs, so pitch gaps do not create a
    spurious pitch jump.  The caller's F0 values themselves are never modified.
    """
    f0 = _validate_f0_track(f0_hz)
    if not np.isfinite(mean_semitones) or not np.isfinite(scale_semitones) \
            or scale_semitones <= 0:
        raise ValueError("F0 semitone normalization must have a positive scale")
    if not np.isfinite(delta_scale) or delta_scale <= 0:
        raise ValueError("F0 delta scale must be positive")
    voiced = f0 > spec.voiced_threshold
    semitones = np.zeros(len(f0), dtype=np.float64)
    if voiced.any():
        semitones[voiced] = hz_to_semitone(f0[voiced], spec.f0_ref_hz)
    log_f0 = np.zeros(len(f0), dtype=np.float64)
    log_f0[voiced] = ((semitones[voiced] - float(mean_semitones))
                      / float(scale_semitones))
    delta_window = spec.delta_window if window is None else int(window)
    if delta_window < 1:
        raise ValueError("F0 delta window must be at least 1")
    delta = np.zeros(len(f0), dtype=np.float64)
    for lo, hi in _true_runs(voiced):
        if hi - lo:
            dynamic = add_dynamic_features(
                log_f0[lo:hi, None], use_delta=True, use_delta2=False,
                window=delta_window)
            delta[lo:hi] = dynamic[:, 1] / float(delta_scale)
    return np.column_stack([log_f0, delta, voiced.astype(np.float64)])


def align_source_coefficients(source: SourceSequence,
                              unit_coefficients: np.ndarray,
                              f0_hz: Optional[np.ndarray] = None,
                              voiced_threshold: float = 5.0
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Overlap-resample Phase-1 unit coefficients onto the HMS frame grid.

    Every unit contributes in proportion to its sample overlap with frame
    ``[t*hop, (t+1)*hop)``.  Thus pitch cycles that span several frames are
    held over those frames, several fast cycles in one frame are averaged, and
    frame-synchronous residual units remain exactly aligned.  Frames with no
    source unit remain invalid; there is no interpolation across a silent or
    unvoiced hole.
    """
    coefficients = np.atleast_2d(np.asarray(unit_coefficients, dtype=np.float64))
    if len(coefficients) != source.n_units:
        raise ValueError("one coefficient row is required per Phase-1 source unit")
    if coefficients.size and not np.isfinite(coefficients).all():
        raise ValueError("Phase-1 source coefficients must be finite")
    n_frames = source.n_frames
    f0 = (_validate_f0_track(f0_hz, n_frames)
          if f0_hz is not None else None)
    aligned = np.zeros((n_frames, coefficients.shape[1]), dtype=np.float64)
    coverage = np.zeros(n_frames, dtype=np.float64)
    if not n_frames or not source.n_units:
        return aligned, np.zeros(n_frames, dtype=bool)
    hop = source.hop
    n_samples = source.n_samples if source.n_samples > 0 else n_frames * hop
    for unit, (epoch, period) in enumerate(zip(source.epochs, source.periods)):
        start = max(0, int(epoch))
        stop = min(n_samples, start + int(period))
        if stop <= start:
            continue
        first = max(0, start // hop)
        last = min(n_frames, (stop + hop - 1) // hop)
        for frame in range(first, last):
            frame_start = frame * hop
            frame_stop = min(n_samples, frame_start + hop)
            overlap = max(0, min(stop, frame_stop) - max(start, frame_start))
            if overlap:
                aligned[frame] += coefficients[unit] * overlap
                coverage[frame] += overlap
    valid = coverage > 0.0
    aligned[valid] /= coverage[valid, None]
    if f0 is not None and source.backend == "voice":
        valid &= f0 > float(voiced_threshold)
    aligned[~valid] = 0.0
    return aligned, valid


def _validate_f0_track(f0_hz: np.ndarray, n_frames: Optional[int] = None) -> np.ndarray:
    """Strict version of the HMS external-F0 frame/Hz convention."""
    try:
        raw = np.asarray(f0_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("F0 must be a one-dimensional array of Hz values") from exc
    if raw.dtype.kind not in "fiu":
        raise ValueError("F0 must contain real numeric Hz values")
    f0 = raw.astype(np.float64)
    if f0.ndim != 1:
        raise ValueError(f"F0 must have shape ({n_frames if n_frames is not None else 'T'},), "
                         f"got {f0.shape}")
    if n_frames is not None and len(f0) != int(n_frames):
        raise ValueError(f"F0 has {len(f0)} frame(s), expected exactly {int(n_frames)}; "
                         "source-model F0 is never resized or interpolated")
    if not np.isfinite(f0).all():
        index = int(np.flatnonzero(~np.isfinite(f0))[0])
        raise ValueError(f"F0 frame {index} is not finite")
    if (f0 < 0.0).any():
        index = int(np.flatnonzero(f0 < 0.0)[0])
        raise ValueError(f"F0 frame {index} is negative; use 0 Hz for unvoiced")
    return f0.copy()


def _true_runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    if not flags.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], flags, [False])).astype(
        np.int8)))
    return [(int(lo), int(hi)) for lo, hi in zip(edges[::2], edges[1::2])]


def _frame_spans(utterance: Utterance, frame_period: float, n_frames: int,
                 phoneme_set: PhonemeSet
                 ) -> List[Tuple[int, str, int, int]]:
    """Existing label-segment conversion with compact non-empty positions."""
    spans = []
    for phone, lo, hi, _note in segment_boundaries(
            utterance, frame_period, n_frames=n_frames):
        lo, hi = max(0, int(lo)), min(int(n_frames), int(hi))
        if hi > lo:
            # Context neighbours follow the clipped, non-empty label segments,
            # matching Trainer._collect_cached_context_data.
            spans.append((len(spans), phoneme_set.canonical(phone), lo, hi))
    return spans


def _sample_trajectory(trajectory: np.ndarray, positions: np.ndarray) -> np.ndarray:
    trajectory = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    if not len(positions):
        return np.zeros((0, trajectory.shape[1]), dtype=np.float64)
    if len(trajectory) == 0:
        raise ValueError("cannot sample an empty source trajectory")
    grid = np.arange(len(trajectory), dtype=np.float64)
    return np.column_stack([np.interp(positions, grid, trajectory[:, dim])
                            for dim in range(trajectory.shape[1])])


def _average_active_frames(trajectory: np.ndarray, active: np.ndarray,
                           start: int, stop: int, hop: int,
                           n_samples: int) -> Optional[np.ndarray]:
    first = max(0, int(start) // hop)
    last = min(len(active), (int(stop) + hop - 1) // hop)
    total = np.zeros(trajectory.shape[1], dtype=np.float64)
    weight = 0.0
    for frame in range(first, last):
        if not active[frame]:
            continue
        frame_start = frame * hop
        frame_stop = min(n_samples, frame_start + hop)
        overlap = max(0, min(stop, frame_stop) - max(start, frame_start))
        if overlap:
            total += trajectory[frame] * overlap
            weight += overlap
    return total / weight if weight else None


def _hmm_info(hmm: LeftToRightHMM) -> Dict[str, object]:
    return {
        "n_states": int(hmm.n_states),
        "n_components": int(hmm.states[0].gmm.n_components),
        "covariance": str(hmm.covariance_type),
        "allow_skip": bool(hmm.allow_skip),
    }


def _encoded_key(key: str) -> str:
    return base64.urlsafe_b64encode(str(key).encode("utf-8")).decode("ascii").rstrip("=")


def _hmm_prefix(category: str, key: str) -> str:
    return f"{category}/{_encoded_key(key)}/hmm/"


def _regressor_prefix(category: str, key: str, state: int) -> str:
    return f"{category}/{_encoded_key(key)}/f0/{int(state)}/"


def _load_hmm(archive: Mapping[str, np.ndarray], category: str, key: str,
              info: Mapping[str, object]) -> LeftToRightHMM:
    prefix = _hmm_prefix(category, key)
    arrays = {name[len(prefix):]: value for name, value in archive.items()
              if name.startswith(prefix)}
    if not arrays:
        raise ValueError(f"source model is missing HMM arrays for {category}:{key}")
    return LeftToRightHMM.from_arrays(
        arrays, allow_skip=bool(info.get("allow_skip", False)),
        covariance_type=str(info.get("covariance", "diag")))


def _load_regressors(archive: Mapping[str, np.ndarray], category: str, key: str,
                     hmm: LeftToRightHMM) -> List[SourceF0Regressor]:
    regressors = []
    for state in range(hmm.n_states):
        prefix = _regressor_prefix(category, key, state)
        arrays = {name[len(prefix):]: value for name, value in archive.items()
                  if name.startswith(prefix)}
        if not arrays:
            raise ValueError(f"source model is missing F0 regression arrays for "
                             f"{category}:{key}, state {state}")
        regressor = SourceF0Regressor.from_arrays(arrays)
        if regressor.n_components != hmm.states[state].gmm.n_components \
                or regressor.output_dim != hmm.states[state].gmm.dim:
            raise ValueError("source F0 regressor and HMM/GMM dimensions differ")
        regressors.append(regressor)
    return regressors


__all__ = [
    "F0_CONDITION_DIM", "SourceF0Regressor", "SourceHMMModel",
    "SourcePrediction", "align_source_coefficients", "f0_condition_features",
]
