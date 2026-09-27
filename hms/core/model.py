"""The HMS voice model: what training produces and synthesis consumes.

On-disk format (a directory, both halves human-readable):

``model.yaml``
    Everything scalar and structural: metadata, the `FeatureSpec`, the phoneme
    inventory, feature normalisation, the duration model, pitch/vibrato
    settings, voicing priors, and the *index* of the HMMs (state counts, number
    of components, parameter counts).  You can read the whole acoustic design of
    a voice in a text editor, and diff two models.

``hmm.npz``
    The numeric payload: per phoneme, per state GMM weights / means /
    variances, self-loop probabilities, duration and voicing statistics.  One
    array set per phoneme keeps it inspectable with ``numpy.load`` and any
    ``.npz`` tool.

Backoff models (``backoff`` section in the YAML, ``backoff.npz``) are aggregated
models per phoneme class, used when a score asks for a symbol the model has
never seen.  That is what lets a new phoneme be added to `phonemes.yaml`
without retraining the engine or crashing: it degrades to a plausible sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from hms.core.context import (GLOBAL_KEY, context_wildcard,
                              left_diphone_key, right_diphone_key,
                              triphone_key)
from hms.core.duration import DurationModel
from hms.core.features import FeatureSpec
from hms.core.hmm import LeftToRightHMM
from hms.core.phonemes import PhonemeSet
from hms.core.pitch import PitchModel

#: Bumped when the on-disk layout changes incompatibly.
#: 2 -- the spectral envelope is sampled on `2 * n_mcep` mel bands instead of
#:      `n_mcep + 2` (see `hms.core.features.mel_band_count`), so the stored
#:      cepstral coefficients mean something different from format 1.
#: 3 -- adds optional sparse phoneme-context modelling: a `context_index` in
#:      the YAML and a `context.npz` array file (plus an optional global
#:      backoff model).  Format-2 files carry no contexts and still load.
MODEL_FORMAT_VERSION = 3

#: Format versions this build can read.
SUPPORTED_FORMAT_VERSIONS = (2, 3)

_MODEL_YAML = "model.yaml"
_HMM_NPZ = "hmm.npz"
_BACKOFF_NPZ = "backoff.npz"
_CONTEXT_NPZ = "context.npz"


@dataclass
class ModelStats:
    """Book-keeping about how a model was trained (shown by inspect-model)."""

    utterances: int = 0
    frames: int = 0
    phoneme_occurrences: int = 0
    duration_seconds: float = 0.0
    training_method: str = "viterbi"
    n_iterations: int = 0
    created: str = ""
    notes: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> Dict[str, object]:
        return {
            "utterances": self.utterances,
            "frames": self.frames,
            "phoneme_occurrences": self.phoneme_occurrences,
            "duration_seconds": round(self.duration_seconds, 3),
            "training_method": self.training_method,
            "n_iterations": self.n_iterations,
            "created": self.created,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ModelStats":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class HMSModel:
    """A trained single-speaker singing voice."""

    def __init__(self, name: str, spec: FeatureSpec, phoneme_set: PhonemeSet,
                 hmms: Dict[str, LeftToRightHMM],
                 duration_model: Optional[DurationModel] = None,
                 pitch_model: Optional[PitchModel] = None,
                 normalization: Optional[Dict[str, np.ndarray]] = None,
                 stats: Optional[ModelStats] = None,
                 backoff: Optional[Dict[str, LeftToRightHMM]] = None,
                 metadata: Optional[Dict[str, object]] = None,
                 contexts: Optional[Dict[str, LeftToRightHMM]] = None,
                 context_index: Optional[Dict[str, dict]] = None,
                 global_backoff: Optional[LeftToRightHMM] = None) -> None:
        self.name = name
        self.spec = spec
        self.phoneme_set = phoneme_set
        self.hmms = hmms
        self.duration_model = duration_model or DurationModel()
        self.pitch_model = pitch_model or PitchModel()
        #: feature-space offset/scale: z = (x - offset) * scale
        self.offset = (normalization or {}).get("offset")
        self.scale = (normalization or {}).get("scale")
        self.stats = stats or ModelStats()
        self.backoff = backoff or {}
        self.metadata = metadata or {}
        #: sparse phone-context HMMs (empty unless trained with contexts on)
        self.contexts = contexts or {}
        #: per context key: kind, phones, support, trained geometry
        self.context_index = context_index or {}
        #: optional pooled catch-all HMM, last rung of the fallback hierarchy
        self.global_backoff = global_backoff
        #: format version the model was read from (set by `load`)
        self.loaded_format_version: Optional[int] = None

    # -- geometry ----------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return self.spec.dim

    @property
    def static_dim(self) -> int:
        return self.spec.static_dim

    @property
    def feature_offset(self) -> np.ndarray:
        if self.offset is None:
            return np.zeros(self.static_dim)
        return np.asarray(self.offset, dtype=np.float64)

    @property
    def feature_scale(self) -> np.ndarray:
        if self.scale is None:
            return np.ones(self.static_dim)
        return np.asarray(self.scale, dtype=np.float64)

    # -- normalisation -----------------------------------------------------

    def normalize(self, static: np.ndarray) -> np.ndarray:
        return (np.asarray(static, dtype=np.float64)
                - self.feature_offset) * self.feature_scale

    def denormalize(self, static: np.ndarray) -> np.ndarray:
        return np.asarray(static, dtype=np.float64) / self.feature_scale \
            + self.feature_offset

    # -- lookup ------------------------------------------------------------

    def get_hmm(self, phoneme: str) -> Optional[LeftToRightHMM]:
        """The HMM for a phoneme, following aliases, else ``None``."""
        canonical = self.phoneme_set.canonical(phoneme)
        hmm = self.hmms.get(canonical)
        if hmm is not None:
            return hmm
        return None

    def get_or_backoff(self, phoneme: str) -> LeftToRightHMM:
        """HMM for a phoneme, falling back to its class model when unknown."""
        hmm = self.get_hmm(phoneme)
        if hmm is not None:
            return hmm
        definition = self.phoneme_set.resolve(phoneme)
        klass = definition.type if definition else "unvoiced_consonant"
        for key in (klass, "unvoiced_consonant"):
            if key in self.backoff:
                return self.backoff[key]
        if self.global_backoff is not None:
            return self.global_backoff
        raise KeyError(f"no model for phoneme {phoneme!r} and no backoff "
                       f"available (classes: {sorted(self.backoff)})")

    # -- context resolution --------------------------------------------------

    def context_support(self, key: str) -> Optional[int]:
        """Pooled training frames of a context key, if that context exists."""
        if key not in self.contexts:
            return None
        info = self.context_index.get(key) or {}
        try:
            return int(info.get("frames", 0))
        except (TypeError, ValueError):
            return 0

    def resolve_unit(self, pre: str, curr: str, post: str
                     ) -> Tuple[Optional[str], LeftToRightHMM, str]:
        """Pick the HMM for a phone occurrence with context fallbacks.

        Returns ``(context_key_or_None, hmm, tier)`` following the fixed
        hierarchy: exact triphone, best-supported one-sided diphone (ties
        favour the left context), dedicated current-phone HMM, phone-class
        backoff, optional global backoff.  With no trained contexts this is
        exactly `get_or_backoff`.
        """
        canonical = self.phoneme_set.canonical
        pre_c, curr_c, post_c = canonical(pre), canonical(curr), canonical(post)
        if self.contexts:
            tri = triphone_key(pre_c, curr_c, post_c)
            if tri in self.contexts:
                return tri, self.contexts[tri], "triphone"
            wildcard = context_wildcard(self.phoneme_set.phonemes)
            left_key = left_diphone_key(pre_c, curr_c, wildcard)
            right_key = right_diphone_key(curr_c, post_c, wildcard)
            left_support = self.context_support(left_key)
            right_support = self.context_support(right_key)
            if left_support is not None \
                    and (right_support is None
                         or left_support >= right_support):
                return left_key, self.contexts[left_key], "left"
            if right_support is not None:
                return right_key, self.contexts[right_key], "right"
        hmm = self.get_hmm(curr)
        if hmm is not None:
            return None, hmm, "phone"
        definition = self.phoneme_set.resolve(curr)
        klass = definition.type if definition else "unvoiced_consonant"
        for key in (klass, "unvoiced_consonant"):
            if key in self.backoff:
                return None, self.backoff[key], "class"
        if self.global_backoff is not None:
            return None, self.global_backoff, "global"
        raise KeyError(f"no model for phoneme {curr!r} and no backoff "
                       f"available (classes: {sorted(self.backoff)})")

    @property
    def phoneme_n_free_params(self) -> int:
        """Free parameters in the dedicated per-phoneme HMMs."""
        return sum(h.n_free_params for h in self.hmms.values())

    @property
    def backoff_n_free_params(self) -> int:
        """Free parameters in the pooled class backoff HMMs."""
        return sum(h.n_free_params for h in self.backoff.values())

    @property
    def context_n_free_params(self) -> int:
        """Free parameters in the sparse phone-context HMMs."""
        return sum(h.n_free_params for h in self.contexts.values())

    @property
    def global_backoff_n_free_params(self) -> int:
        """Free parameters in the optional pooled global backoff HMM."""
        if self.global_backoff is None:
            return 0
        return self.global_backoff.n_free_params

    @property
    def n_free_params(self) -> int:
        """Total free parameters across every model tier."""
        return (self.phoneme_n_free_params + self.context_n_free_params
                + self.backoff_n_free_params
                + self.global_backoff_n_free_params)

    def parameter_report(self) -> List[str]:
        lines = [
            f"  phonemes modelled : {len(self.hmms)}",
            f"  context models    : {len(self.contexts)}"
            + (" + global backoff" if self.global_backoff is not None else ""),
            f"  feature dim       : {self.feature_dim} "
            f"({self.static_dim} static x {len(self.spec.stream_sizes)} streams)",
            f"  phoneme HMM params: {self.phoneme_n_free_params:,}",
            f"  context HMM params: {self.context_n_free_params:,}",
            f"  backoff HMM params: {self.backoff_n_free_params:,}",
            f"  global HMM params : {self.global_backoff_n_free_params:,}",
            f"  total HMM params  : {self.n_free_params:,}",
            f"  frames of training: {self.stats.frames:,} "
            f"({self.stats.duration_seconds:.1f} s)",
        ]
        if self.stats.frames:
            lines.append(f"  params per frame  : "
                         f"{self.n_free_params / self.stats.frames:.2f}")
        return lines

    # -- serialisation -----------------------------------------------------

    def save(self, directory) -> Path:
        """Write the model to ``directory`` (creating it if needed)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        hmm_arrays: Dict[str, np.ndarray] = {}
        index: Dict[str, object] = {}
        for phoneme, hmm in sorted(self.hmms.items()):
            for key, value in hmm.to_arrays().items():
                hmm_arrays[f"{phoneme}/{key}"] = value
            index[phoneme] = {
                "n_states": hmm.n_states,
                "n_components": hmm.states[0].gmm.n_components,
                "covariance": hmm.covariance_type,
                "allow_skip": bool(hmm.allow_skip),
                "stated_type": self.phoneme_set.resolve(phoneme).type
                if self.phoneme_set.resolve(phoneme) else "unknown",
                "n_free_params": hmm.n_free_params,
            }
        np.savez_compressed(directory / _HMM_NPZ, **hmm_arrays)

        backoff_arrays: Dict[str, np.ndarray] = {}
        backoff_index: Dict[str, object] = {}
        for key, hmm in sorted(self.backoff.items()):
            for name, value in hmm.to_arrays().items():
                backoff_arrays[f"{key}/{name}"] = value
            backoff_index[key] = {
                "n_states": hmm.n_states,
                "n_components": hmm.states[0].gmm.n_components,
                "covariance": hmm.covariance_type,
                "allow_skip": bool(hmm.allow_skip),
            }
        if backoff_arrays:
            np.savez_compressed(directory / _BACKOFF_NPZ, **backoff_arrays)

        context_arrays: Dict[str, np.ndarray] = {}
        for key, hmm in sorted(self.contexts.items()):
            for name, value in hmm.to_arrays().items():
                context_arrays[f"{key}/{name}"] = value
        global_index: Dict[str, object] = {}
        if self.global_backoff is not None:
            for name, value in self.global_backoff.to_arrays().items():
                context_arrays[f"{GLOBAL_KEY}/{name}"] = value
            global_index = {
                "n_states": self.global_backoff.n_states,
                "n_components":
                    self.global_backoff.states[0].gmm.n_components,
                "covariance": self.global_backoff.covariance_type,
                "allow_skip": bool(self.global_backoff.allow_skip),
                "n_free_params": self.global_backoff.n_free_params,
            }
        if context_arrays:
            np.savez_compressed(directory / _CONTEXT_NPZ, **context_arrays)

        document = {
            "format_version": MODEL_FORMAT_VERSION,
            "name": self.name,
            "stats": self.stats.to_dict(),
            "metadata": self.metadata,
            "feature_spec": self.spec.to_dict(),
            "normalization": {
                "offset": [float(v) for v in self.feature_offset],
                "scale": [float(v) for v in self.feature_scale],
                "units": {
                    "0": "semitones re. the sung note",
                    f"1:{1 + self.spec.n_mcep}": "mel-cepstrum (c0..)",
                    f"{1 + self.spec.n_mcep}:": "aperiodicity bands",
                },
            },
            "phonemes": self.phoneme_set.to_dict(),
            "duration_model": self.duration_model.to_dict(),
            "pitch_model": self.pitch_model.to_dict(),
            "hmm_index": index,
            "backoff_index": backoff_index,
        }
        if self.contexts:
            document["context_index"] = dict(sorted(self.context_index.items()))
        if global_index:
            document["global_backoff_index"] = global_index
        with open(directory / _MODEL_YAML, "w", encoding="utf-8") as handle:
            yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=True)
        return directory

    @classmethod
    def load(cls, directory) -> "HMSModel":
        directory = Path(directory)
        if directory.is_file():
            directory = directory.parent
        yaml_path = directory / _MODEL_YAML
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"{directory} does not look like an HMS model "
                f"(missing {_MODEL_YAML})")
        with open(yaml_path, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}

        version = int(document.get("format_version", 0))
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"model format version {version} is not supported by this "
                f"build (supported: {SUPPORTED_FORMAT_VERSIONS}; this build "
                f"writes {MODEL_FORMAT_VERSION})")

        spec = FeatureSpec.from_dict(document.get("feature_spec") or {})
        phoneme_set = PhonemeSet.from_dict(document.get("phonemes") or {})

        with np.load(directory / _HMM_NPZ) as handle:
            arrays = {key: handle[key] for key in handle.files}
        hmms: Dict[str, LeftToRightHMM] = {}
        for phoneme, info in (document.get("hmm_index") or {}).items():
            prefix = f"{phoneme}/"
            subset = {k[len(prefix):]: v for k, v in arrays.items()
                      if k.startswith(prefix)}
            if not subset:
                continue
            hmms[phoneme] = LeftToRightHMM.from_arrays(
                subset, allow_skip=bool(info.get("allow_skip", False)),
                covariance_type=str(info.get("covariance", "diag")))

        backoff: Dict[str, LeftToRightHMM] = {}
        backoff_path = directory / _BACKOFF_NPZ
        if backoff_path.exists():
            with np.load(backoff_path) as handle:
                backoff_arrays = {key: handle[key] for key in handle.files}
            for key, info in (document.get("backoff_index") or {}).items():
                prefix = f"{key}/"
                subset = {k[len(prefix):]: v for k, v in backoff_arrays.items()
                          if k.startswith(prefix)}
                if subset:
                    backoff[key] = LeftToRightHMM.from_arrays(
                        subset, allow_skip=bool(info.get("allow_skip", False)),
                        covariance_type=str(info.get("covariance", "diag")))

        contexts: Dict[str, LeftToRightHMM] = {}
        context_index = dict(document.get("context_index") or {})
        global_index = document.get("global_backoff_index") or {}
        global_backoff: Optional[LeftToRightHMM] = None
        if context_index or global_index:
            context_path = directory / _CONTEXT_NPZ
            if not context_path.exists():
                raise ValueError(
                    f"{directory} declares context models but {_CONTEXT_NPZ} "
                    f"is missing")
            with np.load(context_path) as handle:
                context_arrays = {key: handle[key] for key in handle.files}
            for key, info in context_index.items():
                prefix = f"{key}/"
                subset = {k[len(prefix):]: v
                          for k, v in context_arrays.items()
                          if k.startswith(prefix)}
                if subset:
                    contexts[key] = LeftToRightHMM.from_arrays(
                        subset, allow_skip=bool(info.get("allow_skip", False)),
                        covariance_type=str(info.get("covariance", "diag")))
            context_index = {key: context_index[key] for key in contexts}
            if global_index:
                prefix = f"{GLOBAL_KEY}/"
                subset = {k[len(prefix):]: v
                          for k, v in context_arrays.items()
                          if k.startswith(prefix)}
                if subset:
                    global_backoff = LeftToRightHMM.from_arrays(
                        subset,
                        allow_skip=bool(global_index.get("allow_skip", False)),
                        covariance_type=str(
                            global_index.get("covariance", "diag")))

        normalization = document.get("normalization") or {}
        model = cls(
            name=str(document.get("name", directory.name)),
            spec=spec,
            phoneme_set=phoneme_set,
            hmms=hmms,
            duration_model=DurationModel.from_dict(
                document.get("duration_model") or {}),
            pitch_model=PitchModel.from_dict(document.get("pitch_model") or {}),
            normalization={
                "offset": np.asarray(normalization.get("offset", []),
                                     dtype=np.float64) if
                normalization.get("offset") else None,
                "scale": np.asarray(normalization.get("scale", []),
                                    dtype=np.float64) if
                normalization.get("scale") else None,
            },
            stats=ModelStats.from_dict(document.get("stats") or {}),
            backoff=backoff,
            metadata=document.get("metadata") or {},
            contexts=contexts,
            context_index=context_index,
            global_backoff=global_backoff,
        )
        model.loaded_format_version = version
        return model

    def exists(self, directory) -> bool:
        return (Path(directory) / _MODEL_YAML).exists()

    # -- inspection --------------------------------------------------------

    def summary(self) -> str:
        """Multi-line description used by `hms inspect-model`."""
        version = getattr(self, "loaded_format_version", None) \
            or MODEL_FORMAT_VERSION
        lines = [f"model          : {self.name}",
                 f"format version : {version}",
                 f"loader         : {self.metadata.get('hms_version', 'hms')}"]
        lines += self.parameter_report()
        lines.append("")
        lines.append("feature spec:")
        for key, value in sorted(self.spec.to_dict().items()):
            lines.append(f"  {key:24s} {value}")
        lines.append("")
        lines.append("phonemes:")
        lines += self.phoneme_set.summary()
        if self.backoff:
            lines.append("  backoff models: " + ", ".join(sorted(self.backoff)))
        lines.append("")
        lines.append("HMMs (phoneme: states x components, params):")
        for phoneme, hmm in sorted(self.hmms.items()):
            lines.append(
                f"  {phoneme:8s} {hmm.n_states} x "
                f"{hmm.states[0].gmm.n_components:<2d} "
                f"{hmm.n_free_params:7,d} params")
        if self.contexts or self.global_backoff is not None:
            lines.append("")
            lines.append("context HMMs (sparse, key: states x components, "
                         "params):")
            for key, hmm in sorted(self.contexts.items()):
                info = self.context_index.get(key) or {}
                lines.append(
                    f"  {key:18s} {str(info.get('kind', '?')):8s} "
                    f"{hmm.n_states} x {hmm.states[0].gmm.n_components:<2d} "
                    f"{hmm.n_free_params:7,d} params "
                    f"({info.get('frames', '?')} frames, "
                    f"{info.get('occurrences', '?')} occ)")
            if self.global_backoff is not None:
                lines.append(
                    f"  {'(global backoff)':18s} {'pooled':8s} "
                    f"{self.global_backoff.n_states} x "
                    f"{self.global_backoff.states[0].gmm.n_components:<2d} "
                    f"{self.global_backoff.n_free_params:7,d} params")
        lines.append("")
        lines.append("training:")
        for key, value in self.stats.to_dict().items():
            lines.append(f"  {key:18s} {value}")
        lines.append("")
        lines.append("pitch:")
        lines.append(f"  pitch_variation   {self.pitch_model.pitch_variation}")
        for key, value in self.pitch_model.vibrato.to_dict().items():
            lines.append(f"  vibrato.{key:16s} {value}")
        return "\n".join(lines)


def load_model(directory) -> HMSModel:
    """Convenience wrapper around :meth:`HMSModel.load`."""
    return HMSModel.load(directory)
