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
from typing import Dict, List, Optional

import numpy as np
import yaml

from hms.core.duration import DurationModel
from hms.core.features import FeatureSpec
from hms.core.hmm import LeftToRightHMM
from hms.core.phonemes import PhonemeSet
from hms.core.pitch import PitchModel

#: Bumped when the on-disk layout changes incompatibly.
#: 2 -- the spectral envelope is sampled on `2 * n_mcep` mel bands instead of
#:      `n_mcep + 2` (see `hms.core.features.mel_band_count`), so the stored
#:      cepstral coefficients mean something different from format 1.
MODEL_FORMAT_VERSION = 2

_MODEL_YAML = "model.yaml"
_HMM_NPZ = "hmm.npz"
_BACKOFF_NPZ = "backoff.npz"


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
                 metadata: Optional[Dict[str, object]] = None) -> None:
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
        raise KeyError(f"no model for phoneme {phoneme!r} and no backoff "
                       f"available (classes: {sorted(self.backoff)})")

    @property
    def n_free_params(self) -> int:
        """Total free parameters of the acoustic model (data-efficiency gauge)."""
        return sum(h.n_free_params for h in self.hmms.values())

    def parameter_report(self) -> List[str]:
        lines = [
            f"  phonemes modelled : {len(self.hmms)}",
            f"  feature dim       : {self.feature_dim} "
            f"({self.static_dim} static x {len(self.spec.stream_sizes)} streams)",
            f"  HMM free params   : {self.n_free_params:,}",
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
            backoff_index[key] = {"n_states": hmm.n_states,
                                  "n_components": hmm.states[0].gmm.n_components}
        if backoff_arrays:
            np.savez_compressed(directory / _BACKOFF_NPZ, **backoff_arrays)

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
        if version != MODEL_FORMAT_VERSION:
            raise ValueError(f"model format version {version} is not supported "
                             f"by this build (expected {MODEL_FORMAT_VERSION})")

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
            for key in (document.get("backoff_index") or {}):
                prefix = f"{key}/"
                subset = {k[len(prefix):]: v for k, v in backoff_arrays.items()
                          if k.startswith(prefix)}
                if subset:
                    backoff[key] = LeftToRightHMM.from_arrays(subset)

        normalization = document.get("normalization") or {}
        return cls(
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
        )

    def exists(self, directory) -> bool:
        return (Path(directory) / _MODEL_YAML).exists()

    # -- inspection --------------------------------------------------------

    def summary(self) -> str:
        """Multi-line description used by `hms inspect-model`."""
        lines = [f"model          : {self.name}",
                 f"format version : {MODEL_FORMAT_VERSION}",
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
