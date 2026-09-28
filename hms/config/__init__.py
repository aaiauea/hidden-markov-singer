"""Configuration loading.

`parameters.yaml` describes the whole system; this module turns it into the
dataclasses the core uses, applying command-line overrides on top.  Anything not
mentioned in the YAML keeps the dataclass default, so the file can be trimmed to
just the values you want to change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from hms.core.features import FeatureSpec
from hms.core.phonemes import PhonemeSet
from hms.core.pitch import Vibrato
from hms.core.synthesizer import SynthesisConfig
from hms.core.trainer import TrainingConfig

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMETERS = CONFIG_DIR / "parameters.yaml"
DEFAULT_PHONEMES = CONFIG_DIR / "phonemes.yaml"

#: Accepted aliases so hand-written parameter files can use either spelling.
_ALIASES = {
    "frame_period_ms": "frame_period",
    "frame_period": "frame_period",
    "f0_ceiling": "f0_ceiling",
    "f0_ceil": "f0_ceiling",
}


def load_parameters(path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Read a parameters file (defaults to the bundled one)."""
    path = Path(path) if path else DEFAULT_PARAMETERS
    if not path.exists():
        raise FileNotFoundError(f"parameters file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_phoneme_set(path: Optional[str | Path] = None) -> PhonemeSet:
    path = Path(path) if path else DEFAULT_PHONEMES
    if not path.exists():
        raise FileNotFoundError(f"phoneme file not found: {path}")
    return PhonemeSet.from_yaml(path)


def build_feature_spec(parameters: Mapping[str, Any],
                       fft_size: Optional[int] = None) -> FeatureSpec:
    """`acoustic:` section -> FeatureSpec."""
    acoustic = dict(parameters.get("acoustic") or {})
    spec_kwargs: Dict[str, Any] = {}
    for key, value in acoustic.items():
        name = _ALIASES.get(key, key)
        if value is not None:
            spec_kwargs[name] = value
    if fft_size is not None:
        spec_kwargs["fft_size"] = fft_size
    return FeatureSpec.from_dict(spec_kwargs)


def training_config_from_parameters(parameters: Mapping[str, Any]
                                    ) -> TrainingConfig:
    acoustic = dict(parameters.get("acoustic") or {})
    training = dict(parameters.get("training") or {})
    duration = dict(parameters.get("duration") or {})
    pitch = dict(parameters.get("pitch") or {})

    values: Dict[str, Any] = {}
    for key, value in acoustic.items():
        name = _ALIASES.get(key, key)
        if name in TrainingConfig.__dataclass_fields__ and value is not None:  # type: ignore[attr-defined]
            values[name] = value
    for key, value in training.items():
        name = _ALIASES.get(key, key)
        if name in TrainingConfig.__dataclass_fields__ and value is not None:  # type: ignore[attr-defined]
            values[name] = value
    if duration.get("variance_scale") is not None:
        values["duration_variance_scale"] = duration["variance_scale"]
    context = parameters.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("the context section must be a mapping")
    for key, value in context.items():
        name = f"context_{_ALIASES.get(key, key)}"
        if name not in TrainingConfig.__dataclass_fields__:  # type: ignore[attr-defined]
            raise ValueError(f"unknown context option: {key!r}")
        if value is not None:
            values[name] = value
    conditioning = parameters.get("pitch_conditioning") or {}
    if not isinstance(conditioning, dict):
        raise ValueError("the pitch_conditioning section must be a mapping")
    for key, value in conditioning.items():
        name = f"pitch_conditioning_{_ALIASES.get(key, key)}"
        if name not in TrainingConfig.__dataclass_fields__:  # type: ignore[attr-defined]
            raise ValueError(f"unknown pitch_conditioning option: {key!r}")
        if value is not None:
            values[name] = value
    transfer = parameters.get("transfer") or {}
    if not isinstance(transfer, dict):
        raise ValueError("the transfer section must be a mapping")
    for key, value in transfer.items():
        name = f"transfer_{_ALIASES.get(key, key)}"
        if name not in TrainingConfig.__dataclass_fields__:  # type: ignore[attr-defined]
            raise ValueError(f"unknown transfer option: {key!r}")
        if value is not None:
            values[name] = value
    if "vibrato" in pitch:
        values["vibrato_enabled"] = bool(
            (pitch["vibrato"] or {}).get("enabled", False))
    if "vibrato_estimate_from_data" in pitch:
        values["vibrato_estimate_from_data"] = bool(
            pitch["vibrato_estimate_from_data"])
    return TrainingConfig(**values)


def synthesis_config_from_parameters(parameters: Mapping[str, Any]
                                     ) -> SynthesisConfig:
    synthesis = dict(parameters.get("synthesis") or {})
    values = {k: v for k, v in synthesis.items()
              if k in SynthesisConfig.__dataclass_fields__  # type: ignore[attr-defined]
              and v is not None}
    return SynthesisConfig(**values)


def vibrato_from_parameters(parameters: Mapping[str, Any]) -> Vibrato:
    pitch = dict(parameters.get("pitch") or {})
    return Vibrato.from_dict(pitch.get("vibrato") or {})


def apply_overrides(config: Any, overrides: Mapping[str, Any]) -> Any:
    """Set non-None overrides onto a dataclass instance (in place)."""
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(config, key):
            raise ValueError(f"{type(config).__name__} has no field {key!r}")
        setattr(config, key, value)
    return config
