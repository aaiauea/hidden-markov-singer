"""Phoneme inventory, loaded from `phonemes.yaml`.

The inventory is *data*, not code: adding a phoneme means adding a YAML entry,
and nothing else in the engine has to change.  What the file provides:

``defaults``
    Per phoneme class (``vowel``, ``voiced_consonant``, ``unvoiced_consonant``,
    ``silence``) the default number of HMM states and GMM components.  Vowels
    get more states (attack / steady / transition phrasing needs the resolution)
    and more components; unvoiced consonants and silence get the minimum.  This
    is the main lever for the data-efficiency / quality trade-off.

``phonemes``
    The inventory: symbol -> {type, n_states, n_components, voiced, ...}.
    Missing fields fall back to the class defaults.

``aliases``
    Alternative spellings that map onto an inventory symbol (``pau`` -> ``sil``).

Phonemes seen in labels or scores but absent from the inventory are *not* a
hard error: `PhonemeSet.resolve` returns ``None`` and the trainer/synthesiser
falls back to a per-class "backoff" model, so a new symbol degrades gracefully
instead of crashing the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

#: Recognised phoneme classes.
CLASSES = ("vowel", "voiced_consonant", "unvoiced_consonant", "silence")

#: Defaults used when a class is missing from the YAML file.
FALLBACK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "vowel": {"n_states": 5, "n_components": 2, "voiced": True, "can_hold_note": True},
    "voiced_consonant": {"n_states": 3, "n_components": 1, "voiced": True,
                         "can_hold_note": False},
    "unvoiced_consonant": {"n_states": 2, "n_components": 1, "voiced": False,
                           "can_hold_note": False},
    "silence": {"n_states": 1, "n_components": 1, "voiced": False,
                "can_hold_note": False},
}

#: Symbols treated as silence if the inventory does not say otherwise.
DEFAULT_SILENCE = ("sil", "pau", "_", "SIL", "sp")


@dataclass
class PhonemeDef:
    """One phoneme's modelling metadata."""

    symbol: str
    type: str = "unvoiced_consonant"
    n_states: int = 2
    n_components: int = 1
    voiced: bool = False
    can_hold_note: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": self.type,
            "n_states": self.n_states,
            "n_components": self.n_components,
            "voiced": self.voiced,
            "can_hold_note": self.can_hold_note,
        }
        out.update(self.extra)
        return out


class PhonemeSet:
    """The phoneme inventory used by a model."""

    def __init__(self, phonemes: Dict[str, PhonemeDef],
                 aliases: Optional[Dict[str, str]] = None,
                 defaults: Optional[Dict[str, Dict[str, Any]]] = None,
                 silence: str = "sil") -> None:
        self.phonemes = phonemes
        self.aliases = dict(aliases or {})
        self.defaults = dict(defaults or FALLBACK_DEFAULTS)
        self.silence = silence

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhonemeSet":
        defaults = {name: dict(FALLBACK_DEFAULTS.get(name, {}))
                    for name in CLASSES}
        configured_defaults = data.get("defaults") or {}
        if not isinstance(configured_defaults, dict):
            raise ValueError("phoneme defaults must be a mapping by class")
        for name, values in configured_defaults.items():
            if name not in CLASSES:
                raise ValueError(f"unknown phoneme class in defaults: {name!r}")
            if values is not None and not isinstance(values, dict):
                raise ValueError(f"defaults for {name!r} must be a mapping")
            defaults[name].update(values or {})

        entries = data.get("phonemes") or {}
        if not isinstance(entries, dict) or not entries:
            raise ValueError("phonemes.yaml must define phonemes as a non-empty mapping")
        phonemes: Dict[str, PhonemeDef] = {}
        for symbol, raw_values in entries.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"phoneme symbols must be non-empty strings, "
                                 f"got {symbol!r}")
            if raw_values is not None and not isinstance(raw_values, dict):
                raise ValueError(f"definition for phoneme {symbol!r} must be a mapping")
            values = dict(raw_values or {})
            klass = values.get("type", "unvoiced_consonant")
            if klass not in CLASSES:
                raise ValueError(f"phoneme {symbol!r} has unknown type {klass!r}; "
                                 f"expected one of {CLASSES}")
            base = dict(defaults[klass])
            merged = {**base, **values}
            try:
                n_states = int(merged.get("n_states", 2))
                n_components = int(merged.get("n_components", 1))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"phoneme {symbol!r} state/component counts "
                                 "must be integers") from exc
            if n_states < 1 or n_components < 1:
                raise ValueError(f"phoneme {symbol!r} n_states and n_components "
                                 "must be positive")
            extra = {k: v for k, v in merged.items()
                     if k not in ("type", "n_states", "n_components", "voiced",
                                  "can_hold_note")}
            phonemes[symbol] = PhonemeDef(
                symbol=symbol,
                type=klass,
                n_states=n_states,
                n_components=n_components,
                voiced=bool(merged.get("voiced", klass in ("vowel",
                                                           "voiced_consonant"))),
                can_hold_note=bool(merged.get("can_hold_note",
                                              klass == "vowel")),
                extra=extra)

        aliases = {str(k): str(v) for k, v in (data.get("aliases") or {}).items()}
        silence = str(data.get("silence", "sil"))
        if silence not in phonemes:
            available = [p for p in DEFAULT_SILENCE if p in phonemes]
            if available:
                silence = available[0]
        return cls(phonemes, aliases, defaults, silence)

    @classmethod
    def from_yaml(cls, path) -> "PhonemeSet":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    @classmethod
    def default(cls) -> "PhonemeSet":
        """The inventory shipped in `hms/config/phonemes.yaml`."""
        return cls.from_yaml(Path(__file__).resolve().parents[1]
                             / "config" / "phonemes.yaml")

    # -- lookup ------------------------------------------------------------

    def resolve(self, symbol: str) -> Optional[PhonemeDef]:
        """Look a symbol up, following aliases; ``None`` if unknown."""
        if symbol in self.phonemes:
            return self.phonemes[symbol]
        target = self.aliases.get(symbol)
        if target is not None and target in self.phonemes:
            return self.phonemes[target]
        return None

    def canonical(self, symbol: str) -> str:
        """Canonical symbol, or the input unchanged when unknown."""
        if symbol in self.phonemes:
            return symbol
        target = self.aliases.get(symbol)
        return target if target in self.phonemes else symbol

    def __contains__(self, symbol: str) -> bool:
        return self.resolve(symbol) is not None

    def __len__(self) -> int:
        return len(self.phonemes)

    def __iter__(self) -> Iterable[PhonemeDef]:
        return iter(self.phonemes.values())

    @property
    def symbols(self) -> List[str]:
        return sorted(self.phonemes)

    def of_type(self, klass: str) -> List[PhonemeDef]:
        return [p for p in self.phonemes.values() if p.type == klass]

    def n_states(self, symbol: str) -> int:
        definition = self.resolve(symbol)
        if definition is None:
            return int(self.defaults.get("unvoiced_consonant", {})
                       .get("n_states", 2))
        return definition.n_states

    def n_components(self, symbol: str) -> int:
        definition = self.resolve(symbol)
        if definition is None:
            return 1
        return definition.n_components

    def is_voiced(self, symbol: str) -> bool:
        definition = self.resolve(symbol)
        return bool(definition.voiced) if definition else False

    def unknown(self, symbols: Iterable[str]) -> List[str]:
        """Symbols that would fall back to a backoff model."""
        return sorted({s for s in symbols if self.resolve(s) is None})

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silence": self.silence,
            "defaults": self.defaults,
            "aliases": self.aliases,
            "phonemes": {name: p.to_dict()
                         for name, p in sorted(self.phonemes.items())},
        }

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False,
                           allow_unicode=True)

    # -- helpers -----------------------------------------------------------

    def summary(self) -> List[str]:
        """Human-readable inventory listing (used by `hms inspect-model`)."""
        lines = []
        for klass in CLASSES:
            members = self.of_type(klass)
            if not members:
                continue
            lines.append(f"  {klass:20s} ({len(members)})")
            for definition in members:
                lines.append(
                    f"    {definition.symbol:8s} states={definition.n_states} "
                    f"components={definition.n_components} "
                    f"voiced={'y' if definition.voiced else 'n'}")
        if self.aliases:
            lines.append("  aliases: "
                         + ", ".join(f"{k}->{v}" for k, v in
                                     sorted(self.aliases.items())))
        return lines
