"""Source/excitation models (Phase 1: representation and analysis).

HMS has always modelled the *filter* (a mel-cepstrum envelope per frame) and
left the *source* to the vocoder.  This package adds a representation for the
source itself, so that a later phase can model it next to the spectral model::

                 ┌── spectral model ──→ spectral envelope ──┐
    HMM output ──┤                                           ├─→ filter ─→ audio
                 └── source model ────→ excitation ─────────┘

Phase 1 is analysis and validation only: nothing here is called by the trainer,
the parameter generator or the vocoder, so the existing synthesiser is bit-for-bit
unchanged.  What exists now is

    audio ──► source cycles (128 samples) ──► PCA coefficients (4-16)
                                                │
                                                └──► reconstructed cycles

* :class:`~hms.source.base.SourceFrame` / :class:`~hms.source.base.SourceSequence`
  -- the generic, backend-independent representation.
* :class:`~hms.source.base.SourceModel` -- the interface a source backend
  implements (``analyze`` plus the shared ``encode``/``decode``/``synthesize``).
* :class:`~hms.source.voice.VoiceSourceModel` -- pitch-synchronous residual
  cycles (`get_source_model("voice")`), the first concrete backend.
* :class:`~hms.source.generic.GenericResidualSourceModel` -- the pitch-free,
  frame-synchronous residual (`get_source_model("residual")`), which is what a
  non-voice source can start from.
* :class:`~hms.source.pca.SourcePCA` -- NumPy-only PCA over the source vectors,
  with save/load and reconstruction measurement.

See ``docs/source_model.md`` for the design and ``tools/bench_source_pca.py`` for
the reconstruction experiment (cycles -> coefficients -> cycles, MSE per number
of components).
"""

from __future__ import annotations

from typing import Dict, Optional

from hms.source.base import (DEFAULT_CYCLE_LENGTH, MIN_CYCLE_LENGTH, SourceFrame,
                             SourceModel, SourceSequence)
from hms.source.cycles import (CycleSet, extract_cycles, impulsiveness,
                               noise_level, pick_epochs, place_cycles,
                               resample_cycle)
from hms.source.generic import GenericResidualSourceModel
from hms.source.pca import DEFAULT_N_COMPONENTS, SourcePCA
from hms.source.residual import inverse_filter, spectral_envelope, whiten
from hms.source.voice import VoiceSourceModel, estimate_f0, smooth_f0

#: Registered backends, in the order `get_source_model` considers them.
SOURCE_BACKENDS = ("voice", "residual")

_MODELS = {"voice": VoiceSourceModel, "residual": GenericResidualSourceModel}

__all__ = [
    "SourceFrame", "SourceSequence", "SourceModel", "SourcePCA", "CycleSet",
    "VoiceSourceModel", "GenericResidualSourceModel", "get_source_model",
    "available_backends", "SOURCE_BACKENDS", "DEFAULT_CYCLE_LENGTH",
    "MIN_CYCLE_LENGTH", "DEFAULT_N_COMPONENTS", "pick_epochs", "extract_cycles",
    "resample_cycle", "place_cycles", "noise_level", "impulsiveness", "whiten",
    "inverse_filter", "spectral_envelope", "estimate_f0", "smooth_f0",
]


def available_backends() -> Dict[str, bool]:
    """Which source backends can be constructed on this machine.

    Every backend is pure NumPy, so all of them are always available; the
    function exists so that `hms doctor`-style reporting, diagnostics and tests
    do not have to hard-code the list.
    """
    return {name: True for name in SOURCE_BACKENDS}


def get_source_model(name: str = "voice", **kwargs) -> SourceModel:
    """Build a source backend by name.

    ``"voice"`` is the pitch-synchronous residual backend, ``"residual"`` the
    frame-synchronous one.  Both take the shared ``SourceModel`` keyword
    arguments (``cycle_length``, ``fs``, ``frame_period``, ``n_mcep``); unknown
    names raise rather than silently falling back, because a source backend is
    an explicit analysis choice, not a capability probe.
    """
    if name not in _MODELS:
        raise ValueError(f"unknown source backend {name!r}; expected one of "
                         f"{SOURCE_BACKENDS}")
    return _MODELS[name](**kwargs)


def source_model_for(kind: Optional[str] = None, **kwargs) -> SourceModel:
    """Default backend helper: ``None`` -> the voice backend."""
    return get_source_model(kind or "voice", **kwargs)
