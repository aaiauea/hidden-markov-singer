"""Source/excitation analysis (Phase 1) and standalone prediction (Phase 2).

HMS has traditionally modelled the *filter* (a mel-cepstrum envelope per frame)
and left the *source* to the vocoder. Phase 1 added a generic source-vector
representation, pitch-synchronous and frame-synchronous analyzers, and NumPy
PCA. Phase 2 adds a separate HMM/GMM predictor over the existing PCA
coefficients. It does not change the acoustic trainer, synthesizer or vocoder::

labels + explicit F0 ──► source HMM/GMM ──► PCA coefficients ──► excitation
labels ────────────────► acoustic HMM/GMM ─► spectral envelope / AP ─► filter

* :class:`~hms.source.base.SourceFrame` / :class:`~hms.source.base.SourceSequence`
  -- the generic, backend-independent representation.
* :class:`~hms.source.base.SourceModel` -- the interface a source backend
  implements (``analyze`` plus the shared ``encode``/``decode``/``synthesize``).
* :class:`~hms.source.voice.VoiceSourceModel` and
  :class:`~hms.source.generic.GenericResidualSourceModel` -- the Phase-1
  pitch-synchronous and frame-synchronous backends.
* :class:`~hms.source.pca.SourcePCA` -- NumPy-only PCA over the source vectors.
* :class:`~hms.source.trainer.SourceTrainer` and
  :class:`~hms.source.hmm.SourceHMMModel` -- Phase-2 training, F0-conditioned
  prediction, PCA decoding and separate model save/load.

See ``docs/source_model.md`` for both phases and ``tools/bench_source_pca.py``
for the Phase-1 reconstruction experiment.
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
from hms.source.hmm import (SourceF0Regressor, SourceHMMModel, SourcePrediction,
                            align_source_coefficients, f0_condition_features)
from hms.source.trainer import (SourceTrainer, SourceTrainingConfig,
                                SourceTrainingExample)

#: Registered backends, in the order `get_source_model` considers them.
SOURCE_BACKENDS = ("voice", "residual")

_MODELS = {"voice": VoiceSourceModel, "residual": GenericResidualSourceModel}

__all__ = [
    "SourceFrame", "SourceSequence", "SourceModel", "SourcePCA", "CycleSet",
    "SourceHMMModel", "SourcePrediction", "SourceF0Regressor", "SourceTrainer",
    "SourceTrainingConfig", "SourceTrainingExample", "align_source_coefficients",
    "f0_condition_features", "VoiceSourceModel", "GenericResidualSourceModel",
    "get_source_model",
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
