"""Vocoder backends.

HMS generates WORLD parameters (f0, sp, ap); a backend turns them into audio.
`get_vocoder()` picks the best available backend:

    pyworld  ->  native libhms_world.so (ctypes)  ->  builtin fallback

Only the pyworld/native backends are "real" WORLD; `builtin` and `mlsa` are the
two pure-numpy backends that let the rest of the system run on a machine without
a compiler.  Both implement the same `Vocoder` interface, so swapping them
changes nothing else.  `mlsa` is the MLSA (mel log spectrum approximation)
synthesis filter and is opt-in: it never changes what `auto` resolves to.
"""

from __future__ import annotations

from typing import Optional

from hms.vocoder.base import Vocoder, VocoderUnavailable
from hms.vocoder.builtin import BuiltinVocoder
from hms.vocoder.mlsa import MLSAVocoder

_BACKENDS = ("auto", "pyworld", "native", "builtin", "mlsa")

__all__ = ["Vocoder", "VocoderUnavailable", "BuiltinVocoder", "MLSAVocoder",
           "get_vocoder", "available_backends", "BACKENDS"]

BACKENDS = _BACKENDS


def available_backends() -> dict:
    """Report which backends can be constructed on this machine."""
    status = {"pyworld": False, "native": False, "builtin": True, "mlsa": True}
    try:
        import pyworld  # noqa: F401
        status["pyworld"] = True
    except Exception:
        pass
    try:
        from hms.vocoder.world_native import NativeWorldVocoder
        NativeWorldVocoder()
        status["native"] = True
    except Exception:
        pass
    return status


def get_vocoder(name: str = "auto", fft_size: Optional[int] = None,
                **kwargs) -> Vocoder:
    """Build a vocoder backend by name, falling back gracefully.

    ``auto`` keeps its historical precedence (pyworld, then native, then
    builtin); the MLSA backend is opt-in via ``name="mlsa"`` so that adding it
    cannot silently change what an existing configuration renders with.
    """
    if name not in _BACKENDS:
        raise ValueError(f"unknown vocoder {name!r}; expected one of {_BACKENDS}")

    if name == "mlsa":
        return MLSAVocoder(fft_size=fft_size, **kwargs)

    if name in ("auto", "pyworld"):
        try:
            from hms.vocoder.pyworld_backend import PyWorldVocoder
            return PyWorldVocoder(fft_size=fft_size, **kwargs)
        except VocoderUnavailable:
            if name == "pyworld":
                raise

    if name in ("auto", "native"):
        try:
            from hms.vocoder.world_native import NativeWorldVocoder
            return NativeWorldVocoder(fft_size=fft_size, **kwargs)
        except VocoderUnavailable:
            if name == "native":
                raise

    return BuiltinVocoder(fft_size=fft_size, **kwargs)
