"""Vocoder backends.

HMS generates WORLD parameters (f0, sp, ap); a backend turns them into audio.
`get_vocoder()` picks the best available backend:

    pyworld  ->  native libhms_world.so (ctypes)  ->  builtin fallback

Only the native WORLD backend is "real" WORLD; the builtin backend exists so
that the rest of the system still runs on a machine without a compiler.  Both
implement the same `Vocoder` interface, so swapping them changes nothing else.
"""

from __future__ import annotations

from typing import Optional

from hms.vocoder.base import Vocoder, VocoderUnavailable
from hms.vocoder.builtin import BuiltinVocoder

_BACKENDS = ("auto", "pyworld", "native", "builtin")

__all__ = ["Vocoder", "VocoderUnavailable", "BuiltinVocoder", "get_vocoder",
           "available_backends", "BACKENDS"]

BACKENDS = _BACKENDS


def available_backends() -> dict:
    """Report which backends can be constructed on this machine."""
    status = {"pyworld": False, "native": False, "builtin": True}
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
    """Build a vocoder backend by name, falling back gracefully."""
    if name not in _BACKENDS:
        raise ValueError(f"unknown vocoder {name!r}; expected one of {_BACKENDS}")

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
