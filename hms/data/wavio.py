"""Minimal WAV reader/writer (16-bit PCM, 24-bit PCM, 32-bit float, mono/multi).

Deliberately dependency-free: HMS should not need `soundfile`/`libsndfile` to
read a training clip or write a rendered note.  Everything is converted to
float64 in [-1, 1] and mixed down to mono, which is what the analysis front end
wants.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Tuple

import numpy as np


def read_wav(path) -> Tuple[np.ndarray, int]:
    """Read a WAV file -> (mono float64 signal, sample rate)."""
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        fs = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(raw, dtype="<u1").astype(np.float64) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sample_width == 3:
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        value = (buf[:, 0].astype(np.int32)
                 | (buf[:, 1].astype(np.int32) << 8)
                 | (buf[:, 2].astype(np.int32) << 16))
        value = np.where(value >= 1 << 23, value - (1 << 24), value)
        data = value.astype(np.float64) / float(1 << 23)
    else:
        raise ValueError(f"unsupported sample width: {sample_width * 8} bit")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, int(fs)


def write_wav(path, signal: np.ndarray, fs: int, bit_depth: int = 16) -> None:
    """Write a mono float signal as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = np.nan_to_num(np.asarray(signal, dtype=np.float64).reshape(-1))
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1.0:
        signal = signal / (peak * 1.0001)

    if bit_depth == 16:
        payload = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        sample_width = 2
    elif bit_depth == 32:
        payload = np.asarray(signal, dtype="<f4").tobytes()
        sample_width = 4
    else:
        raise ValueError("bit_depth must be 16 or 32")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(sample_width)
        handle.setframerate(int(fs))
        handle.writeframes(payload)


def audio_info(path) -> dict:
    """Header information without decoding the payload."""
    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sample_rate": handle.getframerate(),
            "frames": handle.getnframes(),
            "sample_width": handle.getsampwidth(),
            "duration": handle.getnframes() / max(handle.getframerate(), 1),
        }


def save_params(path, sequence) -> None:
    """Save WORLD parameters as a small ``.npz`` (f0/sp/ap + geometry)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, f0=sequence.f0, sp=sequence.sp, ap=sequence.ap,
                        frame_period=sequence.frame_period, fs=sequence.fs,
                        fft_size=sequence.fft_size)


def load_params(path):
    """Load parameters written by `save_params` into a AcousticFrameSequence."""
    from hms.core.features import AcousticFrameSequence

    with np.load(path) as handle:
        return AcousticFrameSequence(
            f0=handle["f0"], sp=handle["sp"], ap=handle["ap"],
            frame_period=float(handle["frame_period"]),
            fs=int(handle["fs"]), fft_size=int(handle["fft_size"]))


__all__ = ["read_wav", "write_wav", "audio_info", "save_params", "load_params",
           "struct"]
