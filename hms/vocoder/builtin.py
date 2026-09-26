"""Pure-numpy fallback vocoder.

This is **not** WORLD.  It exists so that `hms extract / train / synth` still run
on a machine without a C++ compiler, and so the vocoder interface itself can be
unit tested.  It keeps exactly the same parameter conventions as the WORLD
backends, so the rest of HMS cannot tell the difference:

    f0   (T,)      Hz, 0.0 = unvoiced
    sp   (T, F+1)  linear power spectral envelope
    ap   (T, F+1)  aperiodicity in [0, 1]

Analysis
    F0        normalised autocorrelation + cumulative-mean octave control
    sp        cepstrally liftered power envelope
    ap        per-mel-band harmonicity (see `hms.core.dsp`)

Synthesis
    pulse/noise excitation (mixed by aperiodicity) filtered by the spectral
    envelope with STFT overlap-add.  WORLD applies a minimum-phase impulse
    response per frame; this uses a zero-phase magnitude response instead,
    which is simpler and audibly close for smooth envelopes, but it is the main
    reason this backend is a fallback rather than the default.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from hms.core.dsp import (autocorrelation_f0, frame_signal,
                          harmonicity_aperiodicity, spectral_flatness)
from hms.core.features import AcousticFrameSequence
from hms.vocoder.base import Vocoder, limit_peak


class BuiltinVocoder(Vocoder):
    """Approximate pure-numpy analysis/synthesis backend."""

    name = "builtin"

    #: Used when the caller does not pin an FFT size (mirrors the WORLD default).
    DEFAULT_FFT_SIZE = 2048
    #: Frames flatter than this are treated as unvoiced regardless of the
    #: autocorrelation peak (white noise has flatness ~0.5, vowels ~0.0).
    FLATNESS_THRESHOLD = 0.30

    def __init__(self, fft_size: int | None = None, fs: int = 44100,
                 frame_period: float = 5.0, cepstral_order: int = 40,
                 n_ap_band: int = 5, seed: int = 12345) -> None:
        super().__init__(fft_size=int(fft_size or self.DEFAULT_FFT_SIZE),
                         fs=fs, frame_period=frame_period)
        self._fft_size_pinned = fft_size is not None
        self.cepstral_order = cepstral_order
        self.n_ap_band = n_ap_band
        self.seed = seed

    @property
    def fft_size(self) -> int:
        return int(self._fft_size)

    # -- analysis ----------------------------------------------------------

    def analyze(self, x: np.ndarray, fs: int | None = None,
                frame_period: float | None = None, f0_floor: float = 71.0,
                f0_ceil: float = 800.0, f0_estimation: str = "dio",
                refine_f0: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fs = int(fs or self.default_fs)
        frame_period = float(frame_period or self.default_frame_period)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        hop = max(1, int(round(fs * frame_period / 1000.0)))
        frame_length = 4 * hop

        frames = frame_signal(x, self.fft_size, hop) * np.hanning(self.fft_size)
        power = np.abs(np.fft.rfft(frames, self.fft_size, axis=1)) ** 2

        f0 = autocorrelation_f0(x, fs, frame_length, hop, f0_floor, f0_ceil)
        n_frames = min(len(f0), len(frames))
        f0, power = f0[:n_frames], power[:n_frames]
        # a peak in the autocorrelation is not enough: noise has one too
        flat = spectral_flatness(power)
        f0 = np.where(flat < self.FLATNESS_THRESHOLD, f0, 0.0)

        sp = self._lifter(power)
        _, ap = harmonicity_aperiodicity(power, f0, fs, self.n_ap_band)
        return f0, sp, ap

    def _lifter(self, power: np.ndarray) -> np.ndarray:
        """Cepstral liftering -> smooth power envelope, floored at 1e-12."""
        log_spec = np.log(np.maximum(power, 1e-12))
        cep = np.fft.irfft(log_spec, axis=1)
        order = min(self.cepstral_order, cep.shape[1] // 2 - 1)
        cep[:, order + 1: cep.shape[1] - order - 1] = 0.0
        env = np.fft.rfft(cep, axis=1).real
        return np.exp(np.clip(env, -40.0, 40.0))

    # -- synthesis ---------------------------------------------------------

    def synthesize(self, params: AcousticFrameSequence) -> np.ndarray:
        f0 = np.asarray(params.f0, dtype=np.float64).reshape(-1)
        n_frames = len(f0)
        if n_frames == 0:
            return np.zeros(0, dtype=np.float64)
        fs = int(params.fs or self.default_fs)
        frame_period = float(params.frame_period or self.default_frame_period)
        hop = max(1, int(round(fs * frame_period / 1000.0)))
        fft_size = int(params.fft_size or self.fft_size)
        bins = fft_size // 2 + 1
        sp = np.asarray(params.sp, dtype=np.float64)[:, :bins]
        ap = np.asarray(params.ap, dtype=np.float64)[:, :bins]

        # Synthesise one window longer than the output so the last frame's
        # overlap-add has somewhere to go, then trim to WORLD's length
        # convention (`f0_length * frame_period * fs`).  Renderers must not
        # change a phrase's duration just because the backend changed.
        tail = int((n_frames - 1) * hop + fft_size)
        y_length = int(n_frames * frame_period / 1000.0 * fs)
        excitation = self._excitation(f0, ap, fs, hop, tail)
        y = self._apply_envelope(excitation, sp, fft_size, hop, tail)
        y = y[:max(y_length, 1)]
        # This backend builds its excitation from scratch, so its absolute
        # level is arbitrary: scale it once, via the shared policy, instead of
        # leaving callers to guess.  WORLD-backed backends return their
        # synthesis verbatim (see `Vocoder.synthesize`).
        return limit_peak(y, ceiling=0.99, headroom=1.0)

    def _excitation(self, f0: np.ndarray, ap: np.ndarray, fs: int, hop: int,
                    y_length: int) -> np.ndarray:
        """Pulse train + high-passed noise, mixed per frame by aperiodicity."""
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal(y_length)
        # cheap high-pass: aperiodicity rises with frequency in real voices, so
        # the noise component should not be flat down to DC
        noise = noise - 0.3 * np.concatenate([[0.0], noise[:-1]])

        pulse = np.zeros(y_length)
        voiced = f0[f0 > 0]
        if voiced.size and float(np.min(ap)) < 1.0:
            # a pulse narrower than the shortest period (and never wider than
            # one frame) -- the shape only needs to be a rough glottal pulse
            width = max(1, min(int(round(0.25 * fs / float(voiced.max()))), hop))
            shape = np.hanning(2 * width + 1)
            shape /= shape.max()
            position = float(np.argmax(f0 > 0) * hop)   # absolute pulse phase
            for t, f in enumerate(f0):
                frame_end = min((t + 1) * hop, y_length)
                if f <= 0:
                    continue
                if position < t * hop:                  # new voiced note: re-lock
                    position = float(t * hop)
                period = fs / f
                while position < frame_end:
                    centre = int(round(position))
                    lo, hi = centre - width, centre + width + 1
                    dst_lo, dst_hi = max(0, lo), min(y_length, hi)
                    if dst_hi > dst_lo:
                        src_lo = dst_lo - lo
                        pulse[dst_lo:dst_hi] += shape[src_lo:src_lo + dst_hi
                                                      - dst_lo]
                    position += period

        noise_ratio = np.clip(ap.mean(axis=1), 0.0, 1.0)
        pulse_gain = np.zeros(y_length)
        noise_gain = np.zeros(y_length)
        for t in range(len(f0)):
            start, end = t * hop, min((t + 1) * hop, y_length)
            if start >= y_length:
                break
            pulse_gain[start:end] = np.sqrt(1.0 - noise_ratio[t])
            noise_gain[start:end] = np.sqrt(noise_ratio[t])
        return pulse * pulse_gain + noise * noise_gain * 0.6

    def _apply_envelope(self, excitation: np.ndarray, sp: np.ndarray,
                        fft_size: int, hop: int, y_length: int) -> np.ndarray:
        """Overlap-add filtering of the excitation by sqrt(sp) per frame."""
        window = np.hanning(fft_size)
        y = np.zeros(y_length + fft_size)
        norm = np.zeros(y_length + fft_size)
        padded = np.pad(excitation, (0, max(0, y_length + fft_size
                                            - len(excitation))))
        for t in range(len(sp)):
            start = t * hop
            segment = padded[start:start + fft_size]
            spec = np.fft.rfft(segment * window)
            spec *= np.sqrt(np.maximum(sp[t], 1e-12))
            y[start:start + fft_size] += np.fft.irfft(spec, fft_size)
            norm[start:start + fft_size] += window ** 2
        return y[:y_length] / np.maximum(norm[:y_length], 1e-6)
