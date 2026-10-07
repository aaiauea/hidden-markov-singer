"""Inverse filtering: audio -> excitation residual.

The source of a sustained sound is heard through a filter (a vocal tract, a
bore, a body).  To get at the source, the filter has to be divided out.  HMS
already owns a compact description of that filter -- the mel-cepstrum envelope
:func:`hms.core.features.power_to_mcep` / :func:`mcep_to_power` -- so this module
uses exactly that to whiten a signal, with no new dependency and no new
envelope definition to keep in sync with the acoustic model::

    audio ──► STFT (Hann, hop) ──► |X|² ──► mel-cepstrum envelope
                                                  │
                            ┌─────────────────────┘
                            ▼
                     X / sqrt(envelope) ──► overlap-add ──► residual

Design choices
--------------
* **Shape, not level, is inverted.**  The envelope is normalised to unit mean
  per frame before dividing, so the residual keeps the frame's overall energy
  and only loses its spectral colour.  This keeps the residual's level in the
  same units as the audio (so per-cycle ``gain`` means something), and it cannot
  explode on a near-silent frame where an absolute envelope estimate is
  meaningless.
* **Zero-phase division.**  Dividing the (rFFT) spectrum by the real
  ``sqrt(envelope)`` is a zero-phase inverse filter.  It is not a minimum-phase
  one, so the residual is not sample-aligned with the true glottal flow
  derivative; it is *consistently* offset, which is all a pitch-synchronous
  cycle extractor and a PCA basis need.  A minimum-phase implentation would need
  a cepstral recursion or a Hilbert transform for no measurable gain here.
* **A short window.**  The analysis window defaults to ``4 * hop`` (20 ms at the
  default frame period), rounded to a power of two -- short enough to track a
  moving filter, long enough to resolve the F0 harmonics at singing pitches.
  The envelope is a *smooth* mel-cepstrum of order ``n_mcep`` (30 by default),
  which is what keeps the residual free of the envelope's fine structure.
"""

from __future__ import annotations

import numpy as np

from hms.core.dsp import frame_signal
from hms.core.features import DEFAULT_N_MCEP, mcep_to_power, power_to_mcep

#: Default peak gain of the inverse filter (~30 dB of envelope inversion).
#: The per-frame envelope is clipped to this dynamic range before dividing, for
#: two reasons: a degenerate envelope estimate cannot turn a quiet frame into a
#: huge residual, and -- the reason this is a *limit* rather than a huge number
#: -- inverting a spectral valley by more than this mostly amplifies the noise
#: floor, which is measured directly: successive residual cycles of the same note
#: correlate at 0.98 with this default and at 0.62 with a 60 dB limit.  Raise it
#: for a very clean source, lower it for a noisy one.
DEFAULT_GAIN_LIMIT = 30.0


def default_fft_size(fs: int, frame_period: float = 5.0) -> int:
    """A power-of-two analysis window around four frame periods."""
    hop = max(1, int(round(float(fs) * float(frame_period) / 1000.0)))
    size = 4 * hop
    power = 1 << max(1, int(np.ceil(np.log2(max(size, 2)))))
    return int(max(256, power))


def spectral_envelope(power: np.ndarray, fs: int, fft_size: int,
                      n_mcep: int = DEFAULT_N_MCEP, f_min: float = 0.0
                      ) -> np.ndarray:
    """Frame power spectra -> smooth mel-cepstrum envelope, same grid.

    This is the model's own envelope definition, reused verbatim so the source
    analysis removes exactly the thing the acoustic model fits.
    """
    power = np.atleast_2d(np.asarray(power, dtype=np.float64))
    mcep = power_to_mcep(power, fft_size, fs, n_mcep, f_min=f_min)
    return mcep_to_power(mcep, fft_size, fs, n_mcep, f_min=f_min)


def inverse_filter(power: np.ndarray, fs: int, fft_size: int,
                   n_mcep: int = DEFAULT_N_MCEP, f_min: float = 0.0,
                   gain_limit: float = DEFAULT_GAIN_LIMIT,
                   eps: float = 1e-12) -> np.ndarray:
    """Real whitening gain ``1 / sqrt(envelope)`` for each frame and bin.

    The envelope is normalised to unit mean across the bins of each frame
    (inverting shape, not level) and its dynamic range is limited to
    ``gain_limit``; the result is finite, positive and bounded by construction.
    """
    envelope = spectral_envelope(power, fs, fft_size, n_mcep, f_min)
    mean = np.maximum(envelope.mean(axis=1, keepdims=True), eps)
    normalised = np.clip(envelope / mean, 1.0 / gain_limit ** 2, gain_limit ** 2)
    gain = 1.0 / np.sqrt(np.maximum(normalised, eps))
    return np.nan_to_num(gain, nan=0.0, posinf=gain_limit, neginf=0.0)


def whiten(x: np.ndarray, fs: int, frame_period: float = 5.0,
           fft_size: int | None = None, n_mcep: int = DEFAULT_N_MCEP,
           f_min: float = 0.0, gain_limit: float = DEFAULT_GAIN_LIMIT,
           remove_dc: bool = True) -> np.ndarray:
    """Waveform -> excitation residual of the same length.

    An analysis-modification-synthesis loop: Hann-windowed frames, divide by
    ``sqrt(envelope)``, overlap-add, and normalise by the accumulated window so
    the result is an *analysis* signal, not a filtered copy with windowing
    ripple.  The signal's DC offset is removed first (a constant is filter
    energy, not source energy); nothing else about the level is touched.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    fs = int(fs)
    if fft_size is None:
        fft_size = default_fft_size(fs, frame_period)
    fft_size = int(fft_size)
    hop = max(1, int(round(fs * float(frame_period) / 1000.0)))
    if remove_dc and np.isfinite(x).all():
        x = x - float(x.mean())

    window = np.hanning(fft_size)
    frames = np.nan_to_num(frame_signal(x, fft_size, hop) * window)
    spectrum = np.fft.rfft(frames, fft_size, axis=1)
    gain = inverse_filter(np.abs(spectrum) ** 2, fs, fft_size, n_mcep, f_min,
                          gain_limit)
    residual_frames = np.fft.irfft(spectrum * gain, fft_size, axis=1)

    # `frame_signal` pads by half a window, so frame t covers original samples
    # [t * hop - fft_size/2, ...).  Accumulate on a pad-shifted grid and read
    # the middle back out; the norm divides out the overlap-add windowing.
    pad = fft_size // 2
    # The overlap-add grid has to cover every frame: for a signal shorter than
    # one window `frame_signal` zero-pads the input up to a window first, so the
    # frames reach further than `x.size + 2 * fft_size`.
    total = int(max(x.size + 2 * fft_size,
                    (residual_frames.shape[0] - 1) * hop + fft_size))
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    window_sq = window ** 2
    for t in range(residual_frames.shape[0]):
        start = t * hop
        stop = start + fft_size
        out[start:stop] += residual_frames[t]
        norm[start:stop] += window_sq
    residual = out[pad:pad + x.size] / np.maximum(norm[pad:pad + x.size], 1e-12)
    return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = ["whiten", "inverse_filter", "spectral_envelope", "default_fft_size",
           "DEFAULT_GAIN_LIMIT"]
