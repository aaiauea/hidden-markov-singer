"""MLSA (Mel Log Spectrum Approximation) synthesis backend.

This is the second pure-numpy backend next to :mod:`hms.vocoder.builtin`.  It
consumes exactly the same acoustic representation as every other backend --
``f0`` (Hz, 0 = unvoiced), ``sp`` (linear power spectral envelope on the WORLD
grid) and ``ap`` (aperiodicity in [0, 1]) -- and turns it into a waveform with
the classical MLSA synthesis filter instead of a zero-phase magnitude filter.

Formulation
-----------
The mel-cepstrum HMS models *is* the log-power envelope on a mel-warped
frequency axis: :func:`hms.core.features.power_to_mcep` writes the DCT-II of
the log mel-band densities and :func:`hms.core.features.mcep_to_power` reads it
back by interpolating between band centres.  The same coefficients are a
truncated Fourier series on that warped axis, so they can be evaluated at any
frequency -- not only at the analysis knots -- which is what the MLSA filter
needs::

    H(z) = exp( sum_{m=0}^{M-1} a_m * z~^{-m} ),   z~^{-1} = (z^{-1} - a)/(1 - a z^{-1})

with ``a`` the mel-warping (all-pass) coefficient, ``z~^{-1}`` the first-order
all-pass that maps the linear frequency axis onto the mel-like warped one, and
``a_m`` the log-*amplitude* coefficients (half the log-power ones).  Evaluating
the same series on the unit circle gives the frequency response used here::

    H(e^{jw}) = exp( sum_m a_m * e^{-j m beta(w)} ),
    beta(w)   = w + 2 atan2(a sin w, 1 - a cos w)

``beta`` is the warped (mel-like) frequency, so ``|H|^2`` is the smooth
interpolation of the modelled log-power envelope across mel.  At the analysis
knots this reproduces :func:`mcep_to_power` exactly (up to the usual
DCT-II/Kronecker scaling); see ``mlsa_log_amplitude``.

Implementation choices
----------------------
* **Exact response, no ladder recursion.**  SPTK realizes ``H(z)`` with an
  Imai ladder / continued fraction that is exact to a configurable order but
  costs ``O(M)`` multiply-adds *per sample* in a sequential recursion.  Here
  the same transfer function is evaluated exactly on an FFT grid and realised
  as a per-frame FIR (its impulse response) -- vectorised over frames, with no
  filter-state or stability concerns.  The truncated FIR length is documented
  as :data:`FILTER_PERIODS` frame periods (>= 128 samples); the impulse response
  of the warp decays like ``a^n`` (a <= 0.6 here), so the cut-off error is
  below -100 dB at the default length.
* **Mixed excitation.**  Voiced frames are driven by a band-limited pulse train
  (fractional pulse positions, unit mean square at any F0) plus white noise;
  the two are mixed *per frequency bin* with ``sqrt(1 - ap)`` / ``sqrt(ap)``,
  exactly the convention WORLD's synthesis uses.  Unvoiced frames (``f0 <= 0``)
  are pure noise, independent of ``ap``.
* **No peak normalisation.**  Like the WORLD backends (and unlike the builtin
  fallback, whose absolute level is arbitrary), the waveform is returned as
  produced; :func:`hms.data.wavio.write_wav` applies the headroom.  The
  excitation is calibrated so the output power tracks ``sp`` (unit-variance
  excitation, unit-mean-square pulses), so no gain constant is needed.

Limitations
-----------
* Frame-wise frozen coefficients: like every frame-based vocoder, the filter is
  held constant inside a frame and cross-faded by the overlap-add, so a
  parameter change faster than one frame is smoothed over the filter length.
* The warping coefficient follows the project's HTK-style mel scale by a
  least-squares fit (:func:`mel_warping_factor`); the mel axis is not exactly a
  first-order all-pass image, so the warped evaluation differs from
  ``mcep_to_power``'s piecewise-linear interpolation between knots by a
  sub-band amount (well under a dB for the orders HMS uses).
* Analysis is the shared pure-numpy estimator set (see `BuiltinVocoder`); MLSA
  is a *synthesis* backend.  Extraction/training with ``--vocoder mlsa``
  therefore produces the same parameters as ``--vocoder builtin``.
* No separate excitation phase model: pulses are re-locked to the frame grid at
  the start of every voiced run, as the builtin backend does.
"""

from __future__ import annotations

import functools
from typing import Tuple

import numpy as np

from hms.core.features import (DEFAULT_N_MCEP, AcousticFrameSequence,
                               hz_to_mel, mel_band_count, power_to_mcep)
from hms.vocoder.base import Vocoder
from hms.vocoder.builtin import BuiltinVocoder

#: The reconstructed log power is clipped to +/- this value, mirroring the
#: envelope guard in `BuiltinVocoder`; it keeps a pathological ``sp`` (NaN,
#: huge values) from overflowing the response.
LOG_POWER_CLIP = 40.0

#: Default FIR length as a multiple of the frame shift (hop).  Two frame
#: periods (10 ms at the default 5 ms) capture > 99.9999 % of the MLSA impulse
#: response energy for the mel-warping coefficients HMS uses, and the length
#: scales with the sample rate / frame period instead of being a fixed count.
FILTER_PERIODS = 2

#: Half-width of the band-limited pulse kernel, in samples.
PULSE_HALF_WIDTH = 16

#: Pulse-kernel bandwidth relative to Nyquist.  The 10 % above it are covered
#: by the noise component (aperiodicity is high there in real voices anyway).
PULSE_CUTOFF = 0.9

#: Number of frequency points used to fit the mel-warping coefficient.
_WARP_SAMPLES = 257
_WARP_COARSE = np.arange(0.05, 0.9001, 0.02)
_WARP_REFINE = np.arange(-0.02, 0.0201, 0.001)


def _next_power_of_two(n: int) -> int:
    return 1 << max(1, int(np.ceil(np.log2(max(int(n), 2)))))


def _finite(values: np.ndarray, fill: float) -> np.ndarray:
    """Replace NaN/inf in ``values`` with ``fill`` (no copy when already finite).

    The returned array may be the input (a view of the caller's data), so
    callers must not write to it in place.
    """
    if np.isfinite(values).all():
        return values
    return np.where(np.isfinite(values), values, fill)


@functools.lru_cache(maxsize=None)
def mel_warping_factor(fs: int) -> float:
    """All-pass coefficient ``a`` whose warped axis best matches the mel scale.

    The first-order all-pass of the MLSA filter maps the linear frequency axis
    onto ``beta(w) = w + 2 atan2(a sin w, 1 - a cos w)``, and the mel-cepstrum
    lives on that warped axis.  There is no closed form for the ``a`` that best
    matches the project's HTK-style mel scale (``1127.01 ln(1 + f/700)``), so it
    is fitted by least squares on a fixed frequency grid -- deterministic, and
    close to the conventional SPTK/HTS values (0.42 at 16 kHz, 0.55 at 48 kHz).

    The fit depends only on the sample rate; results are cached.
    """
    f = np.linspace(0.0, fs / 2.0, _WARP_SAMPLES)
    omega = 2.0 * np.pi * f / fs
    mel = hz_to_mel(f)
    target = np.pi * mel / mel[-1]

    def error(alpha: float) -> float:
        warped = omega + 2.0 * np.arctan2(alpha * np.sin(omega),
                                          1.0 - alpha * np.cos(omega))
        return float(np.mean((warped - target) ** 2))

    best = min((float(a) for a in _WARP_COARSE), key=error)
    best = min((best + float(d) for d in _WARP_REFINE), key=error)
    return float(np.clip(best, 0.0, 0.9))


def mlsa_log_amplitude(sp: np.ndarray, fft_size: int, fs: int,
                       n_mcep: int = DEFAULT_N_MCEP) -> np.ndarray:
    """Linear power envelope -> log-amplitude coefficients on the warped axis.

    ``power_to_mcep`` returns DCT-II coefficients of the log mel-band powers;
    the MLSA filter wants the Fourier coefficients of the same curve on the
    warped axis.  The two differ only by the (orthonormal) DCT normalisation,
    so the conversion is a per-coefficient scaling -- evaluating the Fourier
    series at the analysis knots then reproduces ``mcep_to_power`` exactly.

    Returns ``(T, n_mcep)``: ``0.5 *`` the log-power Fourier coefficients.
    """
    sp = np.atleast_2d(np.asarray(sp, dtype=np.float64))
    mcep = power_to_mcep(sp, fft_size, fs, n_mcep)
    n_bands = mel_band_count(n_mcep)
    scale = np.full(n_mcep, np.sqrt(2.0 / n_bands), dtype=np.float64)
    scale[0] = 1.0 / np.sqrt(n_bands)
    return 0.5 * mcep * scale


def mlsa_response(log_amplitude: np.ndarray, n_fft: int,
                  alpha: float) -> np.ndarray:
    """Sample the MLSA transfer function on an ``n_fft``-point FFT grid.

    ``log_amplitude`` is ``(T, M)`` (see :func:`mlsa_log_amplitude`); the result
    is the one-sided complex response ``(T, n_fft // 2 + 1)`` of

        ``H(e^{jw}) = exp( sum_m a_m e^{-j m beta(w)} )``.

    The log power (twice the log amplitude) is clipped to
    ``+/- LOG_POWER_CLIP`` before exponentiating, so a malformed envelope
    cannot overflow.
    """
    log_amplitude = np.atleast_2d(np.asarray(log_amplitude, dtype=np.float64))
    n_fft = int(n_fft)
    bins = n_fft // 2 + 1
    omega = 2.0 * np.pi * np.arange(bins) / n_fft
    beta = omega + 2.0 * np.arctan2(alpha * np.sin(omega),
                                    1.0 - alpha * np.cos(omega))
    m = np.arange(log_amplitude.shape[1], dtype=np.float64)
    angles = m[:, None] * beta[None, :]
    log_power = np.clip(2.0 * (log_amplitude @ np.cos(angles)),
                        -LOG_POWER_CLIP, LOG_POWER_CLIP)
    phase = -(log_amplitude @ np.sin(angles))
    magnitude = np.exp(0.5 * log_power)
    return magnitude * np.cos(phase) + 1j * (magnitude * np.sin(phase))


def _pulse_kernel(half_width: int = PULSE_HALF_WIDTH,
                  cutoff: float = PULSE_CUTOFF) -> np.ndarray:
    """Hann-windowed sinc pulse: a band-limited impulse of unit height.

    ``sinc`` peaks at 1 and the Hann window is 1 at the centre, so the kernel's
    maximum is exactly 1; its energy (``sum(kernel ** 2)``) enters
    :func:`_pulse_energy`, which `MLSAVocoder._pulse_train` uses to give the
    train unit mean square.
    """
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    return np.sinc(cutoff * offsets) * np.hanning(2 * half_width + 1)


def _pulse_energy(kernel: np.ndarray, frac: float) -> float:
    """Energy of one pulse placed at a fractional (``frac``) sample offset.

    The fractional shift is a linear interpolation between two integer shifts,
    so the placed pulse is ``(1 - frac) * g[n] + frac * g[n - 1]`` and its
    energy is ``((1-f)^2 + f^2) * sum(g^2) + 2 f (1-f) * sum(g g_shifted)``.
    """
    energy = float(np.sum(kernel ** 2))
    correlation = float(np.sum(kernel[1:] * kernel[:-1]))
    return (1.0 - frac) ** 2 * energy + frac ** 2 * energy \
        + 2.0 * frac * (1.0 - frac) * correlation


def _resample_frequency(values: np.ndarray, out_bins: int) -> np.ndarray:
    """Linear interpolation along the last (frequency) axis."""
    values = np.asarray(values, dtype=np.float64)
    in_bins = values.shape[-1]
    if in_bins == out_bins:
        return values
    if in_bins == 1:
        return np.repeat(values, out_bins, axis=-1)
    position = np.linspace(0.0, in_bins - 1.0, out_bins)
    lower = np.clip(np.floor(position).astype(np.intp), 0, in_bins - 2)
    frac = position - lower
    return values[..., lower] * (1.0 - frac) + values[..., lower + 1] * frac


class MLSAVocoder(Vocoder):
    """Mel Log Spectrum Approximation synthesis (pure numpy).

    See the module docstring for the formulation and its limitations.  The
    backend is selected with ``vocoder="mlsa"`` / ``hms synth --vocoder mlsa``;
    analysis is delegated to the shared pure-numpy estimators so that
    ``extract``/``train`` behave exactly like ``--vocoder builtin``.
    """

    name = "mlsa"

    #: Used when the caller does not pin an FFT size (mirrors the WORLD default).
    DEFAULT_FFT_SIZE = 2048

    #: Frames per vectorised synthesis block: bounds the working set (the
    #: batched FFTs and the excitation gather) without making the blocks so
    #: small that the FFT calls dominate.
    _block = 256

    #: Smallest pulse period the excitation generator will place, in samples.
    #: It only matters for an F0 at or above Nyquist, which cannot be rendered.
    _MIN_PERIOD = 2.0

    def __init__(self, fft_size: int | None = None, fs: int = 44100,
                 frame_period: float = 5.0, n_mcep: int = DEFAULT_N_MCEP,
                 alpha: float | None = None, filter_length: int | None = None,
                 seed: int = 12345, pulse_half_width: int = PULSE_HALF_WIDTH,
                 pulse_cutoff: float = PULSE_CUTOFF) -> None:
        super().__init__(fft_size=int(fft_size or self.DEFAULT_FFT_SIZE),
                         fs=fs, frame_period=frame_period)
        if n_mcep < 2:
            raise ValueError("n_mcep must be >= 2")
        if alpha is not None and not 0.0 <= float(alpha) < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if filter_length is not None and int(filter_length) < 2:
            raise ValueError("filter_length must be >= 2")
        self.n_mcep = int(n_mcep)
        self.alpha = None if alpha is None else float(alpha)
        self.filter_length = None if filter_length is None \
            else int(filter_length)
        self.seed = int(seed)
        self.pulse_half_width = int(pulse_half_width)
        self.pulse_cutoff = float(pulse_cutoff)
        # Analysis is deliberately shared with the builtin backend: HMS's
        # frame geometry and (f0, sp, ap) conventions are defined once.
        self._analyzer = BuiltinVocoder(fft_size=fft_size, fs=fs,
                                        frame_period=frame_period)

    @property
    def fft_size(self) -> int:
        return int(self._fft_size)

    # -- analysis ----------------------------------------------------------

    def analyze(self, x: np.ndarray, fs: int | None = None,
                frame_period: float | None = None, f0_floor: float = 71.0,
                f0_ceil: float = 800.0, f0_estimation: str = "dio",
                refine_f0: bool = True) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
        """Waveform -> (f0, sp, ap), using the shared numpy estimators.

        MLSA is a synthesis filter; its analysis is the same autocorrelation /
        cepstral / harmonicity estimator set the builtin backend uses, so
        extracted parameters are interchangeable between the two backends.
        """
        return self._analyzer.analyze(x, fs=fs, frame_period=frame_period,
                                      f0_floor=f0_floor, f0_ceil=f0_ceil,
                                      f0_estimation=f0_estimation,
                                      refine_f0=refine_f0)

    # -- synthesis ---------------------------------------------------------

    def synthesize(self, params: AcousticFrameSequence) -> np.ndarray:
        f0 = np.asarray(params.f0, dtype=np.float64).reshape(-1)
        n_frames = int(f0.size)
        if n_frames == 0:
            return np.zeros(0, dtype=np.float64)
        fs = int(params.fs or self.default_fs)
        frame_period = float(params.frame_period or self.default_frame_period)
        hop = max(1, int(round(fs * frame_period / 1000.0)))
        fft_size = int(params.fft_size or self.fft_size)
        bins = fft_size // 2 + 1

        # Malformed parameters must not reach the filter as NaN/Inf.  The
        # common case is finite, so check first and only allocate when the
        # data really needs repairing.
        f0 = _finite(f0, 0.0)
        if np.any(f0 < 0.0):
            f0 = np.maximum(f0, 0.0)
        sp = np.atleast_2d(np.asarray(params.sp, dtype=np.float64))[:, :bins]
        ap = np.atleast_2d(np.asarray(params.ap, dtype=np.float64))[:, :bins]
        if sp.shape[0] != n_frames or ap.shape[0] != n_frames:
            raise ValueError(
                "f0, sp and ap must describe the same frames, got "
                f"{n_frames}, {sp.shape[0]} and {ap.shape[0]}")
        sp = _finite(sp, 0.0)
        if np.any(sp < 0.0):
            sp = np.maximum(sp, 0.0)
        ap = _finite(ap, 1.0)
        if np.any(ap < 0.0) or np.any(ap > 1.0):
            ap = np.clip(ap, 0.0, 1.0)
        # An unvoiced frame is pure noise, whatever aperiodicity the model
        # emitted for it; a voiced frame mixes pulse and noise per band.
        unvoiced = f0 <= 0.0
        if unvoiced.any():
            ap = np.where(unvoiced[:, None], 1.0, ap)

        # WORLD's duration convention (`f0_length * frame_period * fs`), which
        # every backend must reproduce so a phrase's length is backend-independent.
        y_length = int(n_frames * frame_period / 1000.0 * fs)
        filter_length = self.filter_length or _next_power_of_two(
            FILTER_PERIODS * hop)
        n_fft = _next_power_of_two(hop + filter_length)
        filter_bins = n_fft // 2 + 1

        alpha = self.alpha if self.alpha is not None \
            else mel_warping_factor(fs)
        # (T, n_mcep) -- small; the FFT-grid response and the per-bin weights
        # are built per block below so a long utterance does not hold a
        # (T, filter_bins) complex filter in memory.
        log_amplitude = mlsa_log_amplitude(sp, fft_size, fs, self.n_mcep)

        n_exc = max(n_frames * hop, y_length) + n_fft
        # The frame grid advances by `hop` samples while the rendered length is
        # the exact `n_frames * frame_period * fs`; where the two disagree (a
        # 5 ms hop is 220.5 samples at 44.1 kHz) the excitation must still
        # cover every output sample, so it is generated up to the last sample
        # that can reach the output, with the last frame's F0 replicated (the
        # same edge convention `add_dynamic_features` uses).
        pulse = self._pulse_train(f0, fs, hop, n_exc,
                                  int(min(y_length + filter_length,
                                          n_exc - 1)))
        noise = np.random.default_rng(self.seed).standard_normal(n_exc)

        y = np.zeros(n_exc + n_fft, dtype=np.float64)
        offsets = np.arange(hop)[None, :] + hop * np.arange(
            min(self._block, n_frames))[:, None]
        # A frame's contribution is the linear convolution of its excitation
        # chunk (one hop) with the MLSA impulse response (the filter's FFT
        # response) -- exactly the time-varying filter, overlap-added.  A
        # constant filter reproduces the true convolution sample for sample.
        has_pulse = bool(pulse.any())
        for start in range(0, n_frames, self._block):
            stop = min(start + self._block, n_frames)
            index = offsets[: stop - start] + start * hop
            response = mlsa_response(log_amplitude[start:stop], n_fft, alpha)
            weights = _resample_frequency(ap[start:stop], filter_bins)
            spectrum = np.fft.rfft(noise[index], n_fft, axis=1)
            spectrum *= np.sqrt(weights)
            if has_pulse:
                spectrum += (np.fft.rfft(pulse[index], n_fft, axis=1)
                             * np.sqrt(np.maximum(1.0 - weights, 0.0)))
            spectrum *= response
            segment = np.fft.irfft(spectrum, n_fft, axis=1)
            for i in range(stop - start):
                position = (start + i) * hop
                y[position: position + n_fft] += segment[i]
        return y[:max(y_length, 1)]

    def _pulse_train(self, f0: np.ndarray, fs: int, hop: int, n_exc: int,
                     pulse_end: int) -> np.ndarray:
        """Band-limited pulse train with unit mean square at any F0.

        Pulse positions are tracked exactly (not accumulated from rounded
        samples); each pulse is added through the fractional-shift kernel, so
        the train stays periodic at any F0 instead of acquiring sample jitter.
        The amplitude makes each pulse's energy exactly the period (so the
        train's mean square is 1 regardless of pitch, which is what keeps the
        output level tied to ``sp`` and not to the note); the linear fractional
        shift splits that energy between two samples, which is accounted for.

        Pulse generation stops at ``pulse_end``: a run that reaches the last
        frame is extended to it with the last frame's F0, so the trailing
        samples of a render (where the sample-exact length overtakes the frame
        grid) keep their periodic excitation.
        """
        pulse = np.zeros(n_exc, dtype=np.float64)
        voiced = f0 > 0.0
        if not voiced.any():
            return pulse
        kernel = _pulse_kernel(self.pulse_half_width, self.pulse_cutoff)
        half = self.pulse_half_width
        n_frames = len(f0)
        last = min(int(pulse_end), n_exc - kernel.size)

        # Runs of voiced frames, as half-open [start, stop) frame index pairs.
        edges = np.flatnonzero(np.diff(np.concatenate(
            ([0], voiced.view(np.int8), [0]))))
        for start, stop in zip(edges[0::2], edges[1::2]):
            t0, t1 = int(start), int(stop)
            position = float(t0 * hop)
            end = float(last if t1 >= n_frames else min(t1 * hop, last))
            while position < end:
                frame = min(int(position // hop), t1 - 1)
                frequency = float(f0[frame])
                if frequency <= 0.0:
                    break
                lower = int(np.floor(position))
                frac = position - lower
                lo = lower - half
                hi = lo + kernel.size
                if lo >= 0 and hi + 1 <= n_exc:
                    # scale so that each pulse carries exactly one period's
                    # worth of energy whatever its fractional offset is
                    placed = _pulse_energy(kernel, frac)
                    amplitude = np.sqrt(fs / frequency / max(placed, 1e-12))
                    pulse[lo:hi] += amplitude * (1.0 - frac) * kernel
                    pulse[lo + 1:hi + 1] += amplitude * frac * kernel
                # period in samples; `_MIN_PERIOD` guards an F0 at/above Nyquist
                position += max(fs / frequency, self._MIN_PERIOD)
        return pulse
