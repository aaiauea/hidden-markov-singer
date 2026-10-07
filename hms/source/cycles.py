"""Pitch-synchronous cycle extraction and fixed-length normalisation.

This module is the part of the source analysis that a *periodic* source needs:
where its excitation events are, how to cut one cycle out of the residual, how
to bend every cycle onto the same number of samples, and how to put those
vectors back on the sample grid.  It knows nothing about voices -- the F0 track
and the residual it works on are inputs -- so a future glottal, lip-reed or
bowed-string backend can reuse all of it.

Epochs (excitation event positions)
-----------------------------------
The reference points are found by *phase accumulation* rather than by picking
periods independently: inside a voiced run the per-sample period
``fs / f0[n]`` is accumulated and an epoch is emitted whenever the running
phase crosses an integer.  This is what makes the extractor immune to the
things that break naive period picking:

* an **F0 discontinuity** only changes the slope of the phase, it cannot make
  the epochs jump or run backwards;
* an **octave error** in the F0 track produces evenly spaced epochs one octave
  away -- wrong, but a valid, monotone cycle set rather than nonsense;
* **very high / very low F0** is handled by clamping the period into
  ``[fs / f0_ceil, fs / f0_floor]``: a wild value degrades one period's spacing
  instead of producing an empty vector;
* **unvoiced regions** cannot emit epochs at all: each voiced run starts its
  own phase at zero, so a gap never drags a fractional phase into the next run;
* **edges** are handled by dropping anything that does not fit: a cycle is only
  kept when the whole ``[epoch, epoch + period)`` span lies inside the signal.

The epochs are optionally *refined* to the strongest residual sample inside a
window of ``+/- refine_ratio`` periods (bounded by the neighbouring epochs so
the refinement can never cross cycles).  Excitation events are sharp, so this
snaps the cycle boundaries onto them even when the F0 track is coarse.  The
refinement is applied to each voiced run *independently*: an epoch's neighbours,
its local period and the span it may move inside all come from its own run, so
no epoch is ever pulled across a voiced/unvoiced boundary onto an event that
belongs to another run (or to the gap between them).

:func:`pick_epochs` returns the epochs **and the voiced run each belongs to**.  A
cycle is only a cycle when its two epochs come from the same run: the interval
between the last epoch of one run and the first epoch of the next spans an
unvoiced gap, and treating it as a period would produce exactly the kind of
malformed, arbitrarily long "cycle" this module exists to avoid.

Fixed-length normalisation
--------------------------
Cycles are resampled to ``cycle_length`` samples (128 by default) with a
*periodic* band-limited FFT resample -- a cycle is by definition one period of a
periodic signal, so the resample wraps around instead of interpolating between
two ends that were never adjacent.  Resampling is not free: it low-passes the
cycle to ``min(cycle_length, period) / 2`` harmonics per period, so a 128-sample
vector cannot represent source energy above 64 harmonics.  That is the intended
compression (``cycle_length`` is a parameter, not a constant), and it is why the
same length must be used on the way back out (:func:`place_cycles`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

#: Smallest period, in samples, the epoch tracker will emit an epoch for.
MIN_PERIOD_SAMPLES = 4

#: Resampling methods accepted by :func:`resample_cycle` (and :func:`place_cycles`).
RESAMPLE_METHODS = ("fft", "linear")


def frame_positions(n_frames: int, hop: int) -> np.ndarray:
    """Sample index of the centre of each analysis frame (HMS convention)."""
    return np.arange(int(n_frames), dtype=np.float64) * float(hop)


def _voiced_runs(voiced: np.ndarray) -> list:
    """Contiguous runs of True as (start, stop) index pairs."""
    flags = np.ascontiguousarray(voiced, dtype=np.int8).reshape(-1)
    if not flags.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags, [0]])))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def pick_epochs(f0: np.ndarray, hop: int, n_samples: int, fs: int,
                f0_floor: float = 50.0, f0_ceil: float = 2000.0,
                refine: bool = False, residual: Optional[np.ndarray] = None,
                refine_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Frame F0 track -> ``(epochs, runs)``, both increasing sample-index arrays.

    ``epochs`` are the pitch-synchronous positions, ``runs[i]`` the index of the
    voiced run epoch ``i`` belongs to.  Two epochs of the same run are at most
    one period apart; two epochs of different runs are separated by an unvoiced
    gap and must never be treated as one cycle.

    Parameters
    ----------
    f0
        ``(T,)`` per-frame F0 in Hz; ``0``, negative and non-finite values mean
        unvoiced.
    hop, n_samples, fs
        Frame hop, signal length and sample rate.
    f0_floor, f0_ceil
        Supported source pitch range.  A track value outside it is *clamped*
        (period limits), never dropped: an out-of-range estimate still yields
        monotone, valid cycles.
    refine
        Snap each epoch to the largest residual sample within
        ``+/- refine_ratio * period``.  Requires ``residual``.  Each voiced run
        is refined on its own: a neighbouring run's epochs -- and the unvoiced
        gap between them -- never take part in an epoch's window, its local
        period or its bounds, so refinement cannot move an epoch across a
        voiced/unvoiced boundary.
    """
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
    hop = max(1, int(hop))
    n_samples = int(n_samples)
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if n_samples <= 0 or f0.size == 0:
        return empty

    period_min = float(max(MIN_PERIOD_SAMPLES, int(np.floor(fs / max(f0_ceil, 1.0)))))
    period_max = float(max(period_min + 1.0, np.ceil(fs / max(min(f0_floor, f0_ceil),
                                                              1.0))))
    voiced = np.isfinite(f0) & (f0 > 0)
    positions = frame_positions(f0.size, hop)

    epochs, runs, spans = [], [], {}
    for run, (start_frame, stop_frame) in enumerate(_voiced_runs(voiced)):
        start = int(start_frame) * hop
        stop = min(n_samples, int(stop_frame) * hop)
        if stop - start < 2:
            continue
        samples = np.arange(start, stop)
        track = np.interp(samples, positions[start_frame:stop_frame],
                          f0[start_frame:stop_frame])
        period = np.clip(fs / np.maximum(track, 1e-6), period_min, period_max)
        phase = np.cumsum(1.0 / period)
        if phase[-1] < 1.0:                      # no full period in this run
            continue
        targets = np.arange(1.0, np.floor(phase[-1]) + 1.0)
        index = np.searchsorted(phase, targets)
        index = index[index < len(samples)]
        if len(index):
            epochs.append(start + index)
            runs.append(np.full(len(index), run, dtype=np.int64))
            # the samples this run covers: the region its epochs may be refined
            # inside, so a snap can never leave the run
            spans[run] = (start, stop)

    if not epochs:
        return empty
    epochs = np.concatenate(epochs).astype(np.int64)
    runs = np.concatenate(runs)
    order = np.argsort(epochs, kind="stable")
    epochs, runs = epochs[order], runs[order]
    if refine and residual is not None and len(epochs) > 1:
        epochs = _refine_epochs(epochs, np.asarray(residual, dtype=np.float64),
                                period_min, period_max, refine_ratio,
                                runs=runs, bounds=spans)
    return epochs, runs


def _refine_epochs(epochs: np.ndarray, residual: np.ndarray, period_min: float,
                   period_max: float, ratio: float,
                   runs: Optional[np.ndarray] = None,
                   bounds: Optional[dict] = None) -> np.ndarray:
    """Snap every epoch onto a local residual maximum, one voiced run at a time.

    Two epochs of different voiced runs are separated by an unvoiced gap and must
    never influence each other, so each run is refined on its own: the neighbour
    bounds, the local period and the span an epoch may move inside all come from
    the run the epoch belongs to.  ``bounds`` maps a run id to its sample span
    ``(start, stop)``; a run without an entry (or ``runs=None``, meaning "the
    whole array is one run") is refined as if it spanned the whole signal, which
    is the behaviour of a caller that has no run information.

    Inside a run the window of epoch ``i`` is clipped to
    ``(epoch[i-1] + period_min, epoch[i+1] - period_min)`` so the result stays
    strictly increasing *and* no cycle is ever pushed below the shortest period
    the extractor keeps -- otherwise a snap on a very short cycle (a high F0)
    could shrink its neighbours below the floor and the cycle would be dropped
    even though the tracked epochs were valid.  The first and last epoch of a run
    have no neighbour on that side; their bound is the run's own span, never a
    neighbouring run's epoch, so a snap cannot pull an epoch into the unvoiced
    gap either.  Epochs whose window holds nothing to snap onto keep their
    tracked position.
    """
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    n = len(residual)
    epochs = np.asarray(epochs, dtype=np.int64).reshape(-1)
    if n == 0 or epochs.size == 0:
        return epochs
    if runs is None:
        groups = [(None, 0, epochs.size)]
    else:
        runs = np.asarray(runs).reshape(-1)
        if runs.size != epochs.size:
            raise ValueError("runs must have one entry per epoch")
        # the epochs are sorted by position and each run's epochs are contiguous
        # in it, so a run is a slice of the array
        edges = np.concatenate([[0], np.flatnonzero(np.diff(runs)) + 1, [epochs.size]])
        groups = [(int(runs[a]), a, b) for a, b in zip(edges[:-1], edges[1:])]

    refined = epochs.copy()
    for run, first, last in groups:
        span_start, span_stop = (0, n) if bounds is None else bounds.get(run, (0, n))
        refined[first:last] = _refine_run(
            epochs[first:last], residual, span_start, span_stop,
            period_min, period_max, ratio)
    return refined


def _refine_run(epochs: np.ndarray, residual: np.ndarray, span_start: int,
                span_stop: int, period_min: float, period_max: float,
                ratio: float) -> np.ndarray:
    """Refine the epochs of a single voiced run (increasing sample indices)."""
    n = len(residual)
    refined = epochs.copy()
    spacing = int(max(1, round(period_min)))
    for i in range(len(epochs)):
        # Local period: half the distance between the neighbouring epochs at the
        # edges, the full distance in the middle.  (Using the *span* of two
        # intervals would double the search window and let the refinement hop
        # onto the next period's event, which is exactly the failure this
        # bound exists to prevent.)
        if 0 < i < len(epochs) - 1:
            period = 0.5 * float(epochs[i + 1] - epochs[i - 1])
        elif i + 1 < len(epochs):
            period = float(epochs[i + 1] - epochs[i])
        elif i > 0:
            period = float(epochs[i] - epochs[i - 1])
        else:
            period = period_min          # a lone epoch: no period to size with
        period = float(np.clip(period, period_min, period_max))
        width = max(1, int(round(ratio * period)))
        low = int(refined[i - 1]) + spacing if i > 0 else int(span_start)
        high = (int(epochs[i + 1]) - spacing if i + 1 < len(epochs)
                else int(span_stop) - 1)
        low = min(max(low, 0), n - 1)
        high = min(max(high, 0), n - 1)
        window_low = min(max(epochs[i] - width, low), high)
        window_high = min(max(epochs[i] + width, window_low), high)
        window = np.abs(residual[window_low:window_high + 1])
        if window.size and np.isfinite(window).any():
            refined[i] = window_low + int(np.argmax(np.nan_to_num(window, nan=-1.0)))
        else:
            refined[i] = epochs[i]
    return refined


def resample_cycle(cycle: np.ndarray, length: int, method: str = "fft"
                   ) -> np.ndarray:
    """Resample one cycle to exactly ``length`` samples.

    ``"fft"`` treats the cycle as one period of a periodic signal (band-limited,
    wraps around); ``"linear"`` is plain interpolation between the cycle's first
    and last sample.  Both preserve a constant signal, so a cycle's level survives
    the round trip through a fixed-length vector.
    """
    cycle = np.asarray(cycle, dtype=np.float64).reshape(-1)
    length = int(length)
    if length < 1:
        raise ValueError("length must be positive")
    if cycle.size == 0:
        return np.zeros(length, dtype=np.float64)
    if cycle.size == length:
        return cycle.copy()
    if method == "linear":
        positions = np.linspace(0.0, cycle.size, length, endpoint=False)
        return np.interp(positions, np.arange(cycle.size), cycle)
    if method != "fft":
        raise ValueError(f"unknown resample method {method!r}; expected one of "
                         f"{RESAMPLE_METHODS}")
    spectrum = np.fft.rfft(cycle)
    bins = length // 2 + 1
    out = np.zeros(bins, dtype=complex)
    keep = min(len(spectrum), bins)
    out[:keep] = spectrum[:keep]
    return np.fft.irfft(out, length) * (float(length) / float(cycle.size))


def impulsiveness(cycle: np.ndarray) -> float:
    """How much a source vector looks like one concentrated excitation event.

    Uses the *participation ratio* of the cycle's energy, ``(sum x^2)^2 /
    (N * sum x^4)`` -- the share of the vector that carries energy.  A single
    spike gives ``1/N`` and Gaussian noise gives ``~1/3`` (both exact in the
    limit), so those are the two ends of the scale: 1.0 for one clean event,
    0.0 for energy spread over the cycle.

    It is a cheap, scale-free *time-domain concentration* measure, which is what
    a synthesis backend wants when deciding how much noise to mix in.  It is
    deliberately not called aperiodicity: that is a property of a spectrum
    (which the acoustic model already models in its aperiodicity bands), and a
    smooth cycle with no sharp event but no noise either saturates this measure
    at the noisy end.
    """
    x = np.asarray(cycle, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        return 0.0
    energy = float(np.sum(x * x))
    if energy <= 0.0:
        return 0.0
    fourth = float(np.sum(x ** 4))
    if fourth <= 0.0:
        return 1.0
    ratio = (energy * energy) / (x.size * fourth)
    low, high = 1.0 / x.size, 1.0 / 3.0
    if high <= low:
        return 1.0
    spread = float(np.clip((ratio - low) / (high - low), 0.0, 1.0))
    return float(1.0 - spread)


def noise_level(cycle: np.ndarray) -> float:
    """``1 - impulsiveness``: the noise share a backend should mix in."""
    return float(1.0 - impulsiveness(cycle))


@dataclass
class CycleSet:
    """Extracted, fixed-length source cycles (the output of a cycle backend)."""

    vectors: np.ndarray      # (K, cycle_length), unit RMS
    gains: np.ndarray        # (K,) level restored by `vector * gain`
    noise: np.ndarray        # (K,) in [0, 1]
    epochs: np.ndarray       # (K,) sample index of each cycle's start
    periods: np.ndarray      # (K,) samples spanned (exact, measured)

    def __len__(self) -> int:
        return int(self.epochs.shape[0])


def extract_cycles(residual: np.ndarray, epochs: np.ndarray, cycle_length: int,
                   min_period: int = MIN_PERIOD_SAMPLES, min_rms: float = 1e-9,
                   method: str = "fft", runs: Optional[np.ndarray] = None
                   ) -> CycleSet:
    """Cut one residual cycle per epoch interval and normalise its length.

    Every cycle spans ``[epoch[i], epoch[i+1])`` -- the *measured* period, not a
    nominal one -- and is dropped unless it is fully inside the residual, has at
    least ``min_period`` samples, carries more than ``min_rms`` energy and (when
    ``runs`` is given) has both epochs in the same voiced run.  A dropped cycle
    is simply absent from the result: no entry in ``vectors`` is ever all-zero,
    non-finite or un-normalised.
    """
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    epochs = np.asarray(epochs, dtype=np.int64).reshape(-1)
    empty = CycleSet(vectors=np.zeros((0, int(cycle_length))), gains=np.zeros(0),
                     noise=np.zeros(0), epochs=np.zeros(0, dtype=np.int64),
                     periods=np.zeros(0, dtype=np.int64))
    if len(epochs) < 2 or residual.size == 0:
        return empty

    periods = np.diff(epochs)
    n_samples = residual.size
    usable = ((periods >= max(1, int(min_period)))
              & (epochs[:-1] >= 0)
              & (epochs[:-1] + periods <= n_samples))
    if runs is not None:
        runs = np.asarray(runs).reshape(-1)
        if runs.size != epochs.size:
            raise ValueError("runs must have one entry per epoch")
        usable &= runs[:-1] == runs[1:]

    vectors, gains, noise = [], [], []
    kept_epochs, kept_periods = [], []
    for i in np.flatnonzero(usable):
        epoch = int(epochs[i])
        period = int(periods[i])
        cycle = residual[epoch:epoch + period]
        if cycle.size != period or not np.isfinite(cycle).all():
            continue
        rms = float(np.sqrt(np.mean(cycle * cycle)))
        if not np.isfinite(rms) or rms <= min_rms:
            continue
        vector = resample_cycle(cycle, cycle_length, method)
        if not np.isfinite(vector).all():
            continue
        # Normalise *after* resampling so the stored vector is exactly unit RMS
        # and the gain is exactly the level that is restored on the way out:
        # `vector * gain` is the analysed cycle, band-limited to cycle_length
        # harmonics.  (Band-limiting is the only level loss, and it only exists
        # for periods longer than `cycle_length` samples.)
        norm = float(np.sqrt(np.mean(vector * vector)))
        if not np.isfinite(norm) or norm <= 1e-12:
            continue
        vectors.append(vector / norm)
        gains.append(norm)
        noise.append(noise_level(vector))
        kept_epochs.append(epoch)
        kept_periods.append(period)

    if not vectors:
        return empty
    return CycleSet(
        vectors=np.asarray(vectors, dtype=np.float64).reshape(-1, cycle_length),
        gains=np.asarray(gains, dtype=np.float64),
        noise=np.asarray(noise, dtype=np.float64),
        epochs=np.asarray(kept_epochs, dtype=np.int64),
        periods=np.asarray(kept_periods, dtype=np.int64))


def place_cycles(cycles: np.ndarray, epochs: np.ndarray, periods: np.ndarray,
                 n_samples: int, method: str = "fft") -> np.ndarray:
    """Spread fixed-length source vectors back over the sample grid.

    Each vector is resampled to its own ``period`` and written at its ``epoch``;
    where two units meet the samples are averaged, so a unit list that tiles the
    timeline (as :func:`extract_cycles` produces) reconstructs the signal by
    plain concatenation.  Spans with no unit stay silent -- for a voice that is
    the unvoiced region, whose excitation HMS's vocoder still owns.
    """
    cycles = np.atleast_2d(np.asarray(cycles, dtype=np.float64))
    epochs = np.asarray(epochs, dtype=np.int64).reshape(-1)
    periods = np.asarray(periods, dtype=np.int64).reshape(-1)
    n_samples = max(0, int(n_samples))
    out = np.zeros(n_samples, dtype=np.float64)
    if n_samples == 0 or cycles.size == 0:
        return out
    if len(epochs) != len(cycles) or len(periods) != len(cycles):
        raise ValueError("cycles, epochs and periods must have the same length")

    weight = np.zeros(n_samples, dtype=np.float64)
    for i in range(len(cycles)):
        period = int(periods[i])
        start = int(epochs[i])
        if period <= 0 or start < 0 or start >= n_samples:
            continue
        stop = min(n_samples, start + period)
        segment = resample_cycle(cycles[i], period, method)[:stop - start]
        out[start:stop] += segment
        weight[start:stop] += 1.0
    return out / np.maximum(weight, 1.0)


__all__ = ["pick_epochs", "extract_cycles", "resample_cycle", "place_cycles",
           "noise_level", "impulsiveness", "frame_positions", "CycleSet",
           "MIN_PERIOD_SAMPLES", "RESAMPLE_METHODS"]
