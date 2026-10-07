"""Pitch-synchronous cycle extraction: epochs, normalisation, placement."""

from __future__ import annotations

import numpy as np
import pytest

from hms.source.cycles import (MIN_PERIOD_SAMPLES, extract_cycles, frame_positions,
                               impulsiveness, noise_level, pick_epochs, place_cycles,
                               resample_cycle)


def constant_f0_track(f0: float, n_frames: int, hop: int) -> np.ndarray:
    return np.full(n_frames, float(f0))


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------


def test_resample_cycle_preserves_a_constant():
    cycle = np.full(64, 0.37)
    for length in (8, 16, 128, 200):
        out = resample_cycle(cycle, length)
        assert out.shape == (length,)
        assert np.allclose(out, 0.37, atol=1e-12)


def test_resample_cycle_roundtrip_is_exact_up_to_the_shorter_length():
    """A cycle shorter than the vector must survive 128 -> period -> 128."""
    rng = np.random.default_rng(0)
    for period in (11, 32, 84, 128):
        cycle = rng.standard_normal(period)
        vector = resample_cycle(cycle, 128)
        assert vector.shape == (128,)
        assert np.allclose(resample_cycle(vector, period), cycle, atol=1e-10)


def test_resample_cycle_preserves_a_sinusoids_frequency_and_level():
    period, length = 64, 128
    n = np.arange(period)
    cycle = 2.5 * np.cos(2 * np.pi * 3 * n / period)
    out = resample_cycle(cycle, length)
    expected = 2.5 * np.cos(2 * np.pi * 3 * np.arange(length) / length)
    assert np.allclose(out, expected, atol=1e-10)


def test_resample_cycle_linear_method_is_interpolation():
    cycle = np.array([0.0, 1.0, 0.0, -1.0])
    out = resample_cycle(cycle, 8, method="linear")
    assert out.shape == (8,)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_resample_cycle_rejects_unknown_method():
    with pytest.raises(ValueError):
        resample_cycle(np.ones(8), 16, method="cubic")


def test_resample_cycle_handles_empty_input():
    assert resample_cycle(np.zeros(0), 16).shape == (16,)
    with pytest.raises(ValueError):
        resample_cycle(np.ones(4), 0)


# --------------------------------------------------------------------------
# Epoch picking
# --------------------------------------------------------------------------


def test_epochs_of_a_constant_f0_track_are_one_period_apart():
    fs, hop, f0 = 22050, 110, 300.0
    track = constant_f0_track(f0, 60, hop)
    epochs, runs = pick_epochs(track, hop, 6000, fs, refine=False)
    assert len(epochs) == len(runs) >= 20
    spacing = np.diff(epochs)
    expected = fs / f0
    assert np.all(np.abs(spacing - expected) <= 1)      # one sample of rounding
    assert (runs == 0).all()
    assert (np.diff(epochs) > 0).all()


def test_epochs_track_a_gliding_pitch_without_jumping():
    fs, hop = 22050, 110
    track = np.linspace(150.0, 450.0, 80)
    epochs, runs = pick_epochs(track, hop, 9000, fs, refine=False)
    spacing = np.diff(epochs)
    # the period follows the glide down, never outside what the pitch range allows
    assert spacing.min() >= int(np.floor(fs / 2000.0))
    assert spacing.max() <= int(np.ceil(fs / 50.0)) + 1
    assert (runs[1:] == runs[:-1]).all()


def test_epochs_handle_f0_discontinuities_and_octave_errors():
    """A step, a spike and a doubled value must not break monotonicity."""
    fs, hop = 22050, 110
    track = np.full(60, 200.0)
    track[20] = 800.0        # one-frame octave error
    track[30:] = 100.0       # a step down
    epochs, _ = pick_epochs(track, hop, 10000, fs, refine=False)
    assert (np.diff(epochs) > 0).all()
    assert len(epochs) > 20


def test_epochs_are_never_emitted_in_unvoiced_or_invalid_frames():
    fs, hop = 22050, 110
    track = np.zeros(60)
    track[10:20] = 250.0
    epochs, runs = pick_epochs(track, hop, 11000, fs, refine=False)
    # a run [10, 20) frames is samples [1100, 2200); its first epoch is one
    # period after the run start and none can reach the next run
    assert len(epochs) > 0
    assert epochs.min() >= 1100
    assert epochs.max() < 2200 + int(fs / 250.0)
    assert (runs == 0).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -100.0])
def test_epochs_ignore_bad_f0_values(bad):
    fs, hop = 22050, 110
    track = np.full(40, 220.0)
    track[15:20] = bad
    epochs, runs = pick_epochs(track, hop, 7000, fs, refine=False)
    assert (np.diff(epochs) > 0).all()
    assert runs.max() >= 1                      # the run was split by the bad span
    assert len(np.unique(runs)) == 2


def test_epochs_at_extreme_pitch_are_clamped_not_dropped():
    fs, hop = 22050, 110
    for f0, floor, ceil in ((20.0, 50.0, 2000.0), (5000.0, 50.0, 2000.0)):
        track = constant_f0_track(f0, 40, hop)
        epochs, _ = pick_epochs(track, hop, 20000, fs, f0_floor=floor, f0_ceil=ceil,
                                refine=False)
        assert len(epochs) >= 2
        spacing = np.diff(epochs)
        assert spacing.min() >= MIN_PERIOD_SAMPLES
        assert spacing.max() <= int(np.ceil(fs / floor)) + 1


def test_epochs_of_empty_and_tiny_input():
    fs, hop = 22050, 110
    epochs, runs = pick_epochs(np.zeros(0), hop, 100, fs)
    assert len(epochs) == 0 and len(runs) == 0
    epochs, runs = pick_epochs(np.array([220.0]), hop, 0, fs)
    assert len(epochs) == 0
    # a track shorter than one period leaves no complete cycle
    epochs, _ = pick_epochs(np.array([100.0]), hop, 200, fs)
    assert len(epochs) == 0


def test_epochs_do_not_span_across_an_unvoiced_gap():
    """The failure mode this guards against: a 'cycle' spanning seconds of silence."""
    fs, hop = 22050, 110
    track = np.zeros(200)
    track[0:20] = 300.0
    track[180:200] = 300.0
    epochs, runs = pick_epochs(track, hop, 22000, fs, refine=False)
    assert len(np.unique(runs)) == 2
    cycles = extract_cycles(np.random.default_rng(0).standard_normal(22000),
                            epochs, 128, runs=runs)
    assert cycles.periods.max() <= int(np.ceil(fs / 50.0)) + 1


def test_epoch_refinement_stays_monotone_and_inside_the_signal():
    fs, hop = 22050, 110
    rng = np.random.default_rng(1)
    residual = rng.standard_normal(20000)
    residual[5000:5100] = 50.0                 # a loud event the refinement likes
    track = constant_f0_track(250.0, 150, hop)
    epochs, _ = pick_epochs(track, hop, 20000, fs, refine=True, residual=residual)
    assert (np.diff(epochs) > 0).all()
    assert epochs.min() >= 0 and epochs.max() < 20000
    spacing = np.diff(epochs)
    # refinement may move epochs, but never onto the neighbouring period
    assert spacing.min() >= MIN_PERIOD_SAMPLES
    assert spacing.max() <= int(np.ceil(fs / 50.0)) + 1


def test_epoch_refinement_snaps_onto_the_excitation_event():
    """With a residual that is exactly periodic, refinement must land on it."""
    fs, hop, period = 22050, 110, 88
    f0 = fs / period                 # the track and the pulses agree exactly
    residual = np.zeros(20000)
    half = 5
    shape = np.hanning(2 * half + 1)
    for position in range(0, 20000, period):    # perfect pulse train
        low, high = max(0, position - half), min(20000, position + half + 1)
        residual[low:high] += shape[low - (position - half):high - (position - half)]
    track = constant_f0_track(f0, 150, hop)
    epochs, _ = pick_epochs(track, hop, 20000, fs, refine=True, residual=residual)
    # epochs within one period of the end have no pulse to snap onto, and are
    # allowed to stay where the tracker put them
    interior = epochs[epochs < len(residual) - period]
    assert len(interior) > 100
    assert np.allclose(residual[interior], 1.0)
    assert (interior % period).max() <= 1


def test_epoch_refinement_never_uses_a_neighbour_from_another_voiced_run():
    """Refinement refines each voiced run on its own, not the whole epoch array.

    Two voiced runs separated by a six-frame unvoiced gap.  Every tracked epoch
    has an excitation event of its own, and *inside the gap* sit two much louder
    events: one just after the end of the first run, one just before the start
    of the second.  A window sized from the neighbouring run's epoch reaches
    both of them -- that is the leak this test is about -- while a window that
    only knows the run's own epochs reaches neither.  Each run is therefore
    refined independently, every epoch stays on its own run's event, and no
    cycle is stretched across (or out of) a voiced run.
    """
    fs, hop, period, ratio = 22050, 110, 88, 0.25
    n_samples = 22000
    run0, run1 = (20, 60), (66, 100)                 # gap: samples [6600, 7260)
    spans = [(run0[0] * hop, run0[1] * hop), (run1[0] * hop, run1[1] * hop)]
    track = np.zeros(200)
    track[run0[0]:run0[1]] = fs / period
    track[run1[0]:run1[1]] = fs / period

    tracked, runs = pick_epochs(track, hop, n_samples, fs, refine=False)
    assert len(np.unique(runs)) == 2
    assert runs[0] == 0 and runs[-1] == 1            # the runs are not contiguous

    # one excitation event per tracked epoch, plus two louder ones in the gap
    residual = np.zeros(n_samples)
    half = 5
    shape = np.hanning(2 * half + 1)
    for position in tracked:
        residual[position - half:position + half + 1] += shape
    leak0, leak1 = spans[0][1] + 2, spans[1][0] - 2
    residual[leak0] = residual[leak1] = 100.0

    # the geometry this test is about: a window sized from the *other* run's
    # epoch reaches the gap events (a run-relative width would not)
    last0 = tracked[runs == 0][-1]
    first1 = tracked[runs == 1][0]
    leaked_width = int(round(ratio * 0.5 * (first1 - tracked[runs == 0][-2])))
    assert leak0 <= last0 + leaked_width
    assert leak1 >= first1 - leaked_width

    refined, refined_runs = pick_epochs(track, hop, n_samples, fs, refine=True,
                                        residual=residual, refine_ratio=ratio)
    assert np.array_equal(refined_runs, runs)

    # no epoch leaves the run it belongs to, and none snaps onto a gap event
    for run, (start, stop) in enumerate(spans):
        inside = refined[runs == run]
        assert ((inside >= start) & (inside < stop)).all()
        assert not np.isin(inside, [leak0, leak1]).any()
    # each epoch stays glued to an event of its own run: crossing the boundary
    # moves an epoch a whole period away, a normal snap moves it a sample or two
    for run in (0, 1):
        assert (np.abs(refined[runs == run] - tracked[runs == run]) <= 8).all()
    # the boundary epochs in particular are untouched by the other run
    assert refined[runs == 0][-1] == last0
    assert refined[runs == 1][0] == first1

    # and no cycle is produced across (or sticking out of) a voiced run: the
    # leak used to stretch the last cycle of each run to two periods
    cycles = extract_cycles(residual, refined, 128, runs=runs)
    assert len(cycles) > 50
    assert (np.diff(cycles.epochs) > 0).all()
    for epoch, length in zip(cycles.epochs, cycles.periods):
        assert any(start <= epoch and epoch + length <= stop for start, stop in spans)
        assert length <= period + 2


def test_frame_positions_follow_the_frame_grid():
    assert np.array_equal(frame_positions(4, 110), np.array([0.0, 110.0, 220.0, 330.0]))


# --------------------------------------------------------------------------
# Extraction and normalisation
# --------------------------------------------------------------------------


def test_extract_cycles_normalises_length_and_level():
    fs, hop, f0 = 22050, 110, 220.0
    rng = np.random.default_rng(2)
    residual = rng.standard_normal(20000)
    track = constant_f0_track(f0, 150, hop)
    epochs, runs = pick_epochs(track, hop, 20000, fs, refine=False)
    cycles = extract_cycles(residual, epochs, 128, runs=runs)
    assert cycles.vectors.shape[1] == 128
    assert len(cycles) == len(cycles.vectors) == len(cycles.gains) > 20
    assert np.isfinite(cycles.vectors).all()
    rms = np.sqrt((cycles.vectors ** 2).mean(axis=1))
    assert np.allclose(rms, 1.0, atol=1e-9)
    assert (cycles.gains > 0).all()
    assert ((cycles.noise >= 0) & (cycles.noise <= 1)).all()
    # The gain is exactly the level the normalisation removed, i.e. the RMS of
    # the fixed-length vector, and it tracks the analysed cycle's RMS (the two
    # differ by <5 %, which is how much energy a band-limited interpolation of
    # a 100-sample cycle puts between samples).
    resampled = np.stack([resample_cycle(residual[e:e + p], 128) for e, p in
                          zip(cycles.epochs, cycles.periods)])
    assert np.allclose(cycles.gains, np.sqrt((resampled ** 2).mean(axis=1)),
                       rtol=1e-12)
    cycle_rms = np.array([np.sqrt(np.mean(residual[e:e + p] ** 2)) for e, p in
                          zip(cycles.epochs, cycles.periods)])
    assert np.allclose(cycles.gains, cycle_rms, rtol=0.05)
    assert np.corrcoef(cycles.gains, cycle_rms)[0, 1] > 0.99


def test_extract_cycles_drops_silence_and_degenerate_periods():
    residual = np.zeros(2000)
    residual[200:400] = np.sin(np.linspace(0, 20, 200))      # energy in one span only
    epochs = np.arange(0, 2000, 50)
    cycles = extract_cycles(residual, epochs, 32, min_period=8)
    assert cycles.vectors.shape[0] > 0
    assert np.isfinite(cycles.vectors).all()
    assert (np.abs(cycles.vectors).max(axis=1) > 0).all()    # no all-zero vectors
    for epoch, period in zip(cycles.epochs, cycles.periods):
        assert 0 <= epoch and epoch + period <= len(residual)


def test_extract_cycles_needs_two_epochs():
    assert len(extract_cycles(np.ones(100), np.array([10]), 16)) == 0
    assert len(extract_cycles(np.ones(100), np.zeros(0, dtype=int), 16)) == 0
    assert len(extract_cycles(np.zeros(0), np.array([0, 10]), 16)) == 0


def test_extract_cycles_ignores_non_finite_residual():
    residual = np.full(1000, 1.0)
    residual[500] = np.nan
    epochs = np.arange(0, 1000, 40)
    cycles = extract_cycles(residual, epochs, 16)
    assert np.isfinite(cycles.vectors).all()
    assert all(e + p <= 500 or e >= 501 for e, p in
               zip(cycles.epochs, cycles.periods))


def test_extract_cycles_rejects_a_mismatched_run_array():
    with pytest.raises(ValueError):
        extract_cycles(np.ones(100), np.array([0, 10, 20]), 16, runs=np.array([0, 1]))


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


def test_place_cycles_round_trips_a_periodic_signal():
    fs, f0 = 22050, 220.0
    period = int(round(fs / f0))
    t = np.arange(6000)
    signal = np.sin(2 * np.pi * f0 * t / fs) * 0.5 + 0.2 * np.sin(4 * np.pi * f0 * t / fs)
    epochs = np.arange(0, 6000 - 2 * period, period)
    cycles = extract_cycles(signal, epochs, 128)
    rebuilt = place_cycles(cycles.vectors * cycles.gains[:, None], cycles.epochs,
                           cycles.periods, len(signal))
    coverage = np.zeros(len(signal), dtype=bool)
    for epoch, span in zip(cycles.epochs, cycles.periods):
        coverage[epoch:epoch + span] = True
    assert coverage.sum() > 0.9 * len(signal)
    error = np.linalg.norm(signal[coverage] - rebuilt[coverage]) / np.linalg.norm(
        signal[coverage])
    assert error < 1e-6
    assert np.allclose(rebuilt[~coverage], 0.0)


def test_place_cycles_handles_empty_and_out_of_range_input():
    assert place_cycles(np.zeros((0, 16)), np.zeros(0), np.zeros(0), 100).shape == (100,)
    assert place_cycles(np.zeros((0, 16)), np.zeros(0), np.zeros(0), 0).shape == (0,)
    out = place_cycles(np.ones((1, 16)), np.array([500]), np.array([50]), 100)
    assert out.shape == (100,) and not out.any()          # epoch past the end: ignored
    with pytest.raises(ValueError):
        place_cycles(np.ones((2, 16)), np.array([0]), np.array([10]), 100)


def test_place_cycles_clips_a_unit_that_runs_past_the_end():
    out = place_cycles(np.ones((1, 16)), np.array([90]), np.array([50]), 100)
    assert out.shape == (100,)
    assert out[90:].all() and not out[:90].any()


# --------------------------------------------------------------------------
# Noise level proxy
# --------------------------------------------------------------------------


def test_noise_level_separates_an_impulse_from_noise():
    impulse = np.zeros(128)
    impulse[0] = 1.0
    rng = np.random.default_rng(3)
    assert impulsiveness(impulse) > 0.99
    assert noise_level(impulse) < 0.01
    assert noise_level(rng.standard_normal(128)) > 0.5
    assert noise_level(np.zeros(128)) == 1.0
    assert noise_level(np.array([np.nan] * 4)) == 1.0
    assert noise_level(np.zeros(0)) == 1.0
