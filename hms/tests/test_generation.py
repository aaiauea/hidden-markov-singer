"""Parameter generation: window matrices, banded Cholesky and MLPG.

The important property is that `mlpg` agrees with a brute-force dense solve, and
that the window matrix matches the delta definition used in training.  If either
drifts, every synthesised trajectory silently becomes wrong, so these are
checked numerically rather than by eye.
"""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.features import add_dynamic_features, delta_coeffs
from hms.core import generation
from hms.core.generation import (banded_cholesky, banded_solve, mlpg,
                                 stack_state_statistics, stack_streams,
                                 unstack_streams, window_bandwidth,
                                 window_matrix)


def random_banded_spd(n: int, bandwidth: int, rng) -> np.ndarray:
    matrix = np.zeros((n, n))
    for i in range(n):
        for offset in range(1, bandwidth + 1):
            j = i - offset
            if j >= 0:
                value = rng.normal()
                matrix[i, j] = value
                matrix[j, i] = value
    matrix[np.arange(n), np.arange(n)] = bandwidth + 5.0
    return matrix


def pack_lower_band(matrix: np.ndarray, bandwidth: int) -> np.ndarray:
    n = matrix.shape[0]
    packed = np.zeros((n, bandwidth + 1))
    for i in range(n):
        for offset in range(bandwidth + 1):
            if i - offset >= 0:
                packed[i, offset] = matrix[i, i - offset]
    return packed


def unpack_lower_band(packed: np.ndarray, bandwidth: int) -> np.ndarray:
    n = packed.shape[0]
    full = np.zeros((n, n))
    for i in range(n):
        for offset in range(bandwidth + 1):
            if i - offset >= 0:
                full[i, i - offset] = packed[i, offset]
    return full


@pytest.mark.parametrize("n,bandwidth", [(1, 1), (5, 1), (12, 2), (40, 4), (97, 3)])
def test_banded_cholesky_matches_dense_factorisation(n, bandwidth):
    rng = np.random.default_rng(n * 10 + bandwidth)
    matrix = random_banded_spd(n, bandwidth, rng)
    factor = unpack_lower_band(
        banded_cholesky(pack_lower_band(matrix, bandwidth), bandwidth),
        bandwidth)
    assert np.allclose(factor @ factor.T, matrix, atol=1e-8)


@pytest.mark.parametrize("n,bandwidth,dim", [
    (0, 4, 3), (1, 4, 3), (7, 0, 4), (12, 2, 5), (40, 4, 36),
])
def test_batched_banded_cholesky_matches_featurewise_factors(n, bandwidth, dim):
    """The MLPG D-axis batch must match independent packed factors tightly."""
    rng = np.random.default_rng(n * 100 + bandwidth * 10 + dim)
    packed = np.empty((n, bandwidth + 1, dim))
    expected = []
    for d in range(dim):
        matrix = random_banded_spd(n, bandwidth, rng)
        feature_band = pack_lower_band(matrix, bandwidth)
        packed[:, :, d] = feature_band
        expected.append(banded_cholesky(feature_band, bandwidth))

    batched = banded_cholesky(packed, bandwidth)
    assert batched.shape == packed.shape
    assert np.allclose(batched, np.stack(expected, axis=-1), rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("n,bandwidth,dim", [
    (1, 4, 3), (5, 9, 3), (12, 2, 5), (40, 4, 36),
])
def test_batched_banded_cholesky_repeats_the_scalar_arithmetic_exactly(
        n, bandwidth, dim):
    """The batched kernel must reproduce each feature's rounding bit for bit.

    The batched path factors the feature axis together (one NumPy call per
    panel of short-band products), so it is not obviously the same arithmetic
    as the per-feature scalar path: this pins the terms and their descending
    offset order for the boundary shapes, a bandwidth wider than the sequence,
    and the production ``dim = 36`` case.
    """
    rng = np.random.default_rng(n * 100 + bandwidth * 10 + dim)
    packed = np.empty((n, bandwidth + 1, dim))
    expected = np.empty_like(packed)
    for d in range(dim):
        matrix = random_banded_spd(n, bandwidth, rng)
        feature_band = pack_lower_band(matrix, bandwidth)
        packed[:, :, d] = feature_band
        expected[:, :, d] = banded_cholesky(feature_band, bandwidth)

    assert np.array_equal(banded_cholesky(packed, bandwidth), expected)


@pytest.mark.parametrize("strided", [False, True],
                         ids=["contiguous", "strided"])
@pytest.mark.parametrize("n,bandwidth,dim", [
    (1, 0, 3),              # single frame, no band at all
    (1, 1, 3),              # single frame, band wider than the sequence
    (3, 3, 4),              # n == bandwidth: only the boundary columns run
    (4, 3, 4),              # n == bandwidth + 1: first full column
    (5, 3, 4),              # n == bandwidth + 2: one column of slack to read
    (2400, 4, 36),          # production shape, static + delta (S = 2)
    (1200, 8, 36),          # production shape, static + delta + delta-delta
])
def test_batched_banded_cholesky_at_view_boundaries(
        n, bandwidth, dim, strided):
    """Boundary shapes for the batched kernel's strided column/panel views.

    The batched path reads a factor column ``lower[j + t, t]`` through an
    ``as_strided`` window that reaches `bandwidth` frames past the current
    one, so a wrong stride or a mistreated slack frame shows up at the shapes
    where the band runs off the last row: ``n == bandwidth``, ``n ==
    bandwidth + 1``, ``n == bandwidth + 2``, and the single-frame cases.  The
    factor must stay bit-exact against the per-feature scalar path (which
    touches no view at all), keep exactly ``n`` frames, and leave the packed
    slots above the diagonal zero.
    """
    rng = np.random.default_rng(n * 1000 + bandwidth * 10 + dim)
    if strided:
        # A skimming view: same logical band, deliberately non-contiguous.
        packed = np.zeros((n, bandwidth + 1, 2 * dim))[:, :, ::2]
    else:
        packed = np.empty((n, bandwidth + 1, dim))
    expected = np.empty((n, bandwidth + 1, dim))
    for d in range(dim):
        matrix = random_banded_spd(n, bandwidth, rng)
        feature_band = pack_lower_band(matrix, bandwidth)
        packed[:, :, d] = feature_band
        expected[:, :, d] = banded_cholesky(feature_band, bandwidth)
    # Junk in the packed slots that hold no matrix entry (offset > row).  The
    # scalar path never reads them and returns zeros there; the batched panel
    # does read them, so they must not leak into the factor.
    for row in range(min(bandwidth, n)):
        packed[row, row + 1:] = rng.normal(size=(bandwidth - row, dim))
    assert packed.flags["C_CONTIGUOUS"] is not strided

    factor = banded_cholesky(packed, bandwidth)

    # Same frames, same numbers as the scalar path: the last `bandwidth`
    # frames are exactly the ones whose panel reads reach the slack frames.
    assert factor.shape == (n, bandwidth + 1, dim)
    assert factor.nbytes == packed.dtype.itemsize * packed.size
    assert np.array_equal(factor, expected)

    # Packed slots above the diagonal are not matrix entries.  The panel
    # multiplies them and discards the products, so they must still be zero.
    for row in range(min(bandwidth, n)):
        for offset in range(row + 1, bandwidth + 1):
            assert np.array_equal(factor[row, offset], np.zeros(dim))


@pytest.mark.parametrize("n,bandwidth,dim", [
    (1, 0, 3), (1, 1, 3), (3, 3, 4), (4, 3, 4), (5, 3, 4), (40, 4, 36),
    (16, 8, 36), (2400, 4, 36),
])
def _owner_of(array, roots=None):
    """The ndarray whose memory `array` borrows (metadata only).

    A strided view's own ``.base`` is a ``numpy.lib.stride_tricks`` helper
    object rather than an ndarray, so the walk must start from a real ndarray,
    and a view recorded as built by ``as_strided`` must be resolved through the
    buffer it was built on.  Otherwise the search stops at the view itself and
    every window looks trivially in bounds.
    """
    node = array
    while isinstance(node.base, np.ndarray):
        node = node.base
    if roots is not None and id(node) in roots:
        node = roots[id(node)]
        while isinstance(node.base, np.ndarray):
            node = node.base
    return node


def _element_window(array, roots=None):
    """Byte range of the elements `array` can address, and the owning array.

    The extremes are the sums of the per-axis reaches ``stride * (size - 1)``,
    which real elements attain.  Nothing is dereferenced: only the array
    interface (shape, strides, pointers) is inspected.
    """
    owner = _owner_of(array, roots)
    base = owner.__array_interface__["data"][0]
    start = array.__array_interface__["data"][0]
    reach = [stride * (size - 1)
             for stride, size in zip(array.strides, array.shape)]
    return (start + sum(min(0, value) for value in reach) - base,
            start + sum(max(0, value) for value in reach) - base, owner)


@pytest.mark.parametrize("n,bandwidth,dim", [
    (1, 0, 3), (1, 1, 3), (3, 3, 4), (4, 3, 4), (5, 3, 4), (40, 4, 36),
    (16, 8, 36), (2400, 4, 36),
])
def test_batched_banded_cholesky_views_stay_inside_their_buffer(
        monkeypatch, n, bandwidth, dim):
    """Every strided view must fit inside the buffer it is built from.

    The panel view reads ``mat[j + t, w]`` (up to `bandwidth` frames *behind*
    the current column) and ``mat[j, w - t]`` (down to `bandwidth` packed
    *columns before* it, i.e. a negative packed offset), and ``np.multiply``
    evaluates that whole panel, not just the ``t < w`` half whose results are
    subtracted.  Both directions must therefore be covered by padding of the
    working copy.  This checks the element-address window of every view the
    kernel builds against the buffer it was built from - array metadata only,
    never reading memory - and pins the exact negative reach of the one view
    that addresses below its buffer.
    """
    real_as_strided = generation.as_strided
    roots = {}
    live = []
    windows = []

    def recording_as_strided(buffer, *args, **kwargs):
        view = real_as_strided(buffer, *args, **kwargs)
        roots[id(view)] = buffer
        live.extend([view, buffer])
        if view.size:                      # empty views read nothing
            assert not isinstance(view.base, np.ndarray)   # DummyArray
            low, high, owner = _element_window(view, roots)
            windows.append((owner, low, high, view.strides, view.shape,
                            buffer))
            assert low >= 0, (n, bandwidth, dim, "reads before the buffer",
                              low)
            assert high < owner.nbytes, (n, bandwidth, dim,
                                         "reads past the buffer",
                                         high - owner.nbytes)
        return view

    monkeypatch.setattr(generation, "as_strided", recording_as_strided)

    rng = np.random.default_rng(n * 1000 + bandwidth * 10 + dim)
    packed = np.empty((n, bandwidth + 1, dim))
    expected = np.empty_like(packed)
    for d in range(dim):
        matrix = random_banded_spd(n, bandwidth, rng)
        feature_band = pack_lower_band(matrix, bandwidth)
        packed[:, :, d] = feature_band
        expected[:, :, d] = banded_cholesky(feature_band, bandwidth)

    factor = banded_cholesky(packed, bandwidth)
    assert np.array_equal(factor, expected)

    if bandwidth and n > bandwidth:        # bandwidth 0 has no views at all
        # The hot path did build views (guards against a silent rename), all of
        # them over the same working copy.
        assert windows
        owner = windows[0][0]
        assert all(entry[0] is owner for entry in windows)

        # The factor starts `bandwidth` frames into that copy, with everything
        # before it as front padding for the negative panel reads.
        base = owner.__array_interface__["data"][0]
        offset = factor.__array_interface__["data"][0] - base
        assert offset == bandwidth * owner.strides[0]
        assert offset + n * factor.strides[0] <= owner.nbytes

        # Exactly one view has a negative stride - the `mat[j, w - t]` panel -
        # and its reach below the buffer it was built on is exactly
        # `bandwidth` packed columns, which the front padding covers.
        negative = [entry for entry in windows
                    if any(stride < 0 for stride in entry[3])]
        assert len(negative) == 1
        _, low, _, _, _, buffer = negative[0]
        assert low == (buffer.__array_interface__["data"][0] - base
                       - bandwidth * owner.strides[1])
        assert low >= 0

        # The padding the negative view addresses is allocated, zeroed, and
        # never written by the factorisation: a stray write would show here.
        padding_elements = offset // owner.itemsize
        assert np.array_equal(owner.reshape(-1)[:padding_elements],
                              np.zeros(padding_elements))


@pytest.mark.parametrize("n,bandwidth,dim", [
    (5, 3, 4), (40, 4, 36), (16, 8, 36), (2400, 4, 36),
])
def test_batched_banded_cholesky_accesses_only_allocated_memory(
        monkeypatch, n, bandwidth, dim):
    """Every NumPy operand must be inside its allocation and initialised.

    Three properties, all about memory rather than about arithmetic:

    * every array argument (and ``out``) of every NumPy call the kernel makes
      is checked against the allocation it borrows from, resolving strided
      views through the buffer they were built on - an operand reaching
      outside the working copy fails here even though its result is discarded;
    * every allocation the kernel makes is pre-filled with a sentinel, so a
      read of uninitialised memory changes the factor and fails the
      bit-exactness check, and no sentinel may survive in the working copy;
    * the front padding - the region the negative panel strides address - must
      still be exactly zero afterwards, which fails if anything writes it.
    """
    sentinel = 1e150
    allocations = []
    checked = []
    roots = {}
    live = []
    real_as_strided = generation.as_strided

    def recording_as_strided(buffer, *args, **kwargs):
        view = real_as_strided(buffer, *args, **kwargs)
        roots[id(view)] = buffer
        live.extend([view, buffer])
        allocations.append(_owner_of(buffer))
        return view

    def check(where, args, kwargs):
        for operand in list(args) + list(kwargs.values()):
            if not isinstance(operand, np.ndarray) or operand.size == 0:
                continue
            low, high, owner = _element_window(operand, roots)
            assert low >= 0, (where, "operand before its allocation", low,
                              operand.shape, operand.strides)
            assert high < owner.nbytes, (where, "operand past its allocation",
                                         high - owner.nbytes, operand.shape)
        checked.append(where)

    class Proxy:
        """numpy, with `empty` pre-filled and the kernel's ufuncs checked."""

        def __getattr__(self, name):
            target = getattr(np, name)
            if name == "empty":
                def empty(shape, *args, **kwargs):
                    out = np.full(shape, sentinel, *args, **kwargs)
                    allocations.append(out)
                    return out
                return empty
            if name in ("multiply", "subtract", "divide", "sqrt", "fmax"):
                def wrapper(*args, **kwargs):
                    check(name, args, kwargs)
                    return target(*args, **kwargs)
                return wrapper
            return target

    rng = np.random.default_rng(n * 1000 + bandwidth * 10 + dim)
    packed = np.empty((n, bandwidth + 1, dim))
    expected = np.empty_like(packed)
    for d in range(dim):
        feature_band = pack_lower_band(random_banded_spd(n, bandwidth, rng),
                                       bandwidth)
        packed[:, :, d] = feature_band
        expected[:, :, d] = banded_cholesky(feature_band, bandwidth)

    monkeypatch.setattr(generation, "as_strided", recording_as_strided)
    monkeypatch.setattr(generation, "np", Proxy())
    factor = banded_cholesky(packed, bandwidth)

    # No uninitialised read may affect the result, and every operand of the
    # hot path must have been checked.
    assert np.array_equal(factor, expected)
    assert "multiply" in checked and "divide" in checked

    # The allocation behind the factor is the working copy.  Whatever front
    # padding it carries (test_batched_banded_cholesky_views_stay_inside_
    # their_buffer pins the amount) must be zeroed and must never be written
    # by the factorisation.
    owner = _owner_of(factor, roots)
    assert any(block is owner for block in allocations)
    padding_elements = (factor.__array_interface__["data"][0]
                        - owner.__array_interface__["data"][0]) // owner.itemsize
    assert np.array_equal(owner.reshape(-1)[:padding_elements],
                          np.zeros(padding_elements))
    assert sentinel not in owner          # nothing left uninitialised


def test_batched_banded_cholesky_floors_only_dead_pivots():
    """A small positive pivot is kept; only a dead one hits the 1e-6 floor.

    The batched kernel tests the pivot against the floor instead of always
    evaluating the ``where``, which must not change which pivots are floored.
    """
    band = np.zeros((3, 2, 2))
    band[0, 0] = 1e-9
    assert np.allclose(banded_cholesky(band, 1)[0, 0], np.sqrt(1e-9))
    band[0, 0] = -1.0
    assert np.allclose(banded_cholesky(band, 1)[0, 0], 1e-3)


def test_batched_banded_cholesky_keeps_unused_packed_slots_zero():
    """Packed slots above the diagonal hold no entry and must stay zero."""
    band = np.zeros((6, 3, 2))
    band[0, 1] = 7.0                    # row 0 has no entry at offset 1
    band[1, 2] = 7.0
    factor = banded_cholesky(band, 2)
    assert np.array_equal(factor, banded_cholesky(np.zeros_like(band), 2))
    assert np.array_equal(factor[0, 1], np.zeros(2))
    assert np.array_equal(factor[1, 2], np.zeros(2))


@pytest.mark.parametrize("n,bandwidth", [(6, 1), (20, 2), (50, 3)])
def test_banded_solve_matches_dense_solve(n, bandwidth):
    rng = np.random.default_rng(n + bandwidth)
    matrix = random_banded_spd(n, bandwidth, rng)
    rhs = rng.normal(size=n)
    factor = banded_cholesky(pack_lower_band(matrix, bandwidth), bandwidth)
    assert np.allclose(banded_solve(factor, rhs, bandwidth),
                       np.linalg.solve(matrix, rhs), atol=1e-9)


def test_banded_solve_handles_non_positive_definite_gracefully():
    """A degenerate matrix must not produce NaN/inf (it is variance-floored)."""
    factor = banded_cholesky(np.zeros((3, 2)), 1)
    assert np.isfinite(banded_solve(factor, np.ones(3), 1)).all()


@pytest.mark.parametrize("n,bandwidth,dim", [
    (0, 4, 3), (1, 4, 3), (1, 0, 2), (7, 0, 4), (5, 9, 3), (12, 2, 5),
    (40, 4, 36),
])
def test_batched_banded_solve_matches_featurewise_solves(n, bandwidth, dim):
    """The MLPG D-axis batch must reproduce per-feature solves bit for bit.

    The batched path repeats the single-feature arithmetic in the same
    accumulation order, so the solutions must be exactly equal (not merely
    close) for every supported shape: empty and single-frame sequences, zero
    bandwidth, bandwidth wider than the sequence, and the production
    ``D = 36``, bandwidth-4 shape.
    """
    rng = np.random.default_rng(n * 100 + bandwidth * 10 + dim)
    packed = np.empty((n, bandwidth + 1, dim))
    rhs = rng.normal(size=(n, dim))
    expected = np.empty((n, dim))
    for d in range(dim):
        matrix = random_banded_spd(n, bandwidth, rng)
        factor = banded_cholesky(pack_lower_band(matrix, bandwidth), bandwidth)
        packed[:, :, d] = factor
        expected[:, d] = banded_solve(factor, rhs[:, d], bandwidth)

    batched = banded_solve(packed, rhs, bandwidth)
    assert batched.shape == (n, dim)
    assert np.array_equal(batched, expected)


def test_batched_banded_solve_handles_non_positive_definite_gracefully():
    """A degenerate batch must not produce NaN/inf (it is variance-floored)."""
    factor = banded_cholesky(np.zeros((3, 2, 2)), 1)
    solved = banded_solve(factor, np.ones((3, 2)), 1)
    assert solved.shape == (3, 2)
    assert np.isfinite(solved).all()


def test_banded_solve_rejects_malformed_factors_and_rhs():
    factor = banded_cholesky(np.zeros((6, 3)), 2)
    with pytest.raises(ValueError):
        banded_solve(np.zeros(6), np.ones(6), 2)         # 1-D factor
    with pytest.raises(ValueError):
        banded_solve(np.zeros((6, 2)), np.ones(6), 2)    # narrower than band
    with pytest.raises(ValueError):
        banded_solve(factor, np.ones(5), 2)              # short rhs
    with pytest.raises(ValueError):
        banded_solve(factor, np.ones((6, 1)), 2)         # 2-D rhs, 2-D factor
    batched = banded_cholesky(np.zeros((6, 3, 4)), 2)
    with pytest.raises(ValueError):
        banded_solve(batched, np.ones(6), 2)             # 1-D rhs, 3-D factor
    with pytest.raises(ValueError):
        banded_solve(batched, np.ones((6, 3)), 2)        # wrong feature count


def reference_batched_banded_solve(lower, b, bandwidth):
    """Frozen copy of the pre-optimization batched (3-D) solver.

    The optimised `_substitute_forward` kernel must reproduce this loop bit
    for bit - same products, same accumulation order, same divide - for every
    supported shape.  Kept here as the numerical oracle for the batched path.
    """
    lower = np.asarray(lower, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, _, dim = lower.shape
    y = np.zeros((n, dim), dtype=np.float64)
    for i in range(n):
        acc = np.zeros(dim, dtype=np.float64)
        for d in range(1, min(bandwidth, i) + 1):
            acc += lower[i, d] * y[i - d]
        y[i] = (b[i] - acc) / lower[i, 0]
    for i in range(n - 1, -1, -1):
        acc = np.zeros(dim, dtype=np.float64)
        for d in range(1, min(bandwidth, n - 1 - i) + 1):
            acc += lower[i + d, d] * y[i + d]
        y[i] = (y[i] - acc) / lower[i, 0]
    return y


def assert_same_bits(actual, expected):
    """Bitwise identity, with NaN counted as equal to NaN.

    ``np.array_equal`` treats +0.0 == -0.0 as equal and NaN as unequal; the
    solver's contract is stronger on both counts: identical bits wherever the
    reference is not NaN, and identical NaN-ness everywhere.
    """
    assert actual.shape == expected.shape
    same = actual.view(np.uint64) == expected.view(np.uint64)
    assert bool((same | (np.isnan(actual) & np.isnan(expected))).all())


@pytest.mark.parametrize("n,bandwidth,dim", [
    (0, 0, 2), (0, 4, 3),        # empty sequences
    (1, 0, 2), (1, 5, 4),        # single frame
    (2, 1, 2), (3, 2, 3),        # tiny
    (4, 4, 3), (5, 4, 3),        # n == bw, n == bw + 1
    (3, 4, 3), (7, 9, 3),        # n < bw
    (6, 4, 2), (9, 8, 3),        # n == bw + 2, production bandwidth 8
    (7, 0, 4), (40, 4, 36),      # zero bandwidth, production D
    (257, 8, 5), (1000, 4, 30),  # long sequences
])
def test_batched_banded_solve_is_bit_identical_to_the_reference_loop(
        n, bandwidth, dim):
    """The vectorised kernel must repeat the scalar arithmetic exactly.

    Covers tiny and single-frame sequences, bw = 0, bw = 1, several larger
    bandwidths, n below / at / above the bandwidth, and long sequences.
    """
    rng = np.random.default_rng(n * 100 + bandwidth * 10 + dim)
    lower = rng.normal(size=(n, bandwidth + 1, dim))
    lower[:, 0, :] = np.abs(lower[:, 0, :]) + 1.0
    b = rng.normal(size=(n, dim))
    assert_same_bits(banded_solve(lower, b, bandwidth),
                     reference_batched_banded_solve(lower, b, bandwidth))


def test_batched_banded_solve_matches_the_reference_at_production_scale():
    """The 2400 x 36 x bw-8 workload, bit for bit."""
    rng = np.random.default_rng(2400)
    n, bandwidth, dim = 2400, 8, 36
    lower = rng.normal(size=(n, bandwidth + 1, dim))
    lower[:, 0, :] = np.abs(lower[:, 0, :]) + 1.0
    b = rng.normal(size=(n, dim))
    assert_same_bits(banded_solve(lower, b, bandwidth),
                     reference_batched_banded_solve(lower, b, bandwidth))


@pytest.mark.parametrize("label", ["signed-zero rhs", "signed-zero band",
                                   "zero diagonal", "nan diagonal",
                                   "nan rhs", "inf offdiag", "inf rhs",
                                   "tiny pivot"])
def test_batched_banded_solve_edge_values_match_the_reference(label):
    """NaN/inf propagation and degenerate pivots must behave as before."""
    ones = np.ones((5, 3, 2))
    if label == "signed-zero rhs":
        lower, b, bw = ones, np.full((5, 2), -0.0), 1
    elif label == "signed-zero band":
        lower = ones.copy()
        lower[:, 1, :] = -0.0
        b, bw = np.full((5, 2), -0.0), 1
    elif label == "zero diagonal":
        lower, b, bw = np.zeros((5, 3, 2)), np.ones((5, 2)), 1
    elif label == "nan diagonal":
        lower, b, bw = np.full((5, 3, 2), np.nan), np.ones((5, 2)), 1
    elif label == "nan rhs":
        lower, b, bw = ones, np.full((5, 2), np.nan), 1
    elif label == "inf offdiag":
        lower = np.stack([np.ones(5), np.full(5, np.inf)], 1)[:, :, None]
        lower = np.broadcast_to(lower, (5, 2, 2)).copy()
        b, bw = np.ones((5, 2)), 1
    elif label == "inf rhs":
        lower, b, bw = ones, np.full((5, 2), np.inf), 1
    else:
        lower = np.stack([np.full(5, 1e-300), np.ones(5)], 1)[:, :, None]
        lower = np.broadcast_to(lower, (5, 2, 2)).copy()
        b, bw = np.ones((5, 2)), 1
    with np.errstate(all="ignore"):
        expected = reference_batched_banded_solve(lower, b, bw)
        actual = banded_solve(lower, b, bw)
    assert_same_bits(actual, expected)


def test_batched_banded_solve_accepts_non_contiguous_inputs():
    """Views with arbitrary strides (as produced inside mlpg) are supported."""
    rng = np.random.default_rng(21)
    n, bandwidth, dim = 40, 4, 5
    wide_lower = rng.normal(size=(n, bandwidth + 1, 2 * dim))
    wide_b = rng.normal(size=(n, 2 * dim))
    lower = wide_lower[:, :, ::2]                    # non-contiguous feature axis
    b = wide_b[:, ::2]
    expected = reference_batched_banded_solve(
        np.ascontiguousarray(lower), np.ascontiguousarray(b), bandwidth)
    assert_same_bits(banded_solve(lower, b, bandwidth), expected)
    assert_same_bits(banded_solve(np.asfortranarray(lower), b, bandwidth),
                     expected)
    # same values behind genuinely strided (non-contiguous) buffers
    strided_lower_buf = np.empty((n, bandwidth + 1, 2 * dim))
    strided_lower_buf[:, :, ::2] = lower
    strided_b_buf = np.empty((n, 2 * dim))
    strided_b_buf[:, ::2] = b
    assert_same_bits(banded_solve(strided_lower_buf[:, :, ::2],
                                  strided_b_buf[:, ::2], bandwidth), expected)


def test_batched_banded_solve_does_not_touch_inputs_or_guard_regions():
    """Memory-boundary regression: no read or write outside the inputs.

    The factor and rhs sit inside sentinel-filled backing buffers, as views.
    A solver that reached past the view into the backing store would either
    pick up sentinel values (the result stops matching the reference) or
    overwrite them (the sentinels change).  Both are checked.  The solver
    constructs no stride-computed views - only ordinary slices of its inputs
    and of explicit padded scratch - so this pins the property down.
    """
    rng = np.random.default_rng(23)
    n, bandwidth, dim = 33, 8, 3          # n = 3 * bw + 9: every lane is used
    guard = 5
    sentinel = 123456789.0

    lower_back = np.full((n + 2 * guard, bandwidth + 1, dim), sentinel)
    lower_back[guard:guard + n] = rng.normal(size=(n, bandwidth + 1, dim))
    lower_back[guard:guard + n, 0, :] = np.abs(lower_back[guard:guard + n, 0]) + 1
    lower = lower_back[guard:guard + n]

    b_back = np.full((n + 2 * guard, dim), sentinel)
    b_back[guard:guard + n] = rng.normal(size=(n, dim))
    b = b_back[guard:guard + n]

    expected = reference_batched_banded_solve(lower, b, bandwidth)
    lower_saved = lower.copy()
    b_saved = b.copy()
    assert_same_bits(banded_solve(lower, b, bandwidth), expected)

    assert (lower_back[:guard] == sentinel).all()
    assert (lower_back[guard + n:] == sentinel).all()
    assert (b_back[:guard] == sentinel).all()
    assert (b_back[guard + n:] == sentinel).all()
    # the solved-for rows themselves must be untouched too
    assert_same_bits(lower, lower_saved)
    assert_same_bits(b, b_saved)


def test_mlpg_solve_matches_per_feature_scalar_solves(monkeypatch):
    """mlpg's batched solve must equal feature-by-feature scalar solves.

    Records the factor and right-hand sides mlpg actually passes to
    `banded_solve`, then re-solves each feature with the single-feature
    (2-D packed-band) path: the production trajectory must match exactly.
    """
    rng = np.random.default_rng(21)
    n_frames, dim = 120, 36
    means = rng.normal(size=(n_frames * 2, dim))
    variances = np.exp(rng.normal(-4.0, 1.0, size=means.shape))

    captured = {}
    real_solve = generation.banded_solve

    def recording_solve(lower, b, bandwidth):
        captured.update(lower=lower, b=b, bandwidth=bandwidth)
        return real_solve(lower, b, bandwidth)

    monkeypatch.setattr(generation, "banded_solve", recording_solve)
    trajectory = generation.mlpg(means, variances, (dim, dim))

    lower, rhs, bandwidth = captured["lower"], captured["b"], captured["bandwidth"]
    assert lower.shape == (n_frames, bandwidth + 1, dim)   # batched (3-D) form
    assert rhs.shape == (n_frames, dim)
    expected = np.stack(
        [banded_solve(lower[:, :, d], rhs[:, d], bandwidth)
         for d in range(dim)], axis=-1)
    assert np.array_equal(trajectory, expected)


def test_batched_banded_cholesky_uses_the_scalar_pivot_floor():
    factor = banded_cholesky(np.zeros((3, 2, 2)), 1)
    assert np.allclose(factor[:, 0, :], 1e-3)  # sqrt(1e-6), as in 2-D


def test_banded_cholesky_rejects_malformed_packed_bands():
    with pytest.raises(ValueError):
        banded_cholesky(np.zeros((3, 1)), 1)          # needs bandwidth + 1 cols
    with pytest.raises(ValueError):
        banded_cholesky(np.zeros(3), 1)


@pytest.mark.parametrize("use_delta2", [False, True])
@pytest.mark.parametrize("n_frames", [3, 12, 40])
def test_window_matrix_reproduces_training_deltas(n_frames, use_delta2):
    rng = np.random.default_rng(n_frames)
    dim = 3
    trajectory = np.cumsum(rng.normal(size=(n_frames, dim)) * 0.1, axis=0)
    stream_sizes = (dim, dim, dim) if use_delta2 else (dim, dim)
    features = add_dynamic_features(trajectory, True, use_delta2, window=2)
    stacked = stack_streams(features, stream_sizes)
    matrix = window_matrix(n_frames, stream_sizes, window=2)
    assert np.allclose(matrix @ trajectory.reshape(-1),
                       stacked.reshape(-1), atol=1e-12)


@pytest.mark.parametrize("use_delta2", [False, True])
def test_mlpg_recovers_the_trajectory_that_generated_the_means(use_delta2):
    rng = np.random.default_rng(11)
    n_frames, dim = 25, 3
    trajectory = np.cumsum(rng.normal(size=(n_frames, dim)) * 0.05, axis=0)
    stream_sizes = (dim, dim, dim) if use_delta2 else (dim, dim)
    stacked = stack_streams(add_dynamic_features(trajectory, True, use_delta2),
                            stream_sizes)
    variances = np.full_like(stacked, 1e-4)
    generated = mlpg(stacked, variances, stream_sizes)
    assert np.allclose(generated, trajectory, atol=1e-6)


@pytest.mark.parametrize("use_delta2", [False, True])
def test_mlpg_matches_a_dense_reference_with_uneven_variances(use_delta2):
    rng = np.random.default_rng(12)
    n_frames, dim = 20, 2
    stream_sizes = (dim, dim, dim) if use_delta2 else (dim, dim)
    means = rng.normal(size=(n_frames * len(stream_sizes), dim))
    variances = np.exp(rng.normal(-1.0, 0.7, size=means.shape))

    matrix = window_matrix(n_frames, stream_sizes, window=2)
    precision = np.diag((1.0 / variances).reshape(-1))
    dense = np.linalg.solve(matrix.T @ precision @ matrix,
                            matrix.T @ precision @ means.reshape(-1))
    dense = dense.reshape(n_frames, dim)
    assert np.allclose(mlpg(means, variances, stream_sizes), dense, atol=1e-8)


def test_mlpg_of_a_single_state_sequence_is_the_mean():
    """With one stream there is nothing to smooth: the means come straight back."""
    rng = np.random.default_rng(13)
    means = rng.normal(size=(10, 4))
    variances = np.ones_like(means)
    assert np.allclose(mlpg(means, variances, (4,), smooth=True), means)


def test_mlpg_smoothing_off_returns_static_means():
    rng = np.random.default_rng(14)
    static = rng.normal(size=(20, 3))
    means = np.concatenate([static, np.zeros_like(static)])   # stream-major
    variances = np.ones_like(means)
    assert np.allclose(mlpg(means, variances, (3, 3), smooth=False), static)


def step_statistics(dim=1, n_frames=40, step_at=20, step=5.0,
                    static_variance=0.5, delta_variance=0.02):
    """Stacked statistics for a sequence that steps once, with flat deltas."""
    static = np.zeros((n_frames, dim))
    static[step_at:] = step
    delta = np.zeros_like(static)
    means = np.concatenate([static, delta])
    variances = np.concatenate([np.full_like(static, static_variance),
                                np.full_like(delta, delta_variance)])
    return means, variances


def test_mlpg_interpolates_between_competing_states():
    """A smooth trajectory must not jump between two distant state means."""
    dim = 1
    means, variances = step_statistics(dim, delta_variance=0.01)
    generated = mlpg(means, variances, (dim, dim))[:, 0]
    steps = np.abs(np.diff(generated))
    # a raw step would be 5.0 in a single frame; MLPG spreads it out
    assert steps.max() < 2.0, "MLPG should spread the jump over several frames"
    interior = ((generated > 0.5) & (generated < 4.5)).sum()
    assert interior >= 8, f"transition only spans {interior} frames"
    assert generated[0] == pytest.approx(0.0, abs=0.5)
    assert generated[-1] == pytest.approx(5.0, abs=0.5)


def test_variance_scale_relaxes_the_dynamic_constraints():
    dim = 1
    means, variances = step_statistics(dim, delta_variance=0.01)
    largest_step = lambda x: np.abs(np.diff(x[:, 0])).max()
    literal = mlpg(means, variances, (dim, dim), variance_scale=0.02)
    normal = mlpg(means, variances, (dim, dim))
    relaxed = mlpg(means, variances, (dim, dim), variance_scale=50.0)
    # a larger scale means weaker delta constraints, so the trajectory is
    # allowed to follow the (stepped) static means more closely
    assert (largest_step(literal) < largest_step(normal)
            < largest_step(relaxed))
    # and the extremes really are different trajectories, not rounding
    assert not np.allclose(relaxed, literal, atol=1e-2)
    with pytest.raises(ValueError):
        mlpg(means, variances, (dim, dim), variance_scale=0.0)


def test_stack_and_unstack_streams_are_inverse():
    rng = np.random.default_rng(16)
    features = rng.normal(size=(7, 6))
    stacked = stack_streams(features, (3, 3))
    assert stacked.shape == (14, 3)
    assert np.allclose(unstack_streams(stacked, 7, 3), features)
    with pytest.raises(ValueError):
        stack_streams(features, (2, 2))


def test_window_bandwidth_follows_the_number_of_streams():
    assert window_bandwidth((3,)) == 0
    assert window_bandwidth((3, 3), window=2) == 4
    assert window_bandwidth((3, 3, 3), window=2) == 8


def test_stack_state_statistics_shapes():
    stream_means = [np.zeros((3, 2)), np.ones((3, 2))]
    stream_variances = [np.full((3, 2), 0.5), np.full((3, 2), 0.25)]
    means, variances = stack_state_statistics(stream_means, stream_variances,
                                              [4, 2, 3])
    assert means.shape == (18, 2)          # (4+2+3) frames x 2 streams
    assert variances.shape == means.shape
    assert np.allclose(means[:9], 0.0)     # static block first
    assert np.allclose(means[9:], 1.0)     # then the delta block
    empty = stack_state_statistics(stream_means, stream_variances, [0, 0, 0])
    assert empty[0].shape == (0, 2)


def test_delta_coefficients_are_antisymmetric_and_sum_to_zero():
    coeffs = delta_coeffs(3)
    assert len(coeffs) == 7
    assert coeffs.sum() == pytest.approx(0.0)
    assert np.allclose(coeffs, -coeffs[::-1])
