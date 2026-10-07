"""PCA over source vectors: fitting, codec, determinism, save/load, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from hms.source.pca import DEFAULT_N_COMPONENTS, SourcePCA


def random_cycles(n: int = 60, length: int = 24, rank: int = 3, noise: float = 1e-6,
                  seed: int = 0) -> np.ndarray:
    """Cycles that live in a known ``rank``-dimensional subspace plus noise."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((length, rank)))
    coefficients = rng.standard_normal((n, rank))
    return coefficients @ basis.T + noise * rng.standard_normal((n, length))


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def test_fit_recovers_a_known_subspace():
    cycles = random_cycles(rank=3)
    pca = SourcePCA.fit(cycles, n_components=3)
    report = pca.report(cycles)
    assert pca.n_components == 3
    assert pca.cycle_length == cycles.shape[1]
    assert pca.n_samples == len(cycles)
    assert report["relative_rmse"] < 1e-3
    assert report["explained_variance"] > 0.999
    assert pca.cumulative_explained_variance > 0.999


def test_fit_is_deterministic_and_components_are_orthonormal():
    cycles = random_cycles(n=40, length=32, rank=6)
    first = SourcePCA.fit(cycles, n_components=8)
    second = SourcePCA.fit(cycles, n_components=8)
    assert np.array_equal(first.components, second.components)
    assert np.array_equal(first.mean, second.mean)
    assert np.allclose(first.components @ first.components.T, np.eye(8), atol=1e-10)


def test_fit_sign_convention_is_fixed():
    """The largest-magnitude entry of every component is positive."""
    pca = SourcePCA.fit(random_cycles(n=40, length=32, rank=5), n_components=5)
    for component in pca.components:
        assert component[np.argmax(np.abs(component))] > 0


def test_explained_variance_is_monotone_and_bounded():
    cycles = random_cycles(n=80, length=32, rank=8)
    ratios = []
    for k in (1, 2, 4, 8, 16, 32):
        pca = SourcePCA.fit(cycles, n_components=k)
        ratio = pca.cumulative_explained_variance
        assert 0.0 <= ratio <= 1.0 + 1e-12
        assert (pca.explained_variance_ratio >= -1e-12).all()
        ratios.append(ratio)
    assert ratios == sorted(ratios)
    assert ratios[-1] > 0.999          # 80 samples, rank 8: 32 components suffice
    assert SourcePCA.fit(cycles, n_components=4).cumulative_explained_variance < 1.0


def test_more_components_never_reconstruct_worse():
    cycles = random_cycles(n=60, length=32, rank=6)
    errors = [SourcePCA.fit(cycles, n_components=k).report(cycles)["mse"]
              for k in (1, 2, 4, 8, 16)]
    assert errors == sorted(errors, reverse=True)


def test_fit_clips_component_count_to_the_vector_length():
    cycles = random_cycles(n=20, length=12, rank=4)
    pca = SourcePCA.fit(cycles, n_components=64)
    assert pca.n_components == 12
    assert np.allclose(pca.components @ pca.components.T, np.eye(12), atol=1e-10)


def test_more_components_than_samples_is_still_a_valid_basis():
    """One sample per direction is not a basis to write home about -- but valid."""
    cycles = random_cycles(n=3, length=16, rank=3)
    pca = SourcePCA.fit(cycles, n_components=16)
    assert pca.n_components == 16
    assert np.isfinite(pca.components).all()
    assert np.allclose(pca.components @ pca.components.T, np.eye(16), atol=1e-10)
    coefficients = pca.encode(cycles)
    assert np.isfinite(coefficients).all()
    # the training data is reproduced exactly: it spans at most 3 directions
    assert np.allclose(pca.reconstruct(cycles), cycles, atol=1e-8)
    assert pca.report(cycles)["mse"] < 1e-16
    # the directions with no data behind them carry zero variance
    assert pca.eigenvalues[-1] == 0.0


def test_constant_data_is_degenerate_but_well_behaved():
    cycles = np.repeat(np.arange(8.0)[None, :], 5, axis=0)
    pca = SourcePCA.fit(cycles, n_components=4)
    assert pca.total_variance == pytest.approx(0.0, abs=1e-24)
    assert np.allclose(pca.explained_variance_ratio, 0.0)
    assert np.isfinite(pca.reconstruct(cycles)).all()
    assert np.allclose(pca.reconstruct(cycles), cycles, atol=1e-12)
    assert pca.report(cycles)["mse"] == pytest.approx(0.0, abs=1e-24)
    assert pca.report(cycles)["explained_variance"] == 1.0


def test_variance_floor_drops_directions_without_variance():
    cycles = random_cycles(n=40, length=16, rank=2, noise=0.0)
    # a floor above every direction's variance: nothing is kept, the basis is
    # still a valid orthonormal basis, and the reconstruction is the mean
    pca = SourcePCA.fit(cycles, n_components=6, variance_floor=2.0)
    assert np.count_nonzero(pca.eigenvalues) == 0
    assert np.allclose(pca.components @ pca.components.T, np.eye(6), atol=1e-10)
    assert pca.report(cycles)["relative_rmse"] > 0.1
    assert np.allclose(pca.reconstruct(cycles), np.repeat(pca.mean[None, :], 40, axis=0))

    # a floor between the two directions keeps only the strong one
    partial = SourcePCA.fit(cycles, n_components=6, variance_floor=1.0)
    assert np.count_nonzero(partial.eigenvalues) == 1


@pytest.mark.parametrize("bad", [
    np.zeros((0, 8)),                        # no samples
    np.zeros((4, 0)),                        # no samples per vector
    np.full((4, 8), np.nan),
    np.full((4, 8), np.inf),
])
def test_fit_rejects_unusable_input(bad):
    with pytest.raises(ValueError):
        SourcePCA.fit(bad, n_components=2)


def test_fit_rejects_a_non_positive_component_count():
    with pytest.raises(ValueError):
        SourcePCA.fit(random_cycles(), n_components=0)


# --------------------------------------------------------------------------
# Codec
# --------------------------------------------------------------------------


def test_encode_decode_roundtrip_is_exact_with_enough_components():
    cycles = random_cycles(n=40, length=16, rank=16, noise=0.0)
    pca = SourcePCA.fit(cycles, n_components=16)
    coefficients = pca.encode(cycles)
    assert coefficients.shape == (40, 16)
    assert np.allclose(pca.decode(coefficients), cycles, atol=1e-10)
    assert np.allclose(pca.reconstruct(cycles), cycles, atol=1e-10)
    # coefficients are centred on the training set
    assert np.allclose(pca.encode(cycles).mean(axis=0), 0.0, atol=1e-10)


def test_encode_decode_handle_a_single_vector_and_an_empty_batch():
    pca = SourcePCA.fit(random_cycles(n=20, length=8, rank=4), n_components=4)
    single = pca.encode(np.ones((1, 8)))
    assert single.shape == (1, 4)
    assert pca.decode(single).shape == (1, 8)
    empty = pca.encode(np.zeros((0, 8)))
    assert empty.shape == (0, 4)
    assert pca.decode(empty).shape == (0, 8)
    assert pca.report(np.zeros((0, 8)))["n_vectors"] == 0
    assert np.isnan(pca.report(np.zeros((0, 8)))["mse"])


def test_codec_validates_shapes_and_values():
    pca = SourcePCA.fit(random_cycles(n=20, length=8, rank=4), n_components=4)
    with pytest.raises(ValueError):
        pca.encode(np.ones((2, 9)))
    with pytest.raises(ValueError):
        pca.encode(np.full((2, 8), np.nan))
    with pytest.raises(ValueError):
        pca.decode(np.ones((2, 3)))
    with pytest.raises(ValueError):
        pca.decode(np.full((2, 4), np.inf))
    with pytest.raises(ValueError):
        SourcePCA(mean=np.zeros(4), components=np.zeros((2, 5)))
    with pytest.raises(ValueError):
        SourcePCA(mean=np.zeros(4), components=np.zeros((2, 4)), eigenvalues=np.zeros(3))


def test_reconstruction_errors_are_per_vector():
    cycles = random_cycles(n=30, length=16, rank=4)
    pca = SourcePCA.fit(cycles, n_components=2)
    errors = pca.reconstruction_errors(cycles)
    assert errors.shape == (30,)
    assert np.isfinite(errors).all()
    assert (errors >= 0).all()
    report = pca.report(cycles)
    assert report["median_relative_error"] <= report["p90_relative_error"]
    assert report["relative_rmse"] > 0
    assert set(report) >= {"n_vectors", "mse", "rmse", "relative_rmse",
                           "median_relative_error", "p90_relative_error",
                           "explained_variance", "cumulative_explained_variance"}
    assert pca.reconstruction_errors(np.zeros((0, 16))).shape == (0,)


def test_default_component_count_is_in_the_documented_range():
    assert 4 <= DEFAULT_N_COMPONENTS <= 8


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_save_load_roundtrip_is_exact(tmp_path):
    cycles = random_cycles(n=48, length=32, rank=5)
    pca = SourcePCA.fit(cycles, n_components=6)
    path = pca.save(tmp_path / "basis")
    assert path.name == "basis.npz" and path.exists()

    loaded = SourcePCA.load(path)
    assert np.array_equal(loaded.mean, pca.mean)
    assert np.array_equal(loaded.components, pca.components)
    assert np.array_equal(loaded.eigenvalues, pca.eigenvalues)
    assert loaded.total_variance == pca.total_variance
    assert loaded.residual_variance == pca.residual_variance
    assert loaded.n_samples == pca.n_samples
    assert loaded.n_components == pca.n_components
    assert loaded.cycle_length == pca.cycle_length
    assert np.array_equal(loaded.encode(cycles), pca.encode(cycles))

    again = SourcePCA.load(pca.save(tmp_path / "again.npz"))
    assert np.array_equal(again.components, pca.components)


def test_save_overwrites_and_creates_directories(tmp_path):
    pca = SourcePCA.fit(random_cycles(), n_components=4)
    path = pca.save(tmp_path / "nested" / "dir" / "basis.npz")
    assert path.exists()
    assert SourcePCA.load(path).n_components == 4


def test_load_rejects_files_that_are_not_a_basis(tmp_path):
    path = tmp_path / "not_a_basis.npz"
    np.savez_compressed(path, foo=np.zeros(3))
    with pytest.raises(ValueError):
        SourcePCA.load(path)
    text = tmp_path / "basis.txt"
    text.write_text("not an npz")
    with pytest.raises(Exception):
        SourcePCA.load(text)
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        SourcePCA.load(missing)
