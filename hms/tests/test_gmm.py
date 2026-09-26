"""GMM estimation: correctness, robustness on small data, serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from hms.core.gmm import MIN_WEIGHT, DiagGMM, kmeans, kmeans_plus_plus


def two_cluster_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal([0.0, 0.0, 0.0], 0.25, size=(n, 3)),
        rng.normal([4.0, 4.0, -3.0], 0.25, size=(n, 3)),
    ])


def test_single_gaussian_matches_moments():
    X = two_cluster_data(200)
    gmm = DiagGMM.fit(X, n_components=1)
    assert gmm.n_components == 1
    assert np.allclose(gmm.means[0], X.mean(axis=0), atol=1e-9)
    assert np.allclose(gmm.variances[0], X.var(axis=0), atol=1e-9)
    assert gmm.weights[0] == pytest.approx(1.0)


def test_two_components_find_the_clusters():
    X = two_cluster_data()
    gmm = DiagGMM.fit(X, n_components=2, seed=1)
    means = gmm.means[np.argsort(gmm.means[:, 0])]
    assert np.allclose(means[0], [0.0, 0.0, 0.0], atol=0.15)
    assert np.allclose(means[1], [4.0, 4.0, -3.0], atol=0.15)
    assert np.allclose(gmm.weights, [0.5, 0.5], atol=0.05)


def test_likelihood_improves_with_more_components():
    X = two_cluster_data(n=200)
    single = DiagGMM.fit(X, 1)
    double = DiagGMM.fit(X, 2, seed=2)
    assert double.log_likelihood(X).mean() > single.log_likelihood(X).mean() + 0.5


def test_posteriors_and_likelihoods_are_consistent():
    X = two_cluster_data(n=50)          # 100 frames (two clusters)
    gmm = DiagGMM.fit(X, 2, seed=3)
    posterior = gmm.posterior(X)
    assert posterior.shape == (len(X), 2)
    assert np.allclose(posterior.sum(axis=1), 1.0)
    # total likelihood must equal the mixture of component likelihoods
    component_ll = gmm.log_prob(X)
    mixture_ll = np.log(np.exp(component_ll).sum(axis=0))
    assert np.allclose(gmm.log_likelihood(X), mixture_ll, atol=1e-9)


def test_tied_covariance_shares_one_covariance():
    X = two_cluster_data()
    tied = DiagGMM.fit(X, 2, covariance_type="tied", seed=1)
    assert np.allclose(tied.variances[0], tied.variances[1])
    assert tied.covariance_type == "tied"
    # the shared covariance is the mixture-weighted average of the free ones
    free = DiagGMM.fit(X, 2, covariance_type="diag", seed=1)
    expected = (free.weights[:, None] * free.variances).sum(axis=0)
    assert np.allclose(tied.variances[0], expected, rtol=0.35)


def test_tied_covariance_uses_fewer_parameters():
    X = two_cluster_data(n=100)
    free = DiagGMM.fit(X, 3, covariance_type="diag")
    tied = DiagGMM.fit(X, 3, covariance_type="tied")
    assert tied.n_free_params < free.n_free_params


def test_variance_floor_protects_tiny_datasets():
    """Two identical frames must not produce an infinitely peaked Gaussian."""
    X = np.ones((2, 4))
    gmm = DiagGMM.fit(X, n_components=1, var_floor_ratio=1e-3)
    assert (gmm.variances > 0).all()
    assert np.isfinite(gmm.log_likelihood(X)).all()


def test_component_pruning_keeps_weights_valid():
    X = np.concatenate([np.zeros((50, 2)), np.ones((50, 2)) * 5])
    gmm = DiagGMM.fit(X, n_components=5, seed=4)
    assert (gmm.weights >= MIN_WEIGHT - 1e-12).all()
    assert gmm.weights.sum() == pytest.approx(1.0)
    assert gmm.means.shape[0] == gmm.n_components


def test_weighted_fitting_respects_weights():
    """Zero weight on a cluster must hide it (this is what Baum-Welch needs)."""
    X = two_cluster_data(n=100)
    weights = np.concatenate([np.ones(100), np.zeros(100)])
    gmm = DiagGMM.fit(X, n_components=1, weights=weights)
    assert np.allclose(gmm.means[0], X[:100].mean(axis=0), atol=1e-8)


def test_more_components_than_frames_is_clamped():
    X = np.random.default_rng(0).normal(size=(2, 3))
    gmm = DiagGMM.fit(X, n_components=10)
    assert gmm.n_components <= 2


def test_predictive_mean_modes():
    X = two_cluster_data()
    gmm = DiagGMM.fit(X, 2, seed=5)
    dominant_mean, dominant_var = gmm.predictive_mean()
    assert np.allclose(dominant_mean, gmm.means[gmm.dominant_component()])
    marginal_mean, marginal_var = gmm.predictive_mean(use_dominant=False)
    assert np.allclose(marginal_mean, gmm.weights @ gmm.means)
    assert (dominant_var > 0).all() and (marginal_var > 0).all()
    scaled = gmm.predictive_mean(covariance_scale=4.0)[1]
    assert np.allclose(scaled, dominant_var * 4.0)


def test_serialisation_roundtrip():
    X = two_cluster_data(n=80)
    gmm = DiagGMM.fit(X, 2, covariance_type="tied", seed=6)
    restored = DiagGMM.from_dict(gmm.to_dict())
    assert np.allclose(restored.weights, gmm.weights)
    assert np.allclose(restored.means, gmm.means)
    assert np.allclose(restored.variances, gmm.variances)
    assert restored.covariance_type == "tied"
    arrays = DiagGMM.from_arrays(gmm.to_arrays(), "tied")
    assert np.allclose(arrays.log_likelihood(X), gmm.log_likelihood(X))


def test_kmeans_seeding_is_within_the_data():
    X = two_cluster_data(n=60)
    rng = np.random.default_rng(0)
    centres = kmeans_plus_plus(X, 3, rng)
    assert centres.shape == (3, 3)
    assert centres.min() >= X.min() - 1e-9
    assert centres.max() <= X.max() + 1e-9
    centres, labels = kmeans(X, 2, np.random.default_rng(1))
    assert len(np.unique(labels)) >= 1
