"""Diagonal-covariance Gaussian mixture models.

Everything HMS learns goes through this file.  Two properties matter more than
generality:

*data efficiency.*  Emissions support 1..K components, k-means++ initialisation,
a variance floor expressed relative to the data's own variance, and component
pruning, so a model fitted to three minutes of audio does not blow up.  ``K``
can be set per phoneme class (see `hms.config.parameters`), which is where the
real parameter budget lives.

*a shared ("tied") covariance option.*  With ``covariance_type="tied"`` all
components of one GMM share a single diagonal covariance.  That is the classic
trick for small data: the mixture only has to fit *where* the data is, not how
spread out each cluster is, and it halves the per-component parameter count.

The class also supports **weighted** fitting (``fit(X, weights=...)``), which is
what makes Baum-Welch training in `hms.core.hmm` a two-liner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Numeric floor for variances and weights.
EPS = 1e-10

#: Minimum mixture weight; components below this are merged away.
MIN_WEIGHT = 1e-4

#: Default variance floor as a fraction of the global data variance.
DEFAULT_VAR_FLOOR = 1e-3

#: Covariance representations supported by this implementation.
COVARIANCE_TYPES = ("diag", "tied")


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator
                     ) -> np.ndarray:
    """k-means++ seeding, returns initial centroids (k, D)."""
    n = len(X)
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(n)]
    closest = ((X - centers[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest.sum()
        if total <= EPS:
            centers[i] = X[rng.integers(n)]
        else:
            probs = closest / total
            centers[i] = X[rng.choice(n, p=probs)]
        closest = np.minimum(closest, ((X - centers[i]) ** 2).sum(axis=1))
    return centers


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
           iterations: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ seeding.

    Returns (centroids (k, D), hard assignments (n,)).
    """
    centers = kmeans_plus_plus(X, k, rng)
    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(iterations):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            member = X[labels == ci]
            if len(member):
                centers[ci] = member.mean(axis=0)
    return centers, labels


@dataclass
class DiagGMM:
    """Gaussian mixture with diagonal (or tied-diagonal) covariances.

    Attributes
    ----------
    weights : (K,)
    means : (K, D)
    variances : (K, D)
        Per-component variances; equal rows when ``covariance_type='tied'``.
    covariance_type : 'diag' | 'tied'
    """

    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    covariance_type: str = "diag"

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(f"covariance_type must be one of {COVARIANCE_TYPES}, "
                             f"got {self.covariance_type!r}")

        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.variances = np.asarray(self.variances, dtype=np.float64)
        if self.means.ndim == 1:
            self.means = self.means[None, :]
        if self.variances.ndim == 1:
            self.variances = self.variances[None, :]
        if self.means.ndim != 2 or self.variances.ndim != 2:
            raise ValueError("GMM means and variances must be two-dimensional")
        if self.means.shape != self.variances.shape:
            raise ValueError("GMM means and variances must have identical shapes")
        if self.means.shape[0] != len(self.weights):
            raise ValueError("GMM weights must have one value per component")
        if self.means.shape[0] == 0 or self.means.shape[1] == 0:
            raise ValueError("GMM must have at least one component and feature")
        if not (np.isfinite(self.weights).all()
                and np.isfinite(self.means).all()
                and np.isfinite(self.variances).all()):
            raise ValueError("GMM parameters must be finite")
        if (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("GMM weights must be non-negative with positive sum")
        if (self.variances <= 0).any():
            raise ValueError("GMM variances must be positive")
        if self.covariance_type == "tied" and not np.allclose(
                self.variances, self.variances[0]):
            raise ValueError("tied-covariance GMM components must share variances")
        self.weights = self.weights / self.weights.sum()

    @property
    def n_components(self) -> int:
        return len(self.weights)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def n_free_params(self) -> int:
        """Free parameters (excluding the weight simplex constraint)."""
        if self.covariance_type == "tied":
            return self.n_components * self.dim * 2 + self.n_components - 1
        return self.n_components * self.dim * 3 - 1

    @classmethod
    def single_gaussian(cls, mean: np.ndarray, variance: np.ndarray
                        ) -> "DiagGMM":
        return cls(weights=np.array([1.0]), means=np.atleast_2d(mean),
                   variances=np.atleast_2d(variance), covariance_type="diag")

    # -- estimation --------------------------------------------------------

    @classmethod
    def fit(cls, X: np.ndarray, n_components: int = 1,
            covariance_type: str = "diag", weights: Optional[np.ndarray] = None,
            n_iterations: int = 25, var_floor_ratio: float = DEFAULT_VAR_FLOOR,
            seed: int = 0, tol: float = 1e-5) -> "DiagGMM":
        """Fit a GMM by weighted EM with k-means++ initialisation.

        Data efficiency controls used here:
          * ``n_components == 1`` short-circuits to a plain weighted Gaussian
            (no EM iterations, no initialisation variance);
          * component weights below ``MIN_WEIGHT`` are pruned;
          * ``var_floor_ratio`` keeps variances away from zero, relative to the
            variance of the data actually seen -- important on tiny corpora.
        """
        if covariance_type not in COVARIANCE_TYPES:
            raise ValueError(f"covariance_type must be one of {COVARIANCE_TYPES}, "
                             f"got {covariance_type!r}")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("GMM training data must have shape (frames, features) "
                             "with both dimensions non-empty")
        if not np.isfinite(X).all():
            raise ValueError("GMM training data must contain only finite values")
        n, dim = X.shape
        if not np.isfinite(var_floor_ratio) or var_floor_ratio < 0:
            raise ValueError("var_floor_ratio must be finite and non-negative")
        n_components = int(max(1, min(n_components, max(1, n))))
        rng = np.random.default_rng(seed)
        if weights is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.ndim != 1 or weights.shape[0] != n:
                raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("weights must be finite and non-negative")
            if weights.sum() <= 0:
                raise ValueError("weights must have a positive sum")
        w = weights / weights.sum()

        global_mean = w @ X
        global_var = w @ (X - global_mean) ** 2
        floor = np.maximum(global_var * var_floor_ratio, 1e-12)

        if n_components == 1:
            mean = global_mean
            var = np.maximum(global_var, floor)
            return cls(np.array([1.0]), mean[None, :], var[None, :],
                       covariance_type)

        # --- initialisation ------------------------------------------------
        sample_idx = rng.choice(n, size=min(n, 2000), replace=False)
        sample = X[sample_idx]
        centers, labels = kmeans(sample, n_components, rng)
        means = centers.copy()
        weights_k = np.zeros(n_components)
        variances = np.zeros((n_components, dim))
        for ci in range(n_components):
            member = sample[labels == ci]
            weights_k[ci] = max(len(member), 1) / len(sample)
            if len(member) > 1:
                variances[ci] = member.var(axis=0)
            else:
                variances[ci] = global_var
        variances = np.maximum(variances, floor)
        weights_k = np.maximum(weights_k, MIN_WEIGHT)
        weights_k /= weights_k.sum()

        previous_ll = -np.inf
        for _ in range(max(1, n_iterations)):
            log_prob = cls._log_gauss(X, means, variances)
            log_w = np.log(np.maximum(weights_k, EPS))
            ll_matrix = log_prob + log_w[:, None]
            max_ll = ll_matrix.max(axis=0, keepdims=True)
            log_norm = max_ll + np.log(np.exp(ll_matrix - max_ll).sum(axis=0,
                                                                     keepdims=True))
            gamma = np.exp(ll_matrix - log_norm)          # (K, n)
            gamma_w = gamma * w[None, :]

            nk = gamma_w.sum(axis=1) + EPS
            total_ll = float((w * log_norm[0]).sum())
            new_weights = nk / nk.sum()
            new_means = (gamma_w @ X) / nk[:, None]
            diff = X[None, :, :] - new_means[:, None, :]
            new_var = (gamma_w[:, :, None] * diff ** 2).sum(axis=1) / nk[:, None]
            new_var = np.maximum(new_var, floor)

            if covariance_type == "tied":
                tied = (nk[:, None] * new_var).sum(axis=0) / nk.sum()
                new_var = np.repeat(tied[None, :], n_components, axis=0)

            weights_k, means, variances = new_weights, new_means, new_var
            if abs(total_ll - previous_ll) < tol * max(1.0, abs(previous_ll)):
                break
            previous_ll = total_ll

        return cls._pruned(weights_k, means, variances, covariance_type)

    @staticmethod
    def _pruned(weights: np.ndarray, means: np.ndarray, variances: np.ndarray,
                covariance_type: str) -> "DiagGMM":
        keep = weights >= MIN_WEIGHT
        if keep.sum() == 0:
            keep[int(np.argmax(weights))] = True
        weights, means, variances = weights[keep], means[keep], variances[keep]
        weights = weights / weights.sum()
        return DiagGMM(weights, means, variances, covariance_type)

    # -- inference ---------------------------------------------------------

    @staticmethod
    def _log_gauss(X: np.ndarray, means: np.ndarray,
                   variances: np.ndarray) -> np.ndarray:
        """Log N(x | mu, sigma^2) for each component -> (K, n)."""
        var = np.maximum(variances, EPS)
        diff = X[None, :, :] - means[:, None, :]
        return -0.5 * (np.log(2.0 * np.pi * var).sum(axis=1)[:, None]
                       + (diff ** 2 / var[:, None, :]).sum(axis=2))

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        """Per-component log likelihood -> (K, n)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return (self._log_gauss(X, self.means, self.variances)
                + np.log(np.maximum(self.weights, EPS))[:, None])

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Log p(x) for each frame -> (n,)."""
        component_ll = self.log_prob(X)
        m = component_ll.max(axis=0, keepdims=True)
        return (m + np.log(np.exp(component_ll - m).sum(axis=0,
                                                        keepdims=True)))[0]

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """Component posteriors -> (n, K)."""
        component_ll = self.log_prob(X)
        m = component_ll.max(axis=0, keepdims=True)
        p = np.exp(component_ll - m)
        return (p / p.sum(axis=0, keepdims=True)).T

    # -- use in generation -------------------------------------------------

    def dominant_component(self) -> int:
        return int(np.argmax(self.weights))

    def predictive_mean(self, covariance_scale: float = 1.0,
                         use_dominant: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(mean, variance) to hand to the trajectory generator.

        MLPG needs one Gaussian per frame, but a state has a *mixture* of
        Gaussians and, without an observation, the posterior is unknown.  Two
        conventional choices, both supported:

        ``use_dominant=True``   use the highest-weight component (HTS-style
                                "most probable mixture" selection). Crisp, and
                                what a single-speaker model wants.
        ``use_dominant=False``  collapse the mixture to its marginal moments
                                (mean of means, weighted variance + spread).
                                Smoother, more average-sounding.

        ``covariance_scale`` scales the returned variance.  Values < 1 make the
        generated trajectory follow the means more tightly (less smooth);
        values > 1 make it smoother.  This is the pragmatic counterpart of the
        variance relaxation used in HMM-based TTS.
        """
        if use_dominant:
            i = self.dominant_component()
            return self.means[i], self.variances[i] * covariance_scale
        mean = self.weights @ self.means
        var = self.weights @ (self.variances + self.means ** 2) - mean ** 2
        return mean, np.maximum(var, EPS) * covariance_scale

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "covariance_type": self.covariance_type,
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiagGMM":
        return cls(np.asarray(d["weights"], dtype=np.float64),
                   np.asarray(d["means"], dtype=np.float64),
                   np.asarray(d["variances"], dtype=np.float64),
                   d.get("covariance_type", "diag"))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights, "means": self.means,
                "variances": self.variances}

    @classmethod
    def from_arrays(cls, arrays: Dict[str, np.ndarray],
                    covariance_type: str = "diag") -> "DiagGMM":
        return cls(arrays["weights"], arrays["means"], arrays["variances"],
                   covariance_type)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<DiagGMM K={self.n_components} D={self.dim} "
                f"cov={self.covariance_type}>")
