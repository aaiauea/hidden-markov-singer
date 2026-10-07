"""Principal component analysis of fixed-length source vectors (NumPy only).

The source representation is a 128-sample vector per cycle -- compact compared
with the waveform, but far too many numbers to put a Gaussian mixture on.  PCA
takes it down to the 4-16 coefficients a statistical model can actually afford::

    128-sample cycle ──► PCA ──► 4-8 coefficients ──► (later: the HMM's source model)
                                     │
                                     └──► decode ──► 128-sample cycle

Why a hand-written PCA
----------------------
``sklearn`` is a dependency HMS does not have and does not want, and the job is
twenty lines of linear algebra.  What is worth getting right, and what this
module does explicitly:

* **Determinism.**  The basis is computed with an SVD of the centred data (not
  an eigendecomposition of a covariance matrix, which squares the condition
  number), the sign of every component is fixed by a rule (largest-magnitude
  entry positive), and directions the data does not constrain (a singular value
  at numerical zero, e.g. more components than the data has rank) are filled
  with orthonormalised canonical basis vectors instead of whatever LAPACK
  happened to return.  Fitting the same vectors twice, on any machine, gives
  the same basis -- so a saved basis and a live one agree bit for bit.
* **Numerical stability.**  Raw moments, not ``x - mean`` twice; a rank
  tolerance relative to the largest singular value; zero-variance directions
  kept as valid (zero) directions rather than producing NaN coefficients.
* **What comes next is recorded.**  Besides the mean and the basis, the object
  carries the eigenvalue of every component (the variance a model should give
  each coefficient) and the variance left *outside* the retained subspace --
  the two numbers a source HMM will need, available without a second pass over
  the data.
* **Round-tripping is measured, not assumed.**  :meth:`SourcePCA.report`
  returns the MSE, RMSE and explained variance of a reconstruction, which is
  what the Phase 1 test suite and ``tools/bench_source_pca.py`` quote.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

#: Components kept by default: enough for the dominant excitation shape, few
#: enough that a diagonal Gaussian on the coefficients stays cheap.
DEFAULT_N_COMPONENTS = 8

#: Singular values below ``tol * largest`` are treated as "no variance here".
_RANK_RTOL = 1e-10


class SourcePCA:
    """PCA basis for fixed-length source vectors.

        pca = SourcePCA.fit(cycles, n_components=8)
        coefficients = pca.encode(cycles)          # (K, 8)
        cycles_again = pca.decode(coefficients)    # (K, cycle_length)
        pca.save("source_pca.npz")

    The object is a plain container of arrays: ``mean`` (cycle_length,),
    ``components`` (n_components, cycle_length), ``eigenvalues`` (n_components,),
    ``total_variance``, ``residual_variance`` and ``n_samples``.  Nothing about
    training is hidden in it, so a model file for the future source HMM is a
    serialisation of exactly this.

    Directions with no variance behind them (more components than the data has
    rank, or dropped by ``variance_floor``) are still carried as a deterministic
    orthonormal fill so the basis is always complete and well-formed, but the
    codec ignores them -- see :attr:`active`.
    """

    def __init__(self, mean: np.ndarray, components: np.ndarray,
                 eigenvalues: Optional[np.ndarray] = None, total_variance: float = 0.0,
                 residual_variance: float = 0.0, n_samples: int = 0) -> None:
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        components = np.atleast_2d(np.asarray(components, dtype=np.float64))
        if mean.size == 0:
            raise ValueError("mean must have at least one entry")
        if components.shape[1] != mean.size:
            raise ValueError(f"components must have {mean.size} columns to match "
                             f"the mean, got {components.shape}")
        if not np.isfinite(mean).all() or not np.isfinite(components).all():
            raise ValueError("PCA mean and components must be finite")
        if eigenvalues is None:
            eigenvalues = np.zeros(components.shape[0])
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
        if eigenvalues.size != components.shape[0]:
            raise ValueError("eigenvalues must have one entry per component")
        self.mean = mean
        self.components = components
        self.eigenvalues = eigenvalues
        self.total_variance = float(total_variance)
        self.residual_variance = float(residual_variance)
        self.n_samples = int(n_samples)

    # -- properties --------------------------------------------------------

    @property
    def cycle_length(self) -> int:
        return int(self.mean.size)

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Variance share of each component (zero when nothing was fitted)."""
        if self.total_variance <= 0.0:
            return np.zeros(self.n_components)
        return np.clip(self.eigenvalues / self.total_variance, 0.0, 1.0)

    @property
    def active(self) -> np.ndarray:
        """Which components carry variance (and therefore participate).

        A fitted basis always has ``n_components`` orthonormal directions -- for
        directions the data does not constrain, the fill is a deterministic
        canonical vector instead of whatever LAPACK returned -- but only the
        ones with variance behind them are used by :meth:`encode`.  Fitting 16
        components of data that spans 8 directions therefore yields 8 zero
        coefficients, not 8 coefficients of noise to model later.
        """
        return self.eigenvalues > 0.0

    @property
    def cumulative_explained_variance(self) -> float:
        """Variance share retained by all components together, in [0, 1]."""
        return float(np.clip(self.explained_variance_ratio.sum(), 0.0, 1.0))

    @property
    def component_std(self) -> np.ndarray:
        """Std dev of each coefficient (``sqrt(eigenvalue)``), for a later model."""
        return np.sqrt(np.maximum(self.eigenvalues, 0.0))

    @property
    def residual_std(self) -> float:
        """Std dev per sample of the part of a cycle outside the subspace."""
        return float(np.sqrt(max(self.residual_variance, 0.0)))

    def __len__(self) -> int:
        return self.n_components

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<SourcePCA components={self.n_components} "
                f"cycle_length={self.cycle_length} "
                f"explained={self.cumulative_explained_variance:.3f}>")

    # -- fitting -----------------------------------------------------------

    @classmethod
    def fit(cls, cycles: np.ndarray, n_components: int = DEFAULT_N_COMPONENTS,
            variance_floor: float = 0.0) -> "SourcePCA":
        """Fit a PCA basis to ``(N, cycle_length)`` source vectors.

        ``n_components`` is clipped to the vector length (asking for 16
        components of a 12-sample vector is a caller error, but not one worth a
        crash in the middle of a benchmark) and never exceeds the number of
        training vectors... except that it may: the extra directions are simply
        fitted with zero variance and a deterministic orthonormal fill, so
        :meth:`encode` still returns a well-formed row.

        ``variance_floor`` discards directions whose singular value is below
        ``variance_floor`` (absolute, in the units of the data); the default 0
        keeps everything the rank tolerance accepts.
        """
        x = np.atleast_2d(np.asarray(cycles, dtype=np.float64))
        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("fit needs a non-empty (n_samples, cycle_length) array")
        if not np.isfinite(x).all():
            raise ValueError("source vectors must be finite to fit a PCA")
        n_components = int(n_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        n_components = min(n_components, x.shape[1])

        mean = x.mean(axis=0)
        centred = np.ascontiguousarray(x - mean)
        # full_matrices=False: only the directions the data can support, and the
        # covariance matrix is never formed (SVD of the data is better
        # conditioned than an eigendecomposition of X^T X).
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        denominator = max(x.shape[0] - 1, 1)
        eigenvalues = (singular ** 2) / denominator
        total_variance = float(np.sum(eigenvalues))

        largest = float(singular[0]) if singular.size else 0.0
        tolerance = max(largest * _RANK_RTOL * max(x.shape), 0.0)
        if variance_floor > 0:
            tolerance = max(tolerance, float(np.sqrt(max(variance_floor, 0.0)
                                                    * denominator)))
        rank = int(np.sum(singular > tolerance)) if singular.size else 0

        basis = []
        kept = np.zeros(n_components, dtype=np.float64)
        for i in range(n_components):
            if i < rank:
                vector = vt[i].astype(np.float64, copy=True)
            else:
                vector = _orthonormal_fill(basis, x.shape[1])
                vector = vector if vector is not None else np.zeros(x.shape[1])
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0:
                vector = -vector
            basis.append(vector)
            if i < rank and i < len(eigenvalues):
                kept[i] = eigenvalues[i]
        components = np.asarray(basis, dtype=np.float64)

        explained = float(np.sum(kept))
        residual_variance = max(total_variance - explained, 0.0) / max(x.shape[1], 1)
        return cls(mean=mean, components=components, eigenvalues=kept,
                   total_variance=total_variance,
                   residual_variance=residual_variance, n_samples=x.shape[0])

    # -- codec -------------------------------------------------------------

    def encode(self, cycles: np.ndarray) -> np.ndarray:
        """``(N, cycle_length)`` vectors -> ``(N, n_components)`` coefficients.

        Components without variance behind them are zeroed rather than
        projected, so ``decode(encode(x))`` is always the mean plus the retained
        subspace -- never an arbitrary direction the fit did not really find.
        """
        x = _as_matrix(cycles, self.cycle_length)
        coefficients = (x - self.mean) @ self.components.T
        inactive = ~self.active
        if inactive.any():
            coefficients = np.where(inactive[None, :], 0.0, coefficients)
        return coefficients

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """``(N, n_components)`` coefficients -> ``(N, cycle_length)`` vectors."""
        c = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
        if c.shape[1] != self.n_components:
            raise ValueError(f"coefficients must have {self.n_components} columns, "
                             f"got {c.shape}")
        if c.size and not np.isfinite(c).all():
            raise ValueError("coefficients must be finite")
        return c @ self.components + self.mean

    def reconstruct(self, cycles: np.ndarray) -> np.ndarray:
        """Encode and decode: the best reconstruction in the retained subspace."""
        return self.decode(self.encode(cycles))

    # -- measurement -------------------------------------------------------

    def reconstruction_errors(self, cycles: np.ndarray) -> np.ndarray:
        """Per-vector relative RMSE, ``||x - x_hat|| / ||x||`` (``(N,)``).

        The summary in :meth:`report` is a mean over vectors and is therefore
        dominated by whichever cycles are hardest -- which is useful for a
        benchmark (it answers "how much energy is left behind overall") but
        hides whether *most* cycles are fine.  This is the per-cycle view.
        """
        x = _as_matrix(cycles, self.cycle_length)
        if x.shape[0] == 0:
            return np.zeros(0)
        denominator = np.linalg.norm(x, axis=1)
        numerator = np.linalg.norm(x - self.reconstruct(x), axis=1)
        return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-30), 0.0)

    def report(self, cycles: np.ndarray) -> Dict[str, float]:
        """Reconstruction quality of ``cycles`` under this basis.

        Returns ``n_vectors``, ``mse``, ``rmse``, ``relative_rmse`` (RMSE
        divided by the vectors' own RMS, so it is comparable across material),
        ``median_relative_error`` / ``p90_relative_error`` (per-cycle, so a few
        pathological cycles cannot hide the typical case), ``mean_correlation``
        (average cosine between each vector and its reconstruction),
        ``explained_variance`` (``1 - mse / variance`` of *this* data) and
        ``cumulative_explained_variance`` of the fitted basis itself.  Empty
        input reports NaN errors and is not an error: a benchmark with no
        usable cycles should print "nothing", not crash.
        """
        x = _as_matrix(cycles, self.cycle_length)
        if x.shape[0] == 0:
            return {"n_vectors": 0.0, "mse": float("nan"), "rmse": float("nan"),
                    "relative_rmse": float("nan"), "median_relative_error":
                    float("nan"), "p90_relative_error": float("nan"),
                    "explained_variance": float("nan"),
                    "cumulative_explained_variance": self.cumulative_explained_variance}
        error = x - self.reconstruct(x)
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))
        scale = float(np.mean((x - x.mean(axis=0)) ** 2))
        rms = float(np.sqrt(np.mean(x ** 2)))
        per_cycle = self.reconstruction_errors(x)
        numerator = np.sum(x * self.reconstruct(x), axis=1)
        denominator = (np.linalg.norm(x, axis=1)
                       * np.linalg.norm(self.reconstruct(x), axis=1))
        return {
            "n_vectors": float(x.shape[0]),
            "mse": mse,
            "rmse": rmse,
            "relative_rmse": float(rmse / rms) if rms > 0 else float("nan"),
            "median_relative_error": float(np.median(per_cycle)),
            "mean_correlation": float(np.mean(np.where(
                denominator > 0, numerator / np.maximum(denominator, 1e-30), 0.0))),
            "p90_relative_error": float(np.percentile(per_cycle, 90)),
            "explained_variance": float(1.0 - mse / scale) if scale > 0 else 1.0,
            "cumulative_explained_variance": self.cumulative_explained_variance,
        }

    # -- serialisation -----------------------------------------------------

    def save(self, path) -> Path:
        """Write the basis to a compressed ``.npz`` (deterministic contents)."""
        path = _with_npz_suffix(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps({
            "format": "hms.source.pca",
            "version": 1,
            "cycle_length": self.cycle_length,
            "n_components": self.n_components,
            "n_samples": self.n_samples,
        }, sort_keys=True)
        np.savez_compressed(path, mean=self.mean, components=self.components,
                            eigenvalues=self.eigenvalues,
                            total_variance=np.float64(self.total_variance),
                            residual_variance=np.float64(self.residual_variance),
                            n_samples=np.int64(self.n_samples),
                            meta=np.asarray(meta))
        return path

    @classmethod
    def load(cls, path) -> "SourcePCA":
        """Read a basis written by :meth:`save`; rejects anything malformed."""
        with np.load(str(path), allow_pickle=False) as handle:
            missing = [k for k in ("mean", "components") if k not in handle.files]
            if missing:
                raise ValueError(f"{path} is not a source PCA basis "
                                 f"(missing {', '.join(missing)})")
            mean = handle["mean"]
            components = handle["components"]
            eigenvalues = handle["eigenvalues"] if "eigenvalues" in handle.files else None
            total = float(handle["total_variance"]) if "total_variance" in handle.files else 0.0
            residual = (float(handle["residual_variance"])
                        if "residual_variance" in handle.files else 0.0)
            n_samples = int(handle["n_samples"]) if "n_samples" in handle.files else 0
        return cls(mean=mean, components=components, eigenvalues=eigenvalues,
                   total_variance=total, residual_variance=residual,
                   n_samples=n_samples)


def _orthonormal_fill(basis: list, length: int) -> Optional[np.ndarray]:
    """A canonical basis vector orthogonal to ``basis`` (Gram-Schmidt).

    Used for directions the data does not constrain, so a fitted basis is
    deterministic even when a singular value is numerically zero (LAPACK is
    free to return any orthonormal complement there).
    """
    for j in range(length):
        candidate = np.zeros(length, dtype=np.float64)
        candidate[j] = 1.0
        for vector in basis:
            candidate -= vector * float(candidate @ vector)
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-8:
            return candidate / norm
    return None


def _as_matrix(cycles: np.ndarray, length: int) -> np.ndarray:
    x = np.atleast_2d(np.asarray(cycles, dtype=np.float64))
    if x.shape[1] != length:
        raise ValueError(f"source vectors must have {length} samples, got "
                         f"{x.shape}")
    if x.size and not np.isfinite(x).all():
        raise ValueError("source vectors must be finite")
    return x


def _with_npz_suffix(path) -> Path:
    path = Path(path)
    return path if str(path).endswith(".npz") else Path(str(path) + ".npz")


__all__ = ["SourcePCA", "DEFAULT_N_COMPONENTS"]
