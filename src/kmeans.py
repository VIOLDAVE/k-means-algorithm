"""
Pure scratch implementation of Lloyd's k-means and Arthur & Vassilvitskii (2007) k-means++.
Vectorized linear algebra, invariant precomputation, and empty-cluster fallback.
Compliant with Assignment 5 specifications for Statistical Methods for Machine Learning.
"""

from typing import Literal, Optional
import numpy as np


class ScratchKMeans:
    def __init__(
        self,
        n_clusters: int,
        init: Literal["random", "kmeans++"] = "kmeans++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.k = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0
        self.n_empty_fixes_: int = 0

    @staticmethod
    def _compute_sq_distances(
        X: np.ndarray, centers: np.ndarray, x_sq: np.ndarray
    ) -> np.ndarray:
        """
        Computes pairwise ||x_i - c_j||^2 via binomial expansion:
        ||x - c||^2 = ||x||^2 - 2 * X @ C.T + ||c||^2.
        Reuses precomputed x_sq: (n, 1) to eliminate O(nd) re-allocations per iteration.
        """
        c_sq = np.sum(centers**2, axis=1, keepdims=True).T  # Shape: (1, k)
        dist_sq = np.dot(X, centers.T)
        dist_sq *= -2.0
        dist_sq += x_sq
        dist_sq += c_sq
        np.maximum(dist_sq, 0.0, out=dist_sq)
        return dist_sq

    def _init_random(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Standard uniform random initialization without replacement."""
        indices = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[indices].astype(np.float64, copy=True)

    def _init_kmeans_plus_plus(
        self, X: np.ndarray, x_sq: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Arthur & Vassilvitskii (2007) D^2-sampling.
        Guarantees E[phi] <= 8(ln k + 2) phi_opt.
        Uses O(log n) inverse transform sampling via cumulative sum and searchsorted,
        reusing x_sq to avoid allocating intermediate (n, d) matrices.
        """
        n_samples, n_features = X.shape
        centers = np.empty((self.k, n_features), dtype=np.float64)

        # 1. First center drawn uniformly at random
        first_idx = rng.randint(0, n_samples)
        centers[0] = X[first_idx]
        c0_sq = float(np.sum(centers[0] ** 2))

        # Initial minimum squared Euclidean distance: ||x - c_0||^2
        min_sq = x_sq.ravel() - 2.0 * np.dot(X, centers[0]) + c0_sq
        np.maximum(min_sq, 0.0, out=min_sq)

        # 2. Pick remaining k-1 centers with probability proportional to D(x)^2
        for j in range(1, self.k):
            total_mass = float(np.sum(min_sq))
            if total_mass > 0.0:
                cum_mass = np.cumsum(min_sq)
                cutoff = rng.uniform(0.0, total_mass)
                next_idx = int(np.searchsorted(cum_mass, cutoff))
                if next_idx >= n_samples:
                    next_idx = n_samples - 1
            else:
                next_idx = rng.randint(0, n_samples)

            centers[j] = X[next_idx]
            cj_sq = float(np.sum(centers[j] ** 2))

            # Pairwise distance update without allocating an (n, d) difference matrix
            new_sq = x_sq.ravel() - 2.0 * np.dot(X, centers[j]) + cj_sq
            np.maximum(new_sq, 0.0, out=new_sq)
            np.minimum(min_sq, new_sq, out=min_sq)

        return centers

    def fit(self, X: np.ndarray) -> "ScratchKMeans":
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Precompute ||x_i||^2 once for the entire optimization run
        x_sq = np.sum(X**2, axis=1, keepdims=True)

        if self.init == "random":
            self.centroids = self._init_random(X, rng)
        elif self.init == "kmeans++":
            self.centroids = self._init_kmeans_plus_plus(X, x_sq, rng)
        else:
            raise ValueError(f"Unknown init scheme: {self.init}")

        prev_centroids = self.centroids.copy()
        self.n_empty_fixes_ = 0

        # Lloyd's alternating assignment-update loop
        for iteration in range(1, self.max_iter + 1):
            dist_sq = self._compute_sq_distances(X, self.centroids, x_sq)
            labels = np.argmin(dist_sq, axis=1)

            new_centroids = np.empty_like(self.centroids)
            for c_idx in range(self.k):
                mask = labels == c_idx
                if np.any(mask):
                    new_centroids[c_idx] = np.mean(X[mask], axis=0)
                else:
                    # Empty cluster remedy: reseed center to point with highest error
                    farthest_idx = np.argmax(np.min(dist_sq, axis=1))
                    new_centroids[c_idx] = X[farthest_idx]
                    self.n_empty_fixes_ += 1

            shift = float(np.linalg.norm(new_centroids - prev_centroids))
            self.centroids = new_centroids
            prev_centroids = new_centroids.copy()
            self.n_iter_ = iteration

            if shift < self.tol:
                break

        final_dist_sq = self._compute_sq_distances(X, self.centroids, x_sq)
        self.labels_ = np.argmin(final_dist_sq, axis=1)
        self.inertia_ = float(np.sum(np.min(final_dist_sq, axis=1)))
        return self