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
        # I keep the configuration so the same estimator can be fit repeatedly
        # with explicit choices for initialization and convergence.
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
        # I use the expanded squared-distance formula to avoid building an
        # (n_samples, n_centers, n_features) difference array.
        c_sq = np.sum(centers**2, axis=1, keepdims=True).T  # Shape: (1, k)
        dist_sq = np.dot(X, centers.T)
        dist_sq *= -2.0
        dist_sq += x_sq
        dist_sq += c_sq
        # Floating-point rounding can make exact zero distances slightly negative.
        np.maximum(dist_sq, 0.0, out=dist_sq)
        return dist_sq

    def _init_random(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Standard uniform random initialization without replacement."""
        # I select distinct data rows so every initial centroid is an observed point.
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
        # I allocate once and fill the centroid matrix as each point is selected.
        centers = np.empty((self.k, n_features), dtype=np.float64)

        # 1. First center drawn uniformly at random
        first_idx = rng.randint(0, n_samples)
        centers[0] = X[first_idx]
        c0_sq = float(np.sum(centers[0] ** 2))

        # I track each observation's distance to its nearest chosen center;
        # k-means++ uses these values to weight the next selection.
        min_sq = x_sq.ravel() - 2.0 * np.dot(X, centers[0]) + c0_sq
        np.maximum(min_sq, 0.0, out=min_sq)

        # 2. Pick remaining k-1 centers with probability proportional to D(x)^2
        for j in range(1, self.k):
            total_mass = float(np.sum(min_sq))
            if total_mass > 0.0:
                # I sample from the cumulative distance mass to realize
                # probability proportional to squared distance.
                cum_mass = np.cumsum(min_sq)
                cutoff = rng.uniform(0.0, total_mass)
                next_idx = int(np.searchsorted(cum_mass, cutoff))
                if next_idx >= n_samples:
                    next_idx = n_samples - 1
            else:
                # If every point is already a chosen center (or coincident),
                # distance weighting has no mass, so I choose uniformly.
                next_idx = rng.randint(0, n_samples)

            centers[j] = X[next_idx]
            cj_sq = float(np.sum(centers[j] ** 2))

            # I update the nearest-center distance in place, avoiding a large
            # temporary matrix proportional to samples times features.
            new_sq = x_sq.ravel() - 2.0 * np.dot(X, centers[j]) + cj_sq
            np.maximum(new_sq, 0.0, out=new_sq)
            np.minimum(min_sq, new_sq, out=min_sq)

        return centers

    def fit(self, X: np.ndarray) -> "ScratchKMeans":
        # I normalize the input dtype once so distance and centroid arithmetic
        # use consistent floating-point values.
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # I reuse these squared norms during initialization and every iteration.
        x_sq = np.sum(X**2, axis=1, keepdims=True)

        if self.init == "random":
            self.centroids = self._init_random(X, rng)
        elif self.init == "kmeans++":
            self.centroids = self._init_kmeans_plus_plus(X, x_sq, rng)
        else:
            raise ValueError(f"Unknown init scheme: {self.init}")

        prev_centroids = self.centroids.copy()
        self.n_empty_fixes_ = 0

        # Lloyd's algorithm alternates assigning points and recomputing means
        # until centroid movement is small or the iteration limit is reached.
        for iteration in range(1, self.max_iter + 1):
            dist_sq = self._compute_sq_distances(X, self.centroids, x_sq)
            labels = np.argmin(dist_sq, axis=1)

            new_centroids = np.empty_like(self.centroids)
            for c_idx in range(self.k):
                mask = labels == c_idx
                if np.any(mask):
                    # The mean of assigned points minimizes their total
                    # squared distance to this centroid.
                    new_centroids[c_idx] = np.mean(X[mask], axis=0)
                else:
                    # I reseed an empty cluster at the worst-represented point
                    # so it can claim a point on the next assignment step.
                    farthest_idx = np.argmax(np.min(dist_sq, axis=1))
                    new_centroids[c_idx] = X[farthest_idx]
                    self.n_empty_fixes_ += 1

            # I measure the joint centroid movement to check convergence.
            shift = float(np.linalg.norm(new_centroids - prev_centroids))
            self.centroids = new_centroids
            prev_centroids = new_centroids.copy()
            self.n_iter_ = iteration

            if shift < self.tol:
                break

        # I recompute assignments after the last update so labels and inertia
        # describe the final centroids, including when the loop reaches max_iter.
        final_dist_sq = self._compute_sq_distances(X, self.centroids, x_sq)
        self.labels_ = np.argmin(final_dist_sq, axis=1)
        self.inertia_ = float(np.sum(np.min(final_dist_sq, axis=1)))
        return self