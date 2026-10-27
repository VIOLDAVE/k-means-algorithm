"""
Synthetic data-generating processes and OpenML automated data caching.
"""

from typing import Tuple
import numpy as np
from sklearn.datasets import make_blobs, fetch_openml


def generate_synthetic_separable(
    n_samples: int = 1500, k: int = 5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """2D synthetic dataset with well-separated Gaussian components (Delta mu >> sigma)."""
    blobs = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=k,
        cluster_std=0.75,
        center_box=(-12.0, 12.0),
        random_state=seed,
        return_centers=False,
    )
    return blobs[0], blobs[1]


def generate_synthetic_overlapping(
    n_samples: int = 1500, k: int = 5, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """2D synthetic extension: high-variance overlapping densities (challenging local minima)."""
    blobs = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=k,
        cluster_std=2.4,
        center_box=(-8.0, 8.0),
        random_state=seed,
        return_centers=False,
    )
    return blobs[0], blobs[1]


def generate_synthetic_high_dim(
    n_samples: int = 1500, k: int = 5, d: int = 100, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """High-dimensional synthetic extension: d=100 to study distance concentration."""
    blobs = make_blobs(
        n_samples=n_samples,
        n_features=d,
        centers=k,
        cluster_std=1.5,
        random_state=seed,
        return_centers=False,
    )
    return blobs[0], blobs[1]


def load_mnist_subset(
    n_samples: int = 3000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multiclass real-world benchmark (MNIST, d=784, k=10).
    Caches locally to ./data without committing raw weights to git.
    """
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home="./data",
        parser="liac-arff",
    )
    X, y = mnist.data.astype(np.float64), mnist.target.astype(int)

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X), size=n_samples, replace=False)
    # Scale intensities to [0, 1] range
    return X[indices] / 255.0, y[indices]