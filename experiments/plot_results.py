"""
Generates publication-quality figures for the LaTeX report:
1. Final empirical risk (inertia) distributions across all benchmarks.
2. Lloyd iteration counts to convergence.
3. Voronoi partition comparison: bad local minimum (random) vs. optimal (k-means++).
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# I add the project root so this script can import the project modules when run
# directly from the experiments directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kmeans import ScratchKMeans
from src.datasets import generate_synthetic_separable, load_mnist_subset


def plot_exploratory_projections():
    """Recreate the notebook's PCA and t-SNE MNIST label visualisations."""
    # I use a fixed sample and random seed so these visual comparisons are repeatable.
    X, y = load_mnist_subset(n_samples=3000, seed=42)
    # PCA shows the strongest global directions of variation; t-SNE emphasizes
    # local neighborhoods, so the two views reveal different structure.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    X_tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
    ).fit_transform(X)

    fig, (ax_pca, ax_tsne) = plt.subplots(
        1, 2, figsize=(14, 6), layout="constrained"
    )
    # I keep the same class colors in both panels so digit groups are easy to compare.
    scatter_pca = ax_pca.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", vmin=-0.5, vmax=9.5,
        alpha=0.6, s=10
    )
    ax_pca.set_title("PCA Projection of MNIST (Global Structure)")
    ax_pca.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax_pca.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")

    scatter_tsne = ax_tsne.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="tab10", vmin=-0.5, vmax=9.5,
        alpha=0.6, s=10
    )
    ax_tsne.set_title("t-SNE Projection of MNIST (Local Structure)")
    ax_tsne.set_xlabel("t-SNE component 1")
    ax_tsne.set_ylabel("t-SNE component 2")

    fig.colorbar(
        scatter_tsne,
        ax=[ax_pca, ax_tsne],
        ticks=np.arange(10),
        location="right",
        label="Digit class",
    )
    # I create the output folder when needed so the script works from a fresh checkout.
    figures_dir = Path("report/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "mnist_pca_tsne.pdf", bbox_inches="tight")
    fig.savefig(
        figures_dir / "mnist_pca_tsne.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    print("Exploratory PCA and t-SNE projections saved to report/figures/.")


def plot_boxplots():
    # These plots use experiment results rather than rerunning the benchmark.
    results_path = Path("experiments/results.json")
    if not results_path.exists():
        # I skip only these plots when results are missing; the other figures can
        # still be generated independently.
        print("Warning: experiments/results.json not found. Run run_experiments.py first.")
        return

    # Each dataset contains repeated runs for both initialisation strategies.
    with open(results_path, "r") as f:
        results = json.load(f)

    # I keep labels and dataset keys in matching order for the four-panel figures.
    datasets = ["separable", "overlapping", "high_dimensional", "mnist"]
    names = [
        "Separable (d=2, k=5)",
        "Overlapping (d=2, k=5)",
        "High-Dim (d=100, k=5)",
        "MNIST (d=784, k=10)",
    ]

    figures_dir = Path("report/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # I compare final inertia distributions to show how initialization affects
    # the quality and variability of the resulting clusters.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, ds, title in zip(axes, datasets, names):
        rnd = results[ds]["random"]["raw_inertias"]
        kpp = results[ds]["kmeans++"]["raw_inertias"]
        ax.boxplot([rnd, kpp], tick_labels=["Random", "k-means++"], patch_artist=True)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Final Inertia $\\phi(C)$")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "inertia_distributions.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "inertia_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    # I also compare iteration counts because a lower final inertia does not
    # necessarily mean that a method converges in fewer steps.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, ds, title in zip(axes, datasets, names):
        rnd_it = results[ds]["random"]["raw_iterations"]
        kpp_it = results[ds]["kmeans++"]["raw_iterations"]
        ax.boxplot([rnd_it, kpp_it], tick_labels=["Random", "k-means++"], patch_artist=True)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Iterations to Convergence")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "convergence_iterations.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "convergence_iterations.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Boxplots saved to report/figures/.")


def plot_voronoi_comparison():
    """
    Qualitative 2D comparison demonstrating the split/merge failure mode:
    - Random: trapped in a local minimum where one cluster is split and two others share a center.
    - k-means++: cleanly isolates all 5 true clusters.
    """
    figures_dir = Path("report/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # I use a fixed dataset so differences between the panels come from
    # initialization, not from different generated points.
    X, _ = generate_synthetic_separable(n_samples=1500, k=5, seed=42)

    # Random initialization can settle in different local minima. I search a
    # bounded seed range for a representative split/merge result from the experiments.
    bad_model = None
    for seed in range(2000, 2200):
        model = ScratchKMeans(n_clusters=5, init="random", random_state=seed).fit(X)
        if 3900.0 < model.inertia_ < 4300.0:
            bad_model = model
            break

    # I use the known successful seed to make the k-means++ comparison reproducible.
    good_model = ScratchKMeans(n_clusters=5, init="kmeans++", random_state=2000).fit(X)

    if bad_model is None:
        print("Warning: Could not isolate a split/merge run within seed range.")
        return

    # I evaluate the nearest-centroid rule over a regular grid to visualize the
    # Voronoi regions, including areas between the observed data points.
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    # The distance helper can reuse each grid point's squared norm for every
    # centroid, avoiding repeated work across the full grid.
    g_sq = np.sum(grid**2, axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    models = [bad_model, good_model]
    titles = [
        f"Uniform Random: Suboptimal Local Min (Inertia: {bad_model.inertia_:.1f})",
        f"k-means++: Optimal Voronoi Partition (Inertia: {good_model.inertia_:.1f})",
    ]

    for ax, model, title in zip(axes, models, titles):
        # I assign each grid point to its closest centroid; those assignments
        # define the colored decision regions shown below.
        dist_sq = model._compute_sq_distances(grid, model.centroids, g_sq)
        Z = np.argmin(dist_sq, axis=1).reshape(xx.shape)

        # Plot background Voronoi cells
        ax.contourf(xx, yy, Z, alpha=0.25, cmap="tab10")
        ax.contour(xx, yy, Z, colors="gray", linewidths=0.6, alpha=0.5)

        # Plot data points colored by cluster assignment
        ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=10, cmap="tab10", alpha=0.65)

        # Plot centroids
        ax.scatter(
            model.centroids[:, 0],
            model.centroids[:, 1],
            c="black",
            marker="X",
            s=160,
            edgecolors="white",
            linewidth=1.5,
            label="Centroids",
            zorder=5,
        )

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(figures_dir / "voronoi_comparison.pdf", bbox_inches="tight")
    plt.savefig(figures_dir / "voronoi_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Voronoi comparison saved to report/figures/voronoi_comparison.pdf.")


def main():
    # I generate each figure group in one place so the full report can be
    # refreshed by running this script once.
    plot_exploratory_projections()
    plot_boxplots()
    plot_voronoi_comparison()


if __name__ == "__main__":
    main()