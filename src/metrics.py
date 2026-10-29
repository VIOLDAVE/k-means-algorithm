"""
Statistical evaluation metrics: Inertia statistics, Pairwise ARI Stability, and Hypothesis Testing.
"""

from typing import Dict, List, cast
import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate_stability(labels_list: List[np.ndarray]) -> float:
    """
    Calculates the mean pairwise Adjusted Rand Index across all (T choose 2) independent runs.
    Measures the reproducibility of the partition modes discovered by the initializer.
    """
    n_runs = len(labels_list)
    if n_runs < 2:
        # With one or no runs there is no pair to compare; treat this as
        # perfectly stable rather than returning an undefined mean.
        return 1.0

    # I compare every unique pair once; ARI is invariant to cluster label names.
    scores = [
        adjusted_rand_score(labels_list[i], labels_list[j])
        for i in range(n_runs)
        for j in range(i + 1, n_runs)
    ]
    return float(np.mean(scores))


def evaluate_external_labels(
    predicted: np.ndarray, reference: np.ndarray
) -> Dict[str, float]:
    """Score a fitted partition against reference labels without using them to fit."""
    # I report complementary agreement scores only after fitting, so reference
    # labels evaluate the result without influencing cluster assignments.
    return {
        "ari": float(adjusted_rand_score(reference, predicted)),
        "nmi": float(normalized_mutual_info_score(reference, predicted)),
    }


def summarize_runs(
    inertias: List[float], iterations: List[int], runtimes: List[float]
) -> Dict[str, float]:
    # I compute quartiles once because their difference summarizes spread
    # without being dominated by extreme runs.
    q75, q25 = np.percentile(inertias, [75, 25])
    return {
        "inertia_mean": float(np.mean(inertias)),
        "inertia_std": float(np.std(inertias)),
        "inertia_min": float(np.min(inertias)),
        "inertia_max": float(np.max(inertias)),
        "inertia_iqr": float(q75 - q25),
        "avg_iterations": float(np.mean(iterations)),
        "avg_runtime_ms": float(np.mean(runtimes) * 1000.0),
    }


def bootstrap_mean_difference_ci(
    random_inertias: List[float],
    kmeans_plus_plus_inertias: List[float],
    seed: int,
    n_resamples: int = 20_000,
) -> Dict[str, float]:
    """Percentile bootstrap CI for independent-run mean difference (k-means++ - random)."""
    random_values = np.asarray(random_inertias, dtype=np.float64)
    kpp_values = np.asarray(kmeans_plus_plus_inertias, dtype=np.float64)
    # I use an explicit generator so resampling is reproducible and independent
    # of global NumPy random state.
    rng = np.random.RandomState(seed)
    # Each row represents one bootstrap sample drawn with replacement from
    # the corresponding set of independent trial results.
    random_means = random_values[
        rng.randint(0, random_values.size, size=(n_resamples, random_values.size))
    ].mean(axis=1)
    kpp_means = kpp_values[
        rng.randint(0, kpp_values.size, size=(n_resamples, kpp_values.size))
    ].mean(axis=1)
    # I subtract matched bootstrap means to estimate the interval for the
    # k-means++ minus random mean difference.
    lower, upper = np.percentile(kpp_means - random_means, [2.5, 97.5])
    return {
        "mean_difference_kpp_minus_random": float(kpp_values.mean() - random_values.mean()),
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
    }


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Apply Holm's step-down family-wise error correction to a set of p-values."""
    # Holm's procedure orders p-values before scaling them by the number of
    # remaining tests, controlling family-wise error across the benchmark set.
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    n_tests = len(ordered)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        # The running maximum enforces monotonic adjusted p-values as ranks grow.
        running_max = max(running_max, (n_tests - rank) * p_value)
        adjusted[name] = float(min(running_max, 1.0))
    return adjusted


def conduct_hypothesis_tests(rand_inertias: List[float], kpp_inertias: List[float]) -> Dict[str, float]:
    """
    Conducts non-parametric Mann-Whitney U test (stochastic dominance)
    and Levene test for equality of variance.
    """
    # Mann-Whitney tests whether k-means++ tends to have lower inertia;
    # Levene separately checks whether the two strategies differ in variance.
    u_stat, u_pval = stats.mannwhitneyu(kpp_inertias, rand_inertias, alternative="less")
    lev_stat, lev_pval = cast(
        tuple[float, float], stats.levene(rand_inertias, kpp_inertias)
    )
    return {
        "mann_whitney_u": float(cast(float, u_stat)),
        "mann_whitney_p": float(cast(float, u_pval)),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_pval),
    }