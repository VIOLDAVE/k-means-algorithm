"""
Multi-core Monte Carlo experimental runner across all four benchmark configurations.
"""

import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast

import numpy as np
import joblib
import matplotlib
import scipy
import sklearn
from joblib import Parallel, delayed

# Allow this file to run directly, when Python places only the experiments
# directory on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kmeans import ScratchKMeans
from src.datasets import (
    generate_synthetic_separable,
    generate_synthetic_overlapping,
    generate_synthetic_high_dim,
    load_mnist_subset,
)
from src.metrics import (
    bootstrap_mean_difference_ci,
    conduct_hypothesis_tests,
    evaluate_external_labels,
    evaluate_stability,
    holm_adjust,
    summarize_runs,
)


def _run_single_trial(
    seed: int,
    k: int,
    init: Literal["random", "kmeans++"],
    X: np.ndarray,
    reference_labels: Optional[np.ndarray],
) -> tuple[float, int, float, np.ndarray, Optional[float], Optional[float]]:
    t0 = time.perf_counter()
    model = ScratchKMeans(n_clusters=k, init=init, random_state=seed)
    model.fit(X)
    runtime = time.perf_counter() - t0
    if model.labels_ is None:
        raise RuntimeError("K-means fit completed without producing labels.")
    if reference_labels is None:
        external_ari, external_nmi = None, None
    else:
        external_scores = evaluate_external_labels(model.labels_, reference_labels)
        external_ari, external_nmi = external_scores["ari"], external_scores["nmi"]
    return (
        model.inertia_,
        model.n_iter_,
        runtime,
        model.labels_,
        external_ari,
        external_nmi,
    )


def run_benchmark(
    X: np.ndarray,
    k: int,
    n_trials: int = 60,
    reference_labels: Optional[np.ndarray] = None,
    bootstrap_seed: int = 42,
):
    results: dict[str, dict[str, object]] = {}
    inertias_by_init: dict[str, list[float]] = {}

    initializers: tuple[Literal["random"], Literal["kmeans++"]] = (
        "random",
        "kmeans++",
    )
    for init in initializers:
        seeds = [2000 + i for i in range(n_trials)]
        parallel_out = cast(
            list[tuple[float, int, float, np.ndarray, Optional[float], Optional[float]]],
            Parallel(n_jobs=-1)(
                delayed(_run_single_trial)(s, k, init, X, reference_labels)
                for s in seeds
            ),
        )

        inertias = [item[0] for item in parallel_out]
        iterations = [item[1] for item in parallel_out]
        runtimes = [item[2] for item in parallel_out]
        labels = [item[3] for item in parallel_out]
        external_aris = [item[4] for item in parallel_out if item[4] is not None]
        external_nmis = [item[5] for item in parallel_out if item[5] is not None]

        summary = cast(dict[str, object], summarize_runs(inertias, iterations, runtimes))
        summary["pairwise_ari_stability"] = evaluate_stability(labels)
        summary["raw_inertias"] = inertias
        summary["raw_iterations"] = iterations
        if reference_labels is not None:
            summary["external_ari_mean"] = float(np.mean(external_aris))
            summary["external_ari_std"] = float(np.std(external_aris))
            summary["external_nmi_mean"] = float(np.mean(external_nmis))
            summary["external_nmi_std"] = float(np.std(external_nmis))
            summary["raw_external_ari"] = external_aris
            summary["raw_external_nmi"] = external_nmis
        inertias_by_init[init] = inertias
        results[init] = summary

    results["inertia_difference_bootstrap"] = cast(
        dict[str, object],
        bootstrap_mean_difference_ci(
            inertias_by_init["random"],
            inertias_by_init["kmeans++"],
            seed=bootstrap_seed,
        ),
    )
    # Statistical significance tests
    results["hypothesis_tests"] = cast(
        dict[str, object],
        conduct_hypothesis_tests(
            inertias_by_init["random"], inertias_by_init["kmeans++"]
        ),
    )
    return results


def main():
    benchmarks = {
        "separable": generate_synthetic_separable(n_samples=1500, k=5),
        "overlapping": generate_synthetic_overlapping(n_samples=1500, k=5),
        "high_dimensional": generate_synthetic_high_dim(
            n_samples=1500, k=5, d=100
        ),
        "mnist": load_mnist_subset(n_samples=3000),
    }

    experiment_records = {}
    for index, (name, (X, y)) in enumerate(benchmarks.items()):
        k = 10 if name == "mnist" else 5
        print(f"Running benchmark: {name} (n={X.shape[0]}, d={X.shape[1]}, k={k})...")
        experiment_records[name] = run_benchmark(
            X,
            k=k,
            n_trials=60,
            reference_labels=y,
            bootstrap_seed=4200 + index,
        )

    raw_p_values = {
        name: cast(
            dict[str, float],
            record["hypothesis_tests"],
        )["mann_whitney_p"]
        for name, record in experiment_records.items()
    }
    adjusted_p_values = holm_adjust(raw_p_values)
    for name, adjusted_p in adjusted_p_values.items():
        cast(dict[str, float], experiment_records[name]["hypothesis_tests"])[
            "mann_whitney_p_holm"
        ] = adjusted_p

    cache_path = Path("data/openml/openml.org/data/v1/download/52667/mnist_784.arff.gz")
    cache_timestamp = None
    if cache_path.exists():
        cache_timestamp = datetime.fromtimestamp(
            cache_path.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds")
    experiment_records["_metadata"] = {
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "matplotlib_version": matplotlib.__version__,
        "mnist_cache_timestamp_local": cache_timestamp,
        "n_trials_per_initializer": 60,
        "trial_seeds": "2000-2059",
        "bootstrap_resamples": 20_000,
        "bootstrap_seed_base": 4200,
    }

    with open("experiments/results.json", "w") as f:
        json.dump(experiment_records, f, indent=2)

    print("All trials finished successfully. Data written to experiments/results.json")


if __name__ == "__main__":
    main()