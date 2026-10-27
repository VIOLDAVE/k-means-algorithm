# K-means Initialisation Study

A from-scratch implementation of Lloyd's k-means algorithm and k-means++
initialisation, with repeated-run experiments on synthetic Gaussian blobs and
an MNIST subset. The project compares inertia, convergence, partition
stability, and external label scores. The final report is
[report/report.pdf](report/report.pdf), with its LaTeX source in `report/`.

## Project contents

- `src/`: k-means/k-means++ implementation, dataset loaders, and evaluation
  metrics.
- `experiments/run_experiments.py`: reproducible 60-start benchmarks and
  `experiments/results.json`, including environment metadata.
- `experiments/plot_results.py`: regenerate benchmark plots and the exploratory
  MNIST PCA/t-SNE figure.
- `main.ipynb`: exploratory notebook.
- `report/`: LaTeX report, bibliography, figures, and submitted PDF.

## Setup and reproduction

Python 3.12 was used for the reported runs. Install dependencies from the
project root:

```bash
python -m pip install -r requirements.txt
```

Run all benchmarks and regenerate the figures:

```bash
python experiments/run_experiments.py
python experiments/plot_results.py
```

On the first run, MNIST is fetched from OpenML and cached under `data/`.
The generated dataset seeds, MNIST subset seed, trial seeds, package versions,
and bootstrap settings are recorded in `experiments/results.json`. The local
MNIST cache and virtual environment are intentionally excluded from version
control; the benchmark results and final report PDF are included.

To rebuild the PDF, install a LaTeX distribution with `latexmk`, then run:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```