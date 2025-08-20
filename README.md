K-Nearest Neighbors Tutorial — Theory, NumPy & scikit-learn
===========================================================

[![Releases](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/debanjan787/K-Nearest-Neighbors-Tutorial/releases)  
Download the release file from the releases page and execute the packaged notebooks or scripts: https://github.com/debanjan787/K-Nearest-Neighbors-Tutorial/releases

[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.18%2B-yellowgreen)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.22%2B-orange)](https://scikit-learn.org/)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-blueviolet)](https://jupyter.org/)

Banner
------
![KNN visualization](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

What this repo provides
-----------------------
This repository provides a full K-Nearest Neighbors (KNN) learning path. It covers theory, math, code, datasets, visualizations, and experiments. You will find:

- A clean NumPy implementation of KNN (classifier and regressor).
- Hands-on scikit-learn examples.
- Jupyter notebooks for step-by-step learning.
- Visualizations that show distance, decision boundaries, and prediction behavior.
- Tests and small experiment scripts to evaluate performance across datasets.
- Guidance on distance metrics, weighting, scaling, and hyperparameter selection.

Why this repo helps
-------------------
KNN remains a core algorithm for supervised learning. It uses simple rules, but you must tune distance, k, and preprocessing. This repo breaks the problem into small exercises. You write and run code. You visualize results. You learn what impacts predictions and why.

Topics covered
--------------
artificial-intelligence, distance-measures, documentation, jupyter-notebook, k-nearest-neighbor-classifier, k-nearest-neighbours, k-nearest-neighbours-regressor, lazy-learning-algorithm, learning-materials, machine-learning, machine-learning-algorithms, mathematics, matplotlib-pyplot, non-parametric, numpy, pandas, python, supervised-learning, tutorial-sourcecode, tutorials

Repository layout
-----------------
- README.md — this file.
- notebooks/
  - 01-intro.ipynb — KNN intuition and visualization.
  - 02-from-scratch.ipynb — implement KNN with NumPy.
  - 03-scikit-learn.ipynb — use scikit-learn for classification and regression.
  - 04-hyperparams.ipynb — hyperparameter search, cross-validation.
  - 05-advanced.ipynb — distance metrics, scaling, weighting.
- src/
  - knn_numpy.py — minimal KNN implementation (fit, predict).
  - metrics.py — distance functions (euclidean, manhattan, minkowski, cosine).
  - utils.py — helpers for scaling, splitting, and evaluation.
- data/
  - iris.csv
  - wine.csv
  - synthetic_clusters.csv
- experiments/
  - run_experiments.py — CLI to run test suites and produce plots.
- tests/
  - test_knn.py — unit tests for the NumPy KNN.
- docs/
  - math.md — mathematical derivation and complexity.
  - visualization.md — how plots were made.
- LICENSE

Quick start
-----------
1. Clone the repo.
2. Create a virtual environment and install requirements.
   - pip install -r requirements.txt
3. Open the notebooks or run the scripts.

To use the release bundle:
- Download the release file from the releases page and execute the packaged notebooks or scripts: https://github.com/debanjan787/K-Nearest-Neighbors-Tutorial/releases
- The release contains a single archive. Extract it and open notebooks in Jupyter or run setup scripts.

Releases
--------
Use the release page to get versioned bundles and prebuilt assets. Download the provided release file and execute the included notebooks or runner script. Visit the releases page here: https://github.com/debanjan787/K-Nearest-Neighbors-Tutorial/releases

Core learning modules
---------------------
- Theory and math
  - What is KNN.
  - Complexity: O(n * d) per query for brute force.
  - Bias-variance trade-off for k.
  - Non-parametric behavior and how it shapes decision boundaries.
- Distances
  - Euclidean (L2), Manhattan (L1), Minkowski.
  - Cosine similarity for high-dimension features.
  - Mahalanobis distance and when to use it.
- Scaling and preprocessing
  - Why scaling matters for distance-based methods.
  - StandardScaler vs MinMaxScaler.
  - Handling categorical features with encoding.
- KNN classifier
  - Voting schemes: uniform, distance-weighted.
  - Tie handling.
  - Multi-class predictions.
- KNN regressor
  - Mean, median, and weighted mean predictions.
  - Use cases and limitations.
- Optimization
  - KD-tree and Ball-tree overview for faster queries.
  - Approximate nearest neighbor methods.
  - When to use brute force vs index structures.

From-scratch implementation (what you will build)
-------------------------------------------------
You will implement a minimal KNN with these functions:
- fit(X, y)
- predict(X_test, k=3, metric='euclidean', weights=None)
- predict_proba for classifier
- score for accuracy or MSE

The NumPy implementation stays readable and testable. Test files show expected behavior for edge cases like ties and empty classes.

scikit-learn recipes
--------------------
The notebooks show how to:
- Use sklearn.neighbors.KNeighborsClassifier and KNeighborsRegressor.
- Wrap pipelines with StandardScaler.
- Perform GridSearchCV on k, metric, and weights.
- Use KDTree and BallTree from sklearn.neighbors for faster querying.

Datasets and experiments
------------------------
- Iris and Wine come prepackaged for calibration and visualization.
- Synthetic datasets let you control cluster shape and overlap.
- Experiment scripts run repeated trials, collect metrics, and plot:
  - Accuracy vs k
  - Decision boundary maps for 2D synthetic data
  - Error vs sample size

Example plots
-------------
- Decision boundary for k=1, k=5, k=15
- Distance weighting impact on boundaries
- Scaling effect on mixed-feature datasets

Use these plotting helpers:
- plot_decision_boundary(model, X, y)
- plot_knn_neighbors(X, query_point, k)
- plot_metric_comparison(metrics, scores)

Distance measures implemented
-----------------------------
- Euclidean (L2)
- Manhattan (L1)
- Minkowski with p parameter
- Cosine similarity (converted to distance)
- Mahalanobis (needs covariance estimate)

Hyperparameter tuning
---------------------
- k: small k reduces bias, increases variance. Large k smooths predictions.
- Metric: choose based on data scale and domain knowledge.
- Weighting: uniform vs distance.
- Scaling: standardize continuous features.

Testing and validation
----------------------
- Unit tests for algorithm correctness live in tests/.
- Example CI configuration shows how to run tests on push.
- Scripts support k-fold CV and repeated trials.

Contributing
------------
- Fork the repo.
- Create feature branches.
- Add tests for new code.
- Open a pull request with a clear description of changes.
- Follow PEP8 and use docstrings for functions.

License
-------
This project uses the MIT License. See LICENSE for details.

Credits and resources
---------------------
- scikit-learn documentation and examples for API patterns.
- NumPy and Matplotlib for numerical work and plotting.
- Papers and blog posts on nearest neighbor search, indexing, and approximate methods.

Contact
-------
Report issues or request features via GitHub Issues on the main repo page.

Useful links
------------
- KNN Releases (download and execute the release file): https://github.com/debanjan787/K-Nearest-Neighbors-Tutorial/releases
- scikit-learn KNN guide: https://scikit-learn.org/stable/modules/neighbors.html
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/

Badges and logos
----------------
![NumPy](https://numpy.org/images/logo.svg) ![Matplotlib](https://matplotlib.org/_static/images/logo2.svg) ![scikit-learn](https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png)

Run examples
------------
Run a notebook
- jupyter notebook notebooks/01-intro.ipynb

Run a script
- python experiments/run_experiments.py --dataset iris --k 5 --metric euclidean

Run tests
- pytest -q

Build tips
----------
- Use conda or venv to isolate dependencies.
- Cache data and precomputed distances for large experiments.
- Use smaller sample sizes for debugging.

Persisting results
------------------
Store experiment outputs in experiments/results/ with a timestamp. Use CSV or JSON for metrics, PNG for plots, and pickle for fitted models.

This README aims to guide both learners and practitioners. It contains runnable code, experiments, and theory.