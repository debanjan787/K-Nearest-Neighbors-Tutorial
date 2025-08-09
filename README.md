<div align="center">

# K-Nearest Neighbors (KNN) Tutorial 📘

Comprehensive, concept-to-code walkthrough of the KNN algorithm for both **classification** and **regression**: theory, intuition, math, helper utilities, notebook experimentation, and a roadmap for extending to a full reusable implementation.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#13-installation--environment-setup) [![Tutorial Status](https://img.shields.io/badge/Status-Active-orange.svg)](#) [![Contributions](https://img.shields.io/badge/Contributions-Welcome-success.svg)](#-contributions) [![Made With](https://img.shields.io/badge/Made%20with-NumPy%20%7C%20Pandas%20%7C%20Matplotlib-informational.svg)](#14-quick-start-examples) [![Jupyter](https://img.shields.io/badge/Notebook-Supported-F37626.svg?logo=jupyter&logoColor=white)](notebook/practice.ipynb)

</div>

---

<div align="justify">

## 🔍 Table of Contents

1. Motivation & Goals
2. What Is KNN? (High-Level Intuition)
3. Key Concepts (Supervised / Instance-Based / Non-Parametric)
4. When to Use KNN / When Not To
5. Mathematical Foundations
6. Distance Metrics (Focus: L2 / Euclidean)
7. Choosing K (Bias–Variance & Practical Heuristics)
8. Classification vs Regression Behavior
9. Handling Ties, Weights & Edge Cases
10. Data Preparation (Scaling, Cleaning, Dimensionality)
11. Complexity & Performance Considerations
12. Repository Structure
13. Installation & Environment Setup
14. Quick Start Examples (From-Scratch Logic & Helper Utilities)
15. Typical Workflow (End‑to‑End)
16. Evaluation Metrics (Classification & Regression)
17. Visualizations & Exploratory Analysis
18. Extending the Project (Roadmap)
19. Common Pitfalls & Debugging Checklist
20. Reference Reading & Further Study
21. License

---

## 1. Motivation & Goals

This tutorial aims to give you BOTH the theory (why KNN works, its trade‑offs) and the practice (how to implement core pieces, experiment in a notebook, evaluate, and optimize). It is intentionally incremental so you can:

- Start with conceptual foundations.
- Use helper utilities (distance calculation, majority/mean aggregation).
- Prototype classification & regression in a notebook.
- Later add a reusable KNN class (see roadmap).

## 2. What Is KNN? (High-Level Intuition)

KNN predicts the label (class or numeric value) of a query point by looking at the K most similar (closest) points in the training set. No parameters are “fit” in advance—prediction is deferred until you ask a question of the model. It is sometimes called a **memory-based**, **lazy**, or **instance-based** method.

## 3. Key Concepts

Linked docs in `docs/` expand each topic:

- Supervised Learning (`supervised-learning.md`): labeled examples drive learning.
- Classification / Regression (`classification-problem.md`, `regression-problem.md`).
- Instance-Based (`instance-base.md`): store all examples; compare at query time.
- Non-Parametric (`non-parametric.md`): flexibility grows with data volume.

## 4. When to Use / When Not To

Use KNN when:

- Decision boundary is irregular / hard to parametrize.
- Dataset is small-to-medium (memory fits; latency acceptable).
- You want a strong baseline quickly.

Avoid or be cautious when:

- Very large datasets → prediction cost O(N \* d) per query.
- High-dimensional feature spaces (curse of dimensionality).
- Features on different scales (must scale/normalize first).
- Many irrelevant/noisy features (distance metric degrades).

## 5. Mathematical Foundations (Core Mechanics)

Given training set T = {(xᵢ, yᵢ)} for i = 1..N and a query point x\*:

1. Compute distance d(x\*, xᵢ) for all i.
2. Select K indices of smallest distances: N_K.
3. Classification: predict mode({ yᵢ | i ∈ N_K }).
4. Regression: predict average or weighted average of { yᵢ | i ∈ N_K }.

### Pseudocode (Generic)

```
function predict(x*, X_train, y_train, K, distance_fn, task, weight=False):
	distances = []
	for i in range(len(X_train)):
		d = distance_fn(x*, X_train[i])
		distances.append( (d, y_train[i]) )
	distances.sort by d ascending
	neighbors = first K elements
	if task == 'classification':
		return majority_vote([label for (_, label) in neighbors])
	else: # regression
		if weight:
			return weighted_mean(neighbors, w = 1 / (d + ε))
		else:
			return simple_mean(neighbors)
```

## 6. Distance Metrics (Focus: L2)

The Euclidean (L2) distance is most common (see `L2-distance.md`). Other options:

- Manhattan (L1)
- Minkowski (generalized p-norm)
- Cosine distance (if direction matters more than magnitude)
- Hamming distance (categorical / binary vectors)
  Metric choice affects neighbor relationships—experiment.

## 7. Choosing K

Trade‑off:

- Small K → low bias, high variance (sensitive to noise).
- Large K → smoother, higher bias (can underfit).
  Heuristics:
- Try odd K for binary classification (avoid ties).
- Start with sqrt(N) as a coarse upper bound to explore.
- Use cross‑validation grid (e.g., K in {1,3,5,7,11,15,…}).

## 8. Classification vs Regression Behavior

- Classification: Majority vote; can introduce **distance weighting** (e.g., weight = 1 / (d + ε)).
- Regression: Mean vs weighted mean; handle zero distance (exact duplicate) by immediately returning that label/value or assigning infinite weight (see helper function approach).

## 9. Handling Ties, Weights & Edge Cases

Edge considerations:

- Ties in classification: break by (a) smallest cumulative distance, (b) deterministic ordering, or (c) reducing K by 1 recursively.
- Distance zero: identical point encountered—shortcut return.
- Missing values: impute before KNN (mean / median / KNN-imputer) else distance invalid.
- Feature scaling: always scale numerical features (e.g., StandardScaler / MinMaxScaler) to prevent dominance.

## 10. Data Preparation

Recommended pipeline:

1. Clean anomalies / duplicates.
2. Encode categorical variables (one-hot, ordinal if appropriate).
3. Scale numeric features.
4. Optionally reduce dimensionality (PCA) if d is large.
5. Split into train / validation / test.

## 11. Complexity & Performance

- Training: O(1) (just store data).
- Prediction (naïve): O(N \* d) per query.
- Memory: O(N \* d).
  Acceleration options (future roadmap):
- KD-Tree / Ball Tree (works best for moderate dimensions < ~30).
- Locality Sensitive Hashing (approximate for high-d sparse data).
- Annoy / FAISS for large-scale approximate neighbor search.

## 12. Repository Structure

```
├── docs/                      # Theory & concept explanations
│   ├── KNN.md
│   ├── L2-distance.md
│   ├── instance-base.md
│   ├── non-parametric.md
│   ├── classification-problem.md
│   ├── regression-problem.md
│   └── supervised-learning.md
├── notebook/
│   ├── practice.ipynb         # Interactive experimentation space
│   └── utils/
│       ├── HelperFunctions.py # Plotting, L2 distance, frequency & prediction helpers
│       └── __init__.py
├── resource/
│   └── tutorial.pdf           # Supplemental reading (overview)
├── LICENSE
└── README.md
```

## 13. Installation & Environment Setup

Minimal dependencies (implied by helper code & typical usage):

- Python >= 3.10
- numpy
- pandas
- matplotlib

Create & activate environment (examples):

```
# (Windows PowerShell) create venv
python -m venv .venv
./.venv/Scripts/Activate.ps1

pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn
```

Optional (for experiments):

- seaborn (nicer visuals)
- jupyter / ipykernel

```
pip install seaborn jupyter ipykernel
python -m ipykernel install --user --name knn-tutorial
```

## 14. Quick Start Examples

### Helper: Compute L2 Distance

```python
from notebook.utils.HelperFunctions import HelperFunctions

p1 = [1, 2, 3]
p2 = [4, 6, 8]
dist = HelperFunctions.calculateL2Distance(p1, p2)
print(dist)
```

### Majority Vote (Classification) Using a Pandas DataFrame Column

```python
import pandas as pd
from notebook.utils.HelperFunctions import HelperFunctions

df = pd.DataFrame({
	'label': ['A','B','A','C','A','B']
})
pred = HelperFunctions.predictionClassification(df, 'label')
print('Predicted label:', pred)
```

### Simple Weighted Regression Aggregation

Assume a DataFrame of neighbor targets & distances:

```python
import pandas as pd
from notebook.utils.HelperFunctions import HelperFunctions

neighbors = pd.DataFrame({
	'value': [10.0, 12.0, 15.0],
	'distance': [0.5, 1.2, 2.0]
})
estimate = HelperFunctions.predictionRegression(neighbors, target='value', distance='distance', weight=True)
print('Weighted estimate:', estimate)
```

### Skeleton: Manual KNN Classification Loop (From Scratch)

```python
import math
import numpy as np
from collections import Counter

def l2(a, b):
	return math.dist(a, b)

def knn_predict(x_query, X_train, y_train, K=5):
	distances = [(l2(x_query, X_train[i]), y_train[i]) for i in range(len(X_train))]
	distances.sort(key=lambda t: t[0])
	topk = [label for (_, label) in distances[:K]]
	return Counter(topk).most_common(1)[0][0]

X_train = np.array([[1,2],[2,3],[3,3],[6,5],[7,8]])
y_train = np.array(['A','A','B','B','B'])

print(knn_predict([2.5,2.5], X_train, y_train, K=3))
```

## 15. Typical Workflow (End‑to‑End)

1. Load dataset (CSV or synthetic generation).
2. Split train/validation/test.
3. Scale numeric features (StandardScaler / MinMax).
4. Loop over candidate K values; evaluate.
5. Select best K; retrain (store training set + chosen hyperparams).
6. Evaluate on test set; record metrics.
7. Visualize decision boundaries (2D) or performance curves.
8. Package as a reusable class (future step).

## 16. Evaluation Metrics

Classification:

- Accuracy (baseline)
- Precision / Recall / F1 (esp. for imbalance)
- Confusion Matrix
- ROC-AUC (probability-style adaptation via distance weighting / inverse aggregated scores)

Regression:

- MAE, MSE / RMSE
- R²
- Median Absolute Error (robust to outliers)

## 17. Visualization & Exploration Ideas

Use `matplotlib` (and optionally `seaborn`) to:

- Scatter plot colored by class.
- Plot effect of K vs validation accuracy (elbow view).
- Distance distribution histogram for query points.
- Decision boundary grid (for 2D synthetic datasets).

The `HelperFunctions.drawScatterPlot` provides a light starting point.

## 18. Extending the Project (Roadmap)

Planned / Suggested Enhancements:

- Add a `KNNClassifier` and `KNNRegressor` class (API similar to scikit-learn: `fit`, `predict`).
- Support multiple distance metrics with strategy pattern.
- Implement KD-Tree / Ball Tree fallback based on dimension threshold.
- Add unit tests (pytest) for helper utilities & future classes.
- Add benchmarking script (timing naive vs tree-based search).
- Integrate synthetic dataset generators (circles, moons, blobs).
- Provide decision boundary visualization utility.
- Add cross-validation helper.

## 19. Common Pitfalls & Debugging Checklist

- Forgetting to scale features → distorted neighbor ranking.
- Using too large K → overly smooth / majority class dominance.
- Imbalanced classes → evaluate with F1, not only accuracy.
- High dimensional sparse vectors → consider approximate methods.
- Duplicated rows with contradictory labels → clean data.
- Performance bottleneck → pre-compute structures or batch queries.

## 20. Reference Reading & Further Study

Concept docs inside `docs/` are fully self-contained. For deeper dives:

- Pattern Recognition & Machine Learning – C. M. Bishop (Ch. on non-parametric methods)
- Elements of Statistical Learning – Hastie, Tibshirani, Friedman (KNN & kernel methods)
- scikit-learn user guide (Nearest Neighbors section)
- Research approximate nearest neighbor libraries: FAISS, Annoy, HNSW.

## 21. License

This project is distributed under the terms of the MIT License. See `LICENSE` for details.

## 🤝 Contributions

Feel free to fork & open a pull request adding:

- A formal KNN implementation
- More distance metrics
- Performance benchmarks
- Additional instructional notebooks

## ✅ Summary

This repository provides conceptual foundations, helper utilities, and a sandbox (`practice.ipynb`) to learn and extend the K-Nearest Neighbors algorithm. Build from here toward a robust, tested mini-library and experiment with optimizations as datasets grow.

## 🏁 Conclusion

K-Nearest Neighbors remains one of the most intuitive algorithms in machine learning—powerful as a baseline, instructive for learning core concepts like distance, feature scaling, and bias–variance trade-offs. This tutorial equips you with:

- A solid theoretical grounding (supervised / non‑parametric / instance‑based context).
- Practical utility functions (distance, voting, weighting) to bootstrap experiments.
- A structured workflow for evaluation, tuning, and visualization.
- A forward-looking roadmap to transform learning artifacts into a reusable mini-framework.

Suggested next steps for you:

1. Implement a `KNNClassifier` class with `fit`, `predict`, and optional `predict_proba` (distance-weighted vote proportions).
2. Add a `KNNRegressor` counterpart with optional kernel weighting (e.g., Gaussian kernel over distances).
3. Introduce hyperparameter search (grid / randomized) with cross-validation wrappers.
4. Benchmark naive vs tree-based neighbor search on synthetic datasets of increasing size.
5. Package your implementation (e.g., `pip install -e .`) and publish as a learning artifact.

If you build upon this, consider documenting performance trade-offs and adding reproducible experiment scripts. Your future self (and contributors) will thank you.

</div>

---

<div align="center">

Happy learning & exploring—may your nearest neighbors always be informative! 🚀

</div>
