# L2 Distance (Euclidean Distance)

## Introduction

L2 distance, also known as Euclidean distance, is the most common way to measure the straight-line distance between two points in Euclidean space. It is widely used in machine learning, especially in algorithms like KNN, clustering, and dimensionality reduction.

## Formula

For two points $x = (x_1, x_2, ..., x_n)$ and $y = (y_1, y_2, ..., y_n)$ in n-dimensional space:

$$
L2(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2}
$$

Or, more generally:

$$
L2(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

## Properties

- **Non-negative:** L2 distance is always $\geq 0$.
- **Symmetric:** $L2(x, y) = L2(y, x)$
- **Triangle Inequality:** $L2(x, z) \leq L2(x, y) + L2(y, z)$
- **Zero Distance:** $L2(x, y) = 0$ if and only if $x = y$

## Usage in Machine Learning

- **KNN:** Used to find the closest neighbors.
- **Clustering (e.g., K-Means):** To assign points to the nearest cluster center.
- **Dimensionality Reduction:** To preserve distances between points.

## Example Calculation

Given two points in 2D:  
$x = (1, 2)$, $y = (4, 6)$

$$
L2(x, y) = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

## When to Use L2 Distance

- When the scale of features is similar.
- When you care about the actual geometric distance.
- Not ideal if features have very different scales (normalize or standardize first).
