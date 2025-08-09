# K-Nearest Neighbors (KNN) Algorithm

## Introduction

K-Nearest Neighbors (KNN) is a simple, intuitive, and powerful supervised machine learning algorithm used for both classification and regression tasks. It is a non-parametric, instance-based learning method, meaning it makes predictions based on the entire training dataset rather than learning explicit parameters.

## How KNN Works

1. **Choose the number of neighbors (K):**  
   Decide how many neighbors to consider when making a prediction.

2. **Calculate Distance:**  
   For a new data point, calculate the distance between this point and all points in the training data. Common distance metrics include Euclidean, Manhattan, and Minkowski distances.

3. **Find Nearest Neighbors:**  
   Identify the K training samples closest to the new data point.

4. **Vote or Average:**
   - **Classification:** The new point is assigned the class most common among its K nearest neighbors (majority vote).
   - **Regression:** The prediction is the average (or weighted average) of the values of its K nearest neighbors.

## Key Concepts

- **Instance-based Learning:** KNN stores all available cases and classifies new cases based on similarity.
- **No Training Phase:** KNN does not explicitly learn a model; it simply stores the data.
- **Lazy Learning:** Computation is deferred until prediction time.

## Choosing K

- **Small K:** Sensitive to noise, may overfit.
- **Large K:** Smoother decision boundary, may underfit.
- **Odd K:** For binary classification, use an odd K to avoid ties.

## Pros and Cons

**Pros:**

- Simple to understand and implement.
- No assumptions about data distribution.
- Naturally handles multi-class cases.

**Cons:**

- Computationally expensive for large datasets.
- Sensitive to irrelevant features and the scale of data.
- Poor performance with high-dimensional data (curse of dimensionality).

## Applications

- Handwriting recognition
- Recommender systems
- Image classification
- Medical diagnosis

## Example

Suppose you want to classify a new fruit as an apple or orange based on its weight and color. KNN will:

- Measure the distance from the new fruit to all labeled fruits in the dataset.
- Find the K closest fruits.
- Assign the label (apple or orange) that is most common among those K fruits.
