# Supervised Learning

## What is Supervised Learning?

Supervised learning is a type of machine learning where the model is trained on a labeled dataset. Each training example includes an input and the correct output (label). The goal is to learn a mapping from inputs to outputs so the model can predict labels for new, unseen data.

## Types of Supervised Learning

1. **Classification:**  
   Predicts a discrete label (e.g., spam or not spam, cat or dog).

2. **Regression:**  
   Predicts a continuous value (e.g., house price, temperature).

## How Supervised Learning Works

1. **Collect Labeled Data:**  
   Gather a dataset with input-output pairs.

2. **Split Data:**  
   Divide data into training and test (and sometimes validation) sets.

3. **Train Model:**  
   Use the training set to teach the model the relationship between inputs and outputs.

4. **Evaluate Model:**  
   Test the model on unseen data to measure its performance.

5. **Predict:**  
   Use the trained model to make predictions on new data.

## Common Algorithms

- K-Nearest Neighbors (KNN)
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- Neural Networks

## Key Concepts

- **Features:** Input variables used for prediction.
- **Labels:** The correct output for each input.
- **Loss Function:** Measures how well the model predicts the labels.
- **Overfitting:** Model learns noise instead of the pattern.
- **Underfitting:** Model is too simple to capture the pattern.

## Advantages

- Direct feedback from labeled data.
- Easier to evaluate and interpret.

## Disadvantages

- Requires a large amount of labeled data.
- May not generalize well to unseen data if overfitted.

## Applications

- Email spam detection
- Image recognition
- Medical diagnosis
- Credit scoring
