# Liver Disorder Classification Using Machine Learning

## Overview

This project develops and evaluates multiple machine learning models to classify patients into **critical liver disorder (Class 2)** and **non-critical liver disorder (Class 1)** categories based on clinical measurements.

The project follows a complete end-to-end machine learning pipeline, including data preprocessing, class imbalance handling, feature extraction, model training, and performance evaluation.

## Workflow

### Data Preprocessing

The dataset was cleaned by handling missing values and removing outliers to improve data quality and ensure more reliable model performance.

### Class Imbalance Handling

The dataset contained an imbalance between the two classes. To address this issue, the **Synthetic Minority Oversampling Technique (SMOTE)** was applied to balance the class distribution and reduce model bias toward the majority class.

### Feature Engineering

**Principal Component Analysis (PCA)** was applied to reduce dimensionality and extract the most informative features while minimizing redundant information within the dataset.

### Model Development

The dataset was divided into training and testing sets. Eight supervised learning algorithms were trained and evaluated:

- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Neural Network
- Bagging Ensemble
- Boosting Ensemble

### Model Evaluation

The models were compared using multiple evaluation techniques, including:

- Confusion matrices
- ROC curves
- Accuracy
- Precision
- Recall
- F1-score

The goal of the project was to identify the machine learning model that provides the most effective performance for liver disorder classification.

## Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- SMOTE
- PCA
