Liver Disorder Classification Using Machine Learning

This project develops and evaluates multiple machine learning models to classify patients into critical liver disorder (Class 2) and non-critical liver disorder (Class 1) categories based on clinical measurements.

The workflow follows a complete end-to-end machine learning pipeline:

Data preprocessing: Cleaned the dataset, handled missing values, and removed outliers to improve data quality.

Class imbalance handling: Applied Synthetic Minority Oversampling Technique (SMOTE) to balance the class distribution and reduce model bias.

Feature engineering: Used Principal Component Analysis (PCA) to reduce dimensionality and identify the most informative features while minimizing redundant information.

Model development: Split the dataset into training and testing sets and trained eight supervised learning algorithms:
Support Vector Machine (SVM)
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Naive Bayes
Neural Network
Bagging Ensemble
Boosting Ensemble

-Model evaluation: Compared model performance using confusion matrices, ROC curves, and key classification metrics, including accuracy, precision, recall, and F1-score, to determine the most effective approach for liver disorder prediction.
