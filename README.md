
# Bank Marketing Prediction Using Machine Learning

## 1. Project Overview

This project uses five machine learning classification algorithms
to predict whether a bank customer will subscribe to a term deposit.

## 2. Dataset

Dataset: UCI Bank Marketing Dataset

Source:
https://archive.ics.uci.edu/dataset/222/bank+marketing

- Total records: 45,211
- Input features: 16
- Target column: y
- Target values: yes and no
- Missing values: 0

## 3. Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

## 4. Data Preprocessing

- Numerical features: StandardScaler
- Categorical features: OneHotEncoder
- Training and testing split: 80:20
- Random state: 42
- Stratified sampling

## 5. Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8457 | 0.9079 | 0.4182 | 0.8147 | 0.5527 | 0.5092 |
| Decision Tree | 0.8301 | 0.8644 | 0.3934 | 0.8336 | 0.5345 | 0.4937 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| SVM | 0.8505 | 0.9077 | 0.4266 | 0.8072 | 0.5582 | 0.5136 |

## 6. Best Model

SVM achieved the highest F1-score (0.5582) and MCC (0.5136).

Logistic Regression achieved the highest AUC (0.9079).

KNN achieved the highest accuracy (0.8962).

## 7. Streamlit Application

The application allows users to:

- Upload a labelled CSV dataset.
- Select one of five trained models.
- View six evaluation metrics.
- View the confusion matrix.
- View the classification report.
- Compare model performance.

The application loads saved models and does not retrain them
when a CSV file is uploaded.

## 8. Project Files

- app.py: Streamlit application
- requirements.txt: Python dependencies
- test_data.csv: Labelled test dataset
- model_metrics.csv: Model comparison results
- models/: Saved preprocessing object and five trained models

## 9. How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
