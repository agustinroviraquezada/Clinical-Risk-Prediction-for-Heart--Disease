# Clinical Risk Prediction for Heart Disease

Machine learning project focused on predicting heart disease risk from clinical variables, combining exploratory data analysis, statistical testing, feature engineering, classification modeling, cross-validation, and external validation.

## Project overview

Cardiovascular disease remains one of the leading causes of mortality worldwide. Early risk stratification based on routinely collected clinical variables can support decision-making and preventive care.

This project develops and compares interpretable and non-linear machine learning models to predict the presence of heart disease using structured clinical data. In addition to predictive performance, the project emphasizes statistical rigor, model interpretability, and reproducibility.

## Objectives

- Explore and clean a clinical dataset related to heart disease
- Perform statistical analysis to identify relevant predictors
- Build reproducible machine learning pipelines
- Compare multiple classification models using cross-validation
- Evaluate performance on a held-out test set
- Assess model robustness on a secondary external-like dataset
- Interpret results from a clinical and machine learning perspective

## Dataset

Primary dataset:
- Heart Disease UCI dataset (Kaggle / UCI-derived)

Target variable:
- Binary classification: presence vs absence of heart disease

Examples of predictors:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Serum cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina

## Repository structure

```
clinical-risk-prediction-ml/
│
├── data/                 # raw, interim and processed datasets
├── notebooks/            # step-by-step exploratory and analytical notebooks
├── src/                  # reusable code for preprocessing, modeling and evaluation
├── pipelines/            # training and evaluation scripts
├── models/               # serialized trained models
├── reports/              # generated figures and result tables
└── tests/                # basic unit tests
```