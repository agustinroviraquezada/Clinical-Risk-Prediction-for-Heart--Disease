# Clinical Risk Prediction for Heart Disease
### Statistical Inference and Logistic Regression in Clinical Data

## Project Overview

This project explores mortality risk in patients with heart failure
using statistical analysis and interpretable machine learning
techniques.

The objective is twofold:

1.  Identify clinical variables associated with mortality during
    follow-up.
2.  Build an interpretable logistic regression model to estimate
    mortality risk.

The project demonstrates applied statistical thinking, reproducible
Python workflows, and interpretable predictive modeling using a real
clinical dataset.

This repository is designed as a **portfolio project in advanced
statistics and data science**.

------------------------------------------------------------------------

# Dataset

The analysis uses the **Heart Failure Clinical Records Dataset**, which
contains clinical data from **299 patients with heart failure**.

Source: UCI Machine Learning Repository / Kaggle

The dataset includes **13 clinical features**, such as:

-   age
-   anaemia
-   creatinine phosphokinase
-   diabetes
-   ejection fraction
-   high blood pressure
-   platelets
-   serum creatinine
-   serum sodium
-   smoking
-   follow-up time

Target variable:

**DEATH_EVENT**

-   1 → patient died during follow-up
-   0 → patient survived

------------------------------------------------------------------------

# Research Questions

This project aims to answer the following questions:

1.  Which clinical variables differ between patients who died and those
    who survived?
2.  Which variables are statistically associated with mortality?
3.  Which predictors remain significant in a multivariable logistic
    regression model?
4.  How accurately can mortality risk be estimated from these variables?

------------------------------------------------------------------------

# Methodology

The project follows a structured statistical workflow:

### 1 Data Understanding

Initial inspection of dataset structure, variable types, and outcome
distribution.

### 2 Exploratory Data Analysis (EDA)

-   Distribution analysis
-   Outlier detection
-   Feature relationships with the outcome

### 3 Statistical Inference

-   Group comparisons
-   Hypothesis testing
-   Association analysis

### 4 Logistic Regression Modeling

-   Binary logistic regression
-   Odds ratio interpretation
-   Variable significance

### 5 Model Evaluation

-   Confusion matrix
-   Precision / Recall
-   ROC curve
-   AUC score

### 6 Interpretation

Clinical interpretation of statistical results and model outputs.

------------------------------------------------------------------------

# Repository Structure
```
Clinical Risk Prediction for Heart Disease/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   │   └── heart_failure_clinical_records_dataset.csv
│   └── processed/
│       ├── train.csv
│       ├── test.csv
│       └── data_dictionary.md
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda_and_preprocessing.ipynb
│   ├── 03_statistical_inference.ipynb
│   ├── 04_logistic_regression_model.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_clinical_interpretation_and_conclusions.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── inference.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── plots.py
│
├── reports/
│   ├── figures/
│   └── tables/
│
└── tests/
    └── test_preprocessing.py
```

------------------------------------------------------------------------

# Technologies

Python libraries used:

-   pandas
-   numpy
-   matplotlib
-   seaborn
-   scipy
-   statsmodels
-   scikit-learn

------------------------------------------------------------------------

# Key Skills Demonstrated

This project demonstrates skills in:

-   statistical inference
-   exploratory data analysis
-   logistic regression modeling
-   model interpretation
-   reproducible data science workflows
-   clinical data analysis

------------------------------------------------------------------------

# Results (Summary)

Key findings include:

-  ---.
-   --.
-   T--.

Detailed results are presented in the analysis notebooks.

------------------------------------------------------------------------

# Limitations

-   Small dataset size (299 patients)
-   Observational data (associations, not causation)
-   Limited feature set

------------------------------------------------------------------------


