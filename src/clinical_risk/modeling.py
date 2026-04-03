import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    average_precision_score,
)


def fit_logit_model(X, y):
    """
    Fit a statsmodels Logit model.

    The caller is responsible for including a constant column in X
    (e.g. via sm.add_constant). This avoids silent double-constant
    issues when the constant is added both in the notebook and here.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, must already include a constant column if desired.
    y : pd.Series
        Binary outcome (0/1).

    Returns
    -------
    statsmodels BinaryResultsWrapper
    """
    model = sm.Logit(y, X).fit(disp=False)
    return model


def logistic_or_summary(model):
    """
    Create a summary table with coefficients, odds ratios,
    confidence intervals and p-values from a statsmodels Logit result.

    Parameters
    ----------
    model : statsmodels BinaryResultsWrapper

    Returns
    -------
    pd.DataFrame
        Columns: variable, coef, OR, CI_low, CI_high, p_value.
        Sorted by OR, constant excluded.
    """
    summary = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params.values,
        "OR": np.exp(model.params.values),
        "CI_low": np.exp(model.conf_int().iloc[:, 0].values),
        "CI_high": np.exp(model.conf_int().iloc[:, 1].values),
        "p_value": model.pvalues.values,
    }).reset_index(drop=True).sort_values("OR")

    summary = summary[summary["variable"] != "const"].reset_index(drop=True)
    return summary


def logit_model_metrics(model):
    """Return scalar goodness-of-fit metrics for a fitted Logit model."""
    return {
        "n_obs": int(model.nobs),
        "log_likelihood": model.llf,
        "AIC": model.aic,
        "pseudo_R2_McFadden": model.prsquared,
    }


def predict_logit(model, X, threshold=0.5):
    """
    Generate predicted probabilities and binary class labels.

    Parameters
    ----------
    model : statsmodels BinaryResultsWrapper
    X : pd.DataFrame
        Must include the same columns (including const) used during training.
    threshold : float

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (y_prob, y_pred)
    """
    y_prob = model.predict(X)
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


def evaluate_classifier(y_true, y_prob, threshold=0.5):
    """
    Compute discrimination and calibration metrics at a given threshold.

    Returns
    -------
    dict with ROC-AUC, AUC-PR, recall, precision, f1_score, brier_score.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "AUC-PR": average_precision_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def backward_elimination_aic(X, y, verbose=True):
    """
    Backward stepwise feature selection minimising AIC.

    Parameters
    ----------
    X : pd.DataFrame
        Must include a constant column.
    y : pd.Series

    Returns
    -------
    dict with keys: selected_variables, model, aic
    """
    remaining_vars = list(X.columns)
    best_model = fit_logit_model(X[remaining_vars], y)
    best_aic = best_model.aic

    improvement = True

    while improvement and len(remaining_vars) > 1:
        improvement = False
        candidate_results = []

        for var in remaining_vars:
            if var == "const":
                continue
            trial_vars = [v for v in remaining_vars if v != var]
            try:
                trial_model = fit_logit_model(X[trial_vars], y)
                candidate_results.append((var, trial_model.aic, trial_model))
            except Exception:
                continue

        if not candidate_results:
            break

        candidate_results.sort(key=lambda x: x[1])
        var_to_remove, candidate_aic, candidate_model = candidate_results[0]

        if candidate_aic < best_aic:
            if verbose:
                print(f"Removing '{var_to_remove}' improves AIC: {best_aic:.3f} -> {candidate_aic:.3f}")
            remaining_vars.remove(var_to_remove)
            best_aic = candidate_aic
            best_model = candidate_model
            improvement = True

    return {
        "selected_variables": remaining_vars,
        "model": best_model,
        "aic": best_aic,
    }


def cross_validate_logit(X, y, n_splits=5, random_state=42):
    """
    Stratified k-fold cross-validation for a statsmodels Logit model.

    Returns mean and standard deviation across folds for each metric.

    Parameters
    ----------
    X : pd.DataFrame
        Must include a constant column.
    y : pd.Series

    Returns
    -------
    dict
        Keys: <metric>_mean and <metric>_std for each metric.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for tr, va in cv.split(X, y):
        model = fit_logit_model(X.iloc[tr], y.iloc[tr])
        y_prob, _ = predict_logit(model, X.iloc[va])
        rows.append({
            **logit_model_metrics(model),
            **evaluate_classifier(y.iloc[va], y_prob),
        })

    folds = pd.DataFrame(rows).drop(columns="n_obs", errors="ignore")
    result = {}
    for col in folds.columns:
        result[f"{col}_mean"] = folds[col].mean()
        result[f"{col}_std"] = folds[col].std()
    return result
