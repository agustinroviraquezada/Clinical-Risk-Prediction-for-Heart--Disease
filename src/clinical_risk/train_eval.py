import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve
)

def evaluate_at_threshold(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    return {
        "threshold": threshold,
        "confusion_matrix": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }

def threshold_analysis(y_true, y_prob, model_name, thresholds=np.arange(0.1, 0.91, 0.05)):
    rows = []
    
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        
        rows.append({
            "model": model_name,
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "youden_j": recall + specificity - 1
        })
    
    return pd.DataFrame(rows)


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """
    Hosmer-Lemeshow goodness-of-fit test for logistic regression calibration.

    H0: no significant difference between observed and predicted event rates
    across quantile-based risk groups.

    p > 0.05 -> fail to reject H0 (no evidence of miscalibration).

    Note: with small test sets (n < 100) the test has low power. A non-significant
    result does not guarantee good calibration — the visual calibration curve remains
    the primary assessment.

    Parameters
    ----------
    y_true  : array-like, binary outcome (0/1)
    y_prob  : array-like, predicted probabilities
    n_groups: int, number of quantile bins (default 10 = deciles)

    Returns
    -------
    dict with keys: hl_stat, p_value, df, groups (pd.DataFrame)
    """
    df = pd.DataFrame({'y_true': np.asarray(y_true), 'y_prob': np.asarray(y_prob)})
    df['group'] = pd.qcut(df['y_prob'], q=n_groups, duplicates='drop', labels=False)

    grouped = (
        df.groupby('group', observed=True)
        .agg(n=('y_true', 'count'), observed=('y_true', 'sum'), expected=('y_prob', 'sum'))
        .reset_index()
    )

    denom = grouped['expected'] * (1 - grouped['expected'] / grouped['n'])
    safe_denom = denom.replace(0, np.nan)
    hl_stat = float((((grouped['observed'] - grouped['expected']) ** 2) / safe_denom).sum())

    df_chi = len(grouped) - 2
    p_value = float(1 - chi2_dist.cdf(hl_stat, df=df_chi))

    return {'hl_stat': hl_stat, 'p_value': p_value, 'df': df_chi, 'groups': grouped}


def compute_dca(y_true, y_prob, thresholds=None):
    """
    Decision Curve Analysis (DCA).

    Computes the net benefit of the model, 'treat all', and 'treat none'
    strategies across a range of decision thresholds.

    Net benefit at threshold t:
        NB = TP/n - FP/n * t/(1-t)

    The weight t/(1-t) reflects the harm of a false positive relative to a
    false negative at that threshold. A clinician operating at t=0.3 considers
    a false positive ~2.3x less harmful than a missed event.

    Parameters
    ----------
    y_true     : array-like, binary outcome (0/1)
    y_prob     : array-like, predicted probabilities
    thresholds : array-like, optional. Defaults to linspace(0.01, 0.99, 100).

    Returns
    -------
    pd.DataFrame with columns: threshold, nb_model, nb_all, nb_none
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        nb_model = (tp / n) - (fp / n) * (t / (1 - t))
        nb_all   = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({'threshold': t, 'nb_model': nb_model, 'nb_all': nb_all, 'nb_none': 0.0})

    return pd.DataFrame(rows)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """
    Percentile bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    y_true    : array-like
    y_prob    : array-like
    metric_fn : callable — takes (y_true, y_prob) and returns a scalar
    n_boot    : int, number of bootstrap resamples
    ci        : float, confidence level (default 0.95)
    seed      : int

    Returns
    -------
    tuple (lower, upper)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass

    alpha = (1 - ci) / 2
    return tuple(np.percentile(scores, [alpha * 100, (1 - alpha) * 100]))



def cv_optimal_threshold(
    X, y, pipeline, features, n_splits=5, seed=42, round_digits=4
):
    """
    Compute an optimal classification threshold using cross-validation
    based on the Youden index (TPR - FPR).

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series or array-like
        Target variable.

    pipeline : estimator object
        A scikit-learn compatible pipeline with a `fit` and
        `predict_proba` method.

    features : list of str
        List of feature names to be used for training.

    n_splits : int, default=5
        Number of cross-validation folds.

    seed : int, default=42
        Random seed for reproducibility.

    round_digits : int, default=4
        Number of decimal places for per-fold thresholds.

    Returns
    -------
    dict
        Dictionary containing:
        - 'fold_thresholds' : list of float
            Optimal threshold for each fold.
        - 'cv_threshold' : float
            Mean threshold across folds.

    Notes
    -----
    The Youden index is defined as:
        Youden = TPR - FPR

    The selected threshold maximizes the separation between
    true positive rate and false positive rate.

    This approach helps avoid overfitting when selecting
    classification thresholds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_thresholds = []

    for tr, va in cv.split(X, y):
        pipe = clone(pipeline)

        pipe.fit(X.iloc[tr][features], y.iloc[tr])
        y_prob = pipe.predict_proba(X.iloc[va][features])[:, 1]

        fpr, tpr, thresholds = roc_curve(y.iloc[va], y_prob)
        youden = tpr - fpr
        best_thr = float(thresholds[np.argmax(youden)])

        fold_thresholds.append(round(best_thr, round_digits))
    cv_threshold = float(np.mean(fold_thresholds))
    return fold_thresholds, cv_threshold
