import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
classification_report,
confusion_matrix
)
from sklearn.model_selection import cross_validate

def fit_logit_model(X, y):
    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=False)
    return model


def logistic_or_summary(model):
    """
    Create a summary table with coefficients, odds ratios,
    confidence intervals and p-values from a statsmodels Logit result.
    
    Parameters
    ----------
    model : statsmodels.discrete.discrete_model.BinaryResults
        Fitted model result (e.g. result = sm.Logit(...).fit())
    
    Returns
    -------
    pandas.DataFrame
        Table with coef, OR, CI and p-values sorted by significance.
    """
    
    summary = pd.DataFrame({
        "variable": model.params.index,
        "coef": model.params,
        "OR": np.exp(model.params),
        "CI_low": np.exp(model.conf_int()[0]),
        "CI_high": np.exp(model.conf_int()[1]),
        "p_value": model.pvalues
    }).reset_index(drop=True).sort_values("OR")
    
    summary = summary[summary["variable"] != "const"]

    return summary



def logit_model_metrics(model, model_name="model"):
    
    return {
        "n_obs": int(model.nobs),
        "log_likelihood": model.llf,
        "AIC": model.aic,
        "pseudo_R2_McFadden": model.prsquared
    }



def predict_logit(model, X_test, threshold=0.5):
    """
    Generate predicted probabilities and class predictions
    from a fitted statsmodels Logit model.
    """
    
    X_test_const = sm.add_constant(X_test)
    y_prob = model.predict(X_test_const)
    y_pred = (y_prob >= threshold).astype(int)
    
    return y_prob, y_pred


def evaluate_classifier(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }





def backward_elimination_aic(X, y, verbose=True):
    remaining_vars = list(X.columns)
    best_model = fit_logit_model(X[remaining_vars], y)
    best_aic = best_model.aic
    
    improvement = True
    
    while improvement and len(remaining_vars) > 1:
        improvement = False
        candidate_results = []
        
        for var in remaining_vars:
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
        "aic": best_aic
    }



def cross_validate_logit(X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model = fit_logit_model(X.iloc[tr], y.iloc[tr])
        y_prob, _ = predict_logit(model, X.iloc[va])

        rows.append({
            **logit_model_metrics(model),
            **evaluate_classifier(y.iloc[va], y_prob)
        })

    folds = pd.DataFrame(rows).drop(columns="n_obs", errors="ignore")
    return folds.mean().to_dict()