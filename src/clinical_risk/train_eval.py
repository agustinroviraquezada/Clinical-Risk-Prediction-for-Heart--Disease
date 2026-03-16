import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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
