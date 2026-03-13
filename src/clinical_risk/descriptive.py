import pandas as pd
import numpy as np 

def iqr_outlier_summary(dataframe, columns):
    rows = []
    
    for col in columns:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        n_outliers = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)].shape[0]
        
        rows.append({
            "variable": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_outliers": n_outliers,
            "outlier_pct": 100 * n_outliers / len(dataframe)
        })
    
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)



def compare_skew(df, columns):
    """
    Compare original skewness vs log1p-transformed skewness
    for selected columns in a DataFrame.
    """
    results = []

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")
        
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' contains negative values; log1p may not be appropriate.")

        original_skew = df[col].skew()
        log_skew = np.log1p(df[col]).skew()

        results.append({
            "variable": col,
            "original_skew": original_skew,
            "log_skew": log_skew,
            "abs_original_skew": abs(original_skew),
            "abs_log_skew": abs(log_skew),
            "improvement": abs(original_skew) - abs(log_skew)
        })

    return pd.DataFrame(results).sort_values("improvement", ascending=False)