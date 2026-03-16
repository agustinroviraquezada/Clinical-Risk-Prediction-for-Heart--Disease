import pandas as pd
from typing import Optional
import numpy as np

def _validate_columns_exist(df: pd.DataFrame, columns: list[str]) -> None:
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")


def _validate_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(f"These columns are not numeric: {non_numeric}")


def _validate_target(df: pd.DataFrame, target: Optional[str]) -> None:
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")


def format_p(p,decimals_p):
    if np.isnan(p):
        return "NA"
    if p < 10**(-decimals_p):
        return f"<{10**(-decimals_p):.{decimals_p}f}"
    return f"{p:.{decimals_p}f}"



def add_log_features(df,cols,remove = True):
    df = df.copy()
    df[[f"log_{c}" for c in cols]] = np.log1p(df[cols])
    if remove:
        return df.drop(columns=cols)
    return df
