from scipy import stats
import statsmodels.api as sm
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests
from cliffs_delta import cliffs_delta

def rank_biserial(u_stat, n1, n2):
    """
    Rank-biserial correlation from Mann-Whitney U.
    Range: [-1, 1]
    Positive values indicate higher values in group1 relative to group0
    depending on how U is defined.
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def categorical_test(table):
    """
    For 2x2 contingency tables:
    - use Fisher if expected counts are too small
    - else chi-square
    """
    chi2, p_chi, dof, expected = stats.chi2_contingency(table, correction=False)
    
    if table.shape == (2, 2) and (expected < 5).any():
        oddsratio, p_fisher = stats.fisher_exact(table)
        return {
            "test": "Fisher exact",
            "p_value": p_fisher,
            "expected_min": expected.min()
        }
    else:
        return {
            "test": "Chi-square",
            "p_value": p_chi,
            "expected_min": expected.min()
        }


def compute_vif(X):
    """
    Compute Variance Inflation Factor (VIF) for each feature in X.

    VIF quantifies multicollinearity: VIF=1 means no correlation, VIF>5-10
    indicates high multicollinearity. A constant term is added internally
    before computing VIF.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (without a constant/intercept column).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'variable' and 'VIF', one row per feature
        (including the added constant).
    """
    X_const = sm.add_constant(X)
    vif = pd.DataFrame()
    vif["variable"] = X_const.columns
    vif["VIF"] = [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])
    ]

    return vif


def cliff_delta_bootstrap_ci(x0, x1, n_boot=1000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for Cliff's delta.

    Parameters
    ----------
    x0, x1 : array-like
        Observations for group 0 and group 1.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95 → 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    x0, x1 = np.asarray(x0), np.asarray(x1)
    deltas = []
    for _ in range(n_boot):
        s0 = rng.choice(x0, size=len(x0), replace=True)
        s1 = rng.choice(x1, size=len(x1), replace=True)
        d, _ = cliffs_delta(s1.tolist(), s0.tolist())
        deltas.append(d)
    alpha = (1 - ci) / 2
    return tuple(np.percentile(deltas, [alpha * 100, (1 - alpha) * 100]))


def compare_continuous_groups(df, cols, group_col, n_boot=1000):
    """
    Compare continuous variables between two groups using Mann-Whitney U.
    Reports medians, rank-biserial correlation, Cliff's delta with bootstrap CI,
    and Benjamini-Hochberg FDR-adjusted p-values.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Continuous columns to test.
    group_col : str
        Binary outcome column (values 0 and 1).
    n_boot : int
        Bootstrap resamples for Cliff's delta CI.

    Returns
    -------
    pd.DataFrame
        One row per variable, sorted by p_value.
    """
    group0 = df[df[group_col] == 0]
    group1 = df[df[group_col] == 1]
    results = []

    for col in cols:
        x0 = group0[col].dropna().values
        x1 = group1[col].dropna().values
        u, p = stats.mannwhitneyu(x0, x1, alternative="two-sided")
        delta, mag = cliffs_delta(x1.tolist(), x0.tolist())
        ci_low, ci_high = cliff_delta_bootstrap_ci(x0, x1, n_boot=n_boot)

        results.append({
            "variable": col,
            "group0_median": np.median(x0),
            "group1_median": np.median(x1),
            "median_diff_g1_minus_g0": np.median(x1) - np.median(x0),
            "mannwhitney_u": u,
            "p_value": p,
            "rank_biserial": rank_biserial(u, len(x0), len(x1)),
            "cliffs_delta": delta,
            "cliffs_delta_ci_low": ci_low,
            "cliffs_delta_ci_high": ci_high,
            "cliffs_magnitude": mag,
        })

    out = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    out["p_adj_fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def compare_categorical_groups(df, cols, group_col):
    """
    Compare binary/categorical variables against a binary outcome using
    Chi-square or Fisher exact test. Reports odds ratios with 95% CI
    and Benjamini-Hochberg FDR-adjusted p-values.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list[str]
        Categorical columns to test.
    group_col : str
        Binary outcome column (values 0 and 1).

    Returns
    -------
    pd.DataFrame
        One row per variable, sorted by p_value.
    """
    results = []

    for col in cols:
        table = pd.crosstab(df[col], df[group_col])
        table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        test_result = categorical_test(table)

        death_rate_if_0 = table.loc[0, 1] / table.loc[0].sum() if table.loc[0].sum() > 0 else np.nan
        death_rate_if_1 = table.loc[1, 1] / table.loc[1].sum() if table.loc[1].sum() > 0 else np.nan

        t2x2 = Table2x2(table.values)

        results.append({
            "variable": col,
            "death_rate_if_0": death_rate_if_0,
            "death_rate_if_1": death_rate_if_1,
            "abs_diff": death_rate_if_1 - death_rate_if_0,
            "test_used": test_result["test"],
            "p_value": test_result["p_value"],
            "odds_ratio": t2x2.oddsratio,
            "or_ci_low": t2x2.oddsratio_confint()[0],
            "or_ci_high": t2x2.oddsratio_confint()[1],
        })

    out = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    out["p_adj_fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out