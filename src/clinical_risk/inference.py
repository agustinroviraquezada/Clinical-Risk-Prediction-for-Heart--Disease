from scipy import stats
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

    X_const = sm.add_constant(X)
    vif = pd.DataFrame()
    vif["variable"] = X_const.columns
    vif["VIF"] = [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])
    ]

    return vif