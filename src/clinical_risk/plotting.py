import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional
import math
import os 
import numpy as np
from clinical_risk.utils import (format_p)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def _get_binary_colors(values):
    return ["#4C72B0" if v < 0 else "#C44E52" for v in values]
    
def _setup_plot_theme() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def _create_subplot_grid(
    n_plots: int,
    n_cols: int,
    figsize: Optional[tuple[int, int]] = None
):
    n_rows = math.ceil(n_plots / n_cols)

    if figsize is None:
        n_rows = math.ceil(n_plots / n_cols)
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = [axes] if n_plots == 1 else axes.flatten()

    return fig, axes


def _remove_empty_axes(fig, axes, n_used: int) -> None:
    for ax in axes[n_used:]:
        fig.delaxes(ax)


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _finalize_figure(
    fig,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    fig.suptitle(title, fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_histogram(
    ax,
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    palette: str = "Blues"
) -> None:

    series = df[col].dropna()

    sns.histplot(
        series,
        bins=bins,
        kde=True,
        ax=ax,
        color=sns.color_palette(palette, 6)[3]
    )

    ax.set_title(f"Distribution of {col}", fontsize=12, weight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")

    if not series.empty:
        mean_val = series.mean()
        skew_val = series.skew()

        ax.axvline(
            mean_val,
            linestyle="--",
            linewidth=1.5,
            color="black",
            label=f"Mean: {mean_val:.2f}\nSkewness: {skew_val:.2f}"
        )

        ax.legend(frameon=False)

    _style_axis(ax)


def plot_boxplot(
    ax,
    df: pd.DataFrame,
    col: str,
    target: Optional[str] = None,
    palette: str = "Blues"
) -> None:
    if target is None:
        sns.boxplot(
            x=df[col].dropna(),
            ax=ax,
            color=sns.color_palette(palette, 6)[3]
        )
        ax.set_title(f"Boxplot of {col}", fontsize=12, weight="bold")
        ax.set_xlabel(col)
    else:
        sns.boxplot(
            data=df,
            x=target,
            y=col,
            hue=target,
            palette=palette,
            legend=False,
            ax=ax
        )
        ax.set_title(f"{col} by {target}", fontsize=12, weight="bold")
        ax.set_xlabel(target)
        ax.set_ylabel(col)

    _style_axis(ax)


def plot_countplot(
    ax,
    df: pd.DataFrame,
    col: str,
    target: Optional[str] = None,
    palette: str = "Blues"
) -> None:

    plot_df = df[[col] + ([target] if target else [])].copy()
    plot_df[col] = plot_df[col].astype("object").fillna("Missing")
    order = plot_df[col].value_counts().index

    if target is None:
        prop_df = (
            plot_df[col]
            .value_counts(normalize=True)
            .reindex(order)
            .rename_axis(col)
            .reset_index(name="proportion")
        )

        sns.barplot(
            data=prop_df,
            x=col,
            y="proportion",
            ax=ax,
            color=sns.color_palette(palette, 6)[3]
        )

    else:
        plot_df[target] = plot_df[target].astype("object").fillna("Missing")

        prop_df = (
            plot_df.groupby([col, target], observed=False)
            .size()
            .groupby(level=0)
            .transform(lambda x: x / x.sum())
            .rename("proportion")
            .reset_index()
        )

        sns.barplot(
            data=prop_df,
            x=col,
            y="proportion",
            hue=target,
            order=order,
            palette=palette,
            ax=ax
        )
        ax.legend(title=target, frameon=False)

    ax.set_title(f"Proportion of {col}", fontsize=12, weight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    _style_axis(ax)



def plot_clinical_boxplot(
    ax,
    df: pd.DataFrame,
    col: str,
    df_limits_plot: pd.DataFrame,
    show_legend: bool = False
) -> None:

    lower_cl = df_limits_plot.loc[col, "clinical_lower"]
    upper_cl = df_limits_plot.loc[col, "clinical_upper"]
    unit = df_limits_plot.loc[col, "units"]

    data = df[col].dropna()

    sns.boxplot(
        y=data,
        ax=ax,
        width=0.32,
        color="#9ecae1",
        linewidth=1.3,
        fliersize=3
    )

    # línea inferior con etiqueta
    ax.axhline(
        lower_cl,
        color="#d62728",
        linewidth=2,
        label="Clinical limits" if show_legend else None
    )

    # línea superior sin etiqueta para evitar duplicado
    ax.axhline(
        upper_cl,
        color="#d62728",
        linewidth=2
    )

    ymin = min(data.min(), lower_cl)
    ymax = max(data.max(), upper_cl)
    margin = 0.05 * (ymax - ymin) if ymax > ymin else 1
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_title(col, fontsize=11, weight="bold", pad=8)
    ax.set_ylabel(unit, fontsize=10)
    ax.set_xlabel("")
    ax.set_xticks([])

    if show_legend:
        ax.legend(frameon=False, loc="upper right")

    _style_axis(ax)
    

def plot_grid(
    df: pd.DataFrame,
    columns: list[str],
    plot_func,
    title: str,
    n_cols: int = 3,
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    **plot_kwargs
) -> None:

    fig, axes = _create_subplot_grid(len(columns), n_cols, figsize)

    for ax, col in zip(axes, columns):
        plot_func(ax=ax, df=df, col=col, **plot_kwargs)

    _remove_empty_axes(fig, axes, len(columns))
    _finalize_figure(fig, title=title, save_path=save_path, show=show)






def plot_categorical_effects(
    results_df,
    target_name,
    use_adjusted_p=False,
    figsize=(11, 6),
    decimals_diff=2,
    decimals_p=3,
    decimals_or=2
):
    """
    Plot difference in outcome rate by categorical predictors and annotate
    each bar with p-value, odds ratio, and OR confidence interval.

    Required columns in results_df:
        - variable
        - abs_diff
        - p_value
        - odds_ratio
        - or_ci_low
        - or_ci_high

    Optional columns:
        - p_adj_fdr   (used if use_adjusted_p=True and column exists)

    Parameters
    ----------
    results_df : pd.DataFrame
        Dataframe with categorical test results.
    target_name : str
        Name of the binary target, used in title/x-label/legend.
    use_adjusted_p : bool, default False
        If True, uses p_adj_fdr when available.
    figsize : tuple, default (11, 6)
        Figure size.
    decimals_diff : int, default 2
        Decimals for abs_diff label.
    decimals_p : int, default 3
        Decimals for p-value label.
    decimals_or : int, default 2
        Decimals for OR and CI labels.
    """
    plot_df = results_df.copy().sort_values("abs_diff").reset_index(drop=True)

    p_col = "p_adj_fdr" if use_adjusted_p and "p_adj_fdr" in plot_df.columns else "p_value"
    p_label_name = "adjusted p" if p_col == "p_adj_fdr" else "p"

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#4C72B0" if x < 0 else "#C44E52" for x in plot_df["abs_diff"]]

    ax.barh(
        y=plot_df["variable"],
        width=plot_df["abs_diff"],
        color=colors,
        alpha=0.75
    )

    ax.axvline(0, linestyle="--", color="black", linewidth=1.5, alpha=0.8)

    ax.set_title(
        f"Difference in {target_name} Rate by Categorical Predictors",
        fontsize=15,
        weight="bold"
    )
    ax.set_xlabel(
        f"Difference in {target_name} rate (group=1 − group=0)",
        fontsize=11
    )
    ax.set_ylabel("")

    vals = plot_df["abs_diff"].to_numpy()
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    data_range = max(vmax - vmin, 0.02)

    label_offset = data_range * 0.03
    xpad_left = data_range * 0.32
    xpad_right = data_range * 0.32

    ax.set_xlim(vmin - xpad_left, vmax + xpad_right)

    for i, row in plot_df.iterrows():
        v = row["abs_diff"]
        p = row[p_col]
        or_val = row["odds_ratio"]
        ci_low = row["or_ci_low"]
        ci_high = row["or_ci_high"]

        diff_txt = f"Δ={v:.{decimals_diff}f}"
        p_txt = f"{p_label_name}={format_p(p,decimals_p)}"
        or_txt = (
            f"OR={or_val:.{decimals_or}f} "
            f"[{ci_low:.{decimals_or}f}, {ci_high:.{decimals_or}f}]"
        )

        full_txt = f"{diff_txt}\n{p_txt}\n{or_txt}"

        if v >= 0:
            x_text = v + label_offset
            ha = "left"
        else:
            x_text = v - label_offset
            ha = "right"

        ax.text(
            x_text,
            i,
            full_txt,
            va="center",
            ha=ha,
            fontsize=10
        )

    legend_elements = [
        Patch(
            facecolor="#C44E52",
            alpha=0.75,
            label=f"Higher {target_name} rate in group=1"
        ),
        Patch(
            facecolor="#4C72B0",
            alpha=0.75,
            label=f"Lower {target_name} rate in group=1"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            lw=1.5,
            label=f"No difference in {target_name} rate"
        )
    ]

    ax.legend(
        handles=legend_elements,
        title=f"Target: {target_name}",
        frameon=True,
        loc="best"
    )

    sns.despine(left=False, bottom=False)
    plt.tight_layout()
    plt.show()


def plot_numeric_effects(
    results_df,
    group1_name="group 1",
    group0_name="group 0",
    effect_col="median_diff_g1_minus_g0",
    effect_label="Median difference",
    figsize=(11, 6),
    decimals_effect=2,
    decimals_p=3,
    decimals_delta=2,
):
    df = results_df.sort_values(effect_col).reset_index(drop=True).copy()

    fig, ax = plt.subplots(figsize=figsize)

    values = df[effect_col].to_numpy()
    colors = ["#4C72B0" if v < 0 else "#C44E52" for v in values]

    ax.barh(df["variable"], df[effect_col], color=colors, alpha=0.75)
    ax.axvline(0, linestyle="--", color="black", linewidth=1.5, alpha=0.8)

    ax.set_title("Numeric Predictors: Inferential Results", fontsize=14, weight="bold")
    ax.set_xlabel(f"{effect_label} ({group1_name} − {group0_name})")
    ax.set_ylabel("")

    vmin, vmax = np.nanmin(values), np.nanmax(values)
    data_range = max(vmax - vmin, 0.02)
    pad = data_range * 0.35
    offset = data_range * 0.03
    ax.set_xlim(vmin - pad, vmax + pad)

    for i, row in df.iterrows():
        v = row[effect_col]
        label = (
            f"Δ={v:.{decimals_effect}f}\n"
            f"p={format_p(row['p_value'],decimals_p)}\n"
            f"δ={row['cliffs_delta']:.{decimals_delta}f} ({row['cliffs_magnitude']})"
        )

        ax.text(
            v + offset if v >= 0 else v - offset,
            i,
            label,
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_roc_curves(roc_curves, title="ROC Curves", figsize=(6, 5), save_path=None):
    plt.figure(figsize=figsize)

    for curve in roc_curves:
        plt.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=2,
            label=f'{curve["label"]} (AUC = {curve["auc"]:.3f})'
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_forest_or_on_ax(ax, or_summary, title="Adjusted Odds Ratios (95% CI)"):
    df = or_summary.copy()

    # remove intercept if present
    df = df[df["variable"] != "const"].copy()

    # sort variables
    df = df.sort_values("OR").reset_index(drop=True)

    # label text
    df["label_text"] = df.apply(
        lambda x: f'{x["OR"]:.2f} ({x["CI_low"]:.2f}–{x["CI_high"]:.2f})',
        axis=1
    )

    y_pos = np.arange(len(df))

    for i, row in df.iterrows():
        color = "black" if row["p_value"] < 0.05 else "gray"

        ax.errorbar(
            x=row["OR"],
            y=i,
            xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.4,
            capsize=3,
            markersize=6
        )

    ax.axvline(1, linestyle="--", color="gray", linewidth=1)
    ax.set_xscale("log")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["variable"])

    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_title(title)

    # add OR text
    x_text = df["CI_high"].max() * 1.15
    for i, row in df.iterrows():
        ax.text(
            x_text,
            i,
            row["label_text"],
            va="center",
            fontsize=9
        )

    ax.set_xlim(df["CI_low"].min() * 0.8, df["CI_high"].max() * 2.2)

    legend_elements = [
        Line2D([0], [0], marker="o", color="black", linestyle="None",
               label="Statistically significant (p < 0.05)", markersize=6),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               label="Not significant (p ≥ 0.05)", markersize=6)
    ]
    ax.legend(handles=legend_elements, loc="upper left")



def plot_pred_proba_hist(
    ax,
    df: pd.DataFrame,
    col: str,
    target_col: str,
    bins: int = 30,
    kde: bool = True,
    stat: str = "density",
    element: str = "step"
) -> None:
    sns.histplot(
        data=df,
        x=col,
        hue=target_col,
        bins=bins,
        element=element,
        stat=stat,
        kde=kde,
        ax=ax
    )
    ax.set_xlabel("Predicted probability")
    ax.set_title(f"{col} by true class")