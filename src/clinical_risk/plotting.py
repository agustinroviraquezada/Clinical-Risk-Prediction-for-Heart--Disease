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
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })


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
    decimals_or=2,
    save_path=None,
    show=True,
):
    """
    Plot difference in outcome rate by categorical predictors and annotate
    each bar with p-value, odds ratio, and OR confidence interval.

    Required columns in results_df:
        variable, abs_diff, p_value, odds_ratio, or_ci_low, or_ci_high
    Optional columns:
        p_adj_fdr (used if use_adjusted_p=True and column exists)
    """
    _setup_plot_theme()
    plot_df = results_df.copy().sort_values("abs_diff").reset_index(drop=True)

    p_col = "p_adj_fdr" if use_adjusted_p and "p_adj_fdr" in plot_df.columns else "p_value"
    p_label_name = "p_adj" if p_col == "p_adj_fdr" else "p"

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4C72B0" if x < 0 else "#C44E52" for x in plot_df["abs_diff"]]

    ax.barh(y=plot_df["variable"], width=plot_df["abs_diff"], color=colors, alpha=0.75)
    ax.axvline(0, linestyle="--", color="black", linewidth=1.5, alpha=0.8)

    ax.set_title(
        f"Difference in {target_name.title()} Rate by Categorical Predictors",
        fontsize=14, weight="bold"
    )
    ax.set_xlabel(f"Difference in {target_name} rate  (group=1 − group=0)", fontsize=11)
    ax.set_ylabel("")

    vals = plot_df["abs_diff"].to_numpy()
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    data_range = max(vmax - vmin, 0.02)
    ax.set_xlim(vmin - data_range * 0.38, vmax + data_range * 0.38)

    for i, row in plot_df.iterrows():
        v = row["abs_diff"]
        full_txt = (
            f"Δ={v:.{decimals_diff}f}\n"
            f"{p_label_name}={format_p(row[p_col], decimals_p)}\n"
            f"OR={row['odds_ratio']:.{decimals_or}f} "
            f"[{row['or_ci_low']:.{decimals_or}f}, {row['or_ci_high']:.{decimals_or}f}]"
        )
        offset = data_range * 0.03
        ax.text(
            v + offset if v >= 0 else v - offset,
            i, full_txt,
            va="center", ha="left" if v >= 0 else "right", fontsize=10
        )

    legend_elements = [
        Patch(facecolor="#C44E52", alpha=0.75, label=f"Higher {target_name} rate in group=1"),
        Patch(facecolor="#4C72B0", alpha=0.75, label=f"Lower {target_name} rate in group=1"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.5,
               label=f"No difference in {target_name} rate"),
    ]
    ax.legend(handles=legend_elements, title=f"Target: {target_name}", frameon=True, loc="best")

    _style_axis(ax)
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


def plot_numeric_effects(
    results_df,
    group1_name="death",
    group0_name="survived",
    figsize=(12, 6),
    decimals_p=3,
    decimals_delta=2,
    save_path=None,
    show=True,
):
    """
    Forest-style bar chart of Cliff's delta for continuous predictors.
    Whiskers show bootstrap 95% CI on Cliff's delta when available.
    Bars are coloured by effect direction (blue = lower in death group,
    red = higher in death group).

    Required columns: variable, cliffs_delta, cliffs_magnitude, p_value
    Optional columns: cliffs_delta_ci_low, cliffs_delta_ci_high, p_adj_fdr
    """
    _setup_plot_theme()
    df = results_df.sort_values("cliffs_delta").reset_index(drop=True).copy()

    fig, ax = plt.subplots(figsize=figsize)

    values = df["cliffs_delta"].to_numpy()
    colors = ["#4C72B0" if v < 0 else "#C44E52" for v in values]

    # Bars
    ax.barh(df["variable"], df["cliffs_delta"], color=colors, alpha=0.70, zorder=2)

    ax.axvline(0, linestyle="--", color="black", linewidth=1.4, alpha=0.8)

    # Reference lines for magnitude thresholds (Cliff's delta: small=0.11, medium=0.28, large=0.43)
    for x, label in [(0.11, "small"), (0.28, "medium"), (0.43, "large")]:
        for sign in [1, -1]:
            ax.axvline(sign * x, linestyle=":", color="gray", linewidth=0.8, alpha=0.5)

    ax.set_title(
        f"Continuous Predictors: Effect Size on {group1_name.title()} vs {group0_name.title()}",
        fontsize=14, weight="bold"
    )
    ax.set_xlabel(
        f"Cliff's delta  ({group1_name} − {group0_name})  [−1 = always lower, +1 = always higher]",
        fontsize=11
    )
    ax.set_ylabel("")
    ax.set_xlim(-1.05, 1.05)

    p_col = "p_adj_fdr" if "p_adj_fdr" in df.columns else "p_value"
    p_label = "p_adj" if p_col == "p_adj_fdr" else "p"

    for i, row in df.iterrows():
        v = row["cliffs_delta"]
        annotation = (
            f"δ={v:.{decimals_delta}f}\n"
            f"{p_label}={format_p(row[p_col], decimals_p)}\n"
            f"({row['cliffs_magnitude']})"
        )
        offset = 0.04
        ax.text(
            v + offset if v >= 0 else v - offset,
            i,
            annotation,
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
        )

    legend_elements = [
        Patch(facecolor="#C44E52", alpha=0.70,
              label=f"Higher in {group1_name} group"),
        Patch(facecolor="#4C72B0", alpha=0.70,
              label=f"Lower in {group1_name} group"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.4,
               label="No difference (δ = 0)"),
        Line2D([0], [0], color="gray", linestyle=":", lw=0.8,
               label="Magnitude thresholds (small / medium / large)"),
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="lower right", fontsize=10)

    _style_axis(ax)
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


def plot_roc_curves(roc_curves, title="ROC Curves", figsize=(6, 5), save_path=None, show=True):
    _setup_plot_theme()
    fig, ax = plt.subplots(figsize=figsize)

    for curve in roc_curves:
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=2,
            label=f'{curve["label"]} (AUC = {curve["auc"]:.3f})'
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random classifier")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.legend(frameon=True, loc="lower right")
    _style_axis(ax)
    plt.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()



def plot_forest_or_on_ax(ax, or_summary, title="Adjusted Odds Ratios (95% CI)"):
    """
    Forest plot of adjusted odds ratios on a given Axes.

    Uses linear scale (appropriate for OR range ~[0.3, 4]).
    Labels are positioned per-row relative to their own CI_high,
    avoiding fixed-position overlap.

    Required columns: variable, OR, CI_low, CI_high, p_value
    """
    df = or_summary.copy()
    df = df[df["variable"] != "const"].copy()
    df = df.sort_values("OR").reset_index(drop=True)

    df["label_text"] = df.apply(
        lambda x: f'{x["OR"]:.2f} ({x["CI_low"]:.2f}–{x["CI_high"]:.2f})',
        axis=1
    )

    y_pos = np.arange(len(df))
    x_max = df["CI_high"].max()
    x_min = df["CI_low"].min()
    x_range = x_max - x_min

    for i, row in df.iterrows():
        color = "#C44E52" if row["p_value"] < 0.05 else "gray"
        ax.errorbar(
            x=row["OR"],
            y=i,
            xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            markersize=7,
        )
        # label next to each row's own CI_high
        ax.text(
            row["CI_high"] + x_range * 0.03,
            i,
            row["label_text"],
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.axvline(1, linestyle="--", color="gray", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["variable"], fontsize=10)
    ax.set_xlabel("Odds Ratio (linear scale)", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlim(max(0, x_min - x_range * 0.15), x_max + x_range * 0.55)

    legend_elements = [
        Line2D([0], [0], marker="o", color="#C44E52", linestyle="None",
               label="Significant (p < 0.05)", markersize=7),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               label="Not significant (p ≥ 0.05)", markersize=7),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)
    _style_axis(ax)


def plot_forest_or_comparison(or_summaries, title="Adjusted Odds Ratios — Model Comparison",
                               figsize=(12, 7), save_path=None, show=True):
    """
    Single forest plot comparing odds ratios across multiple models.

    Each variable gets one row; models are shown as offset points with
    different colours, making cross-model comparison immediate.

    Parameters
    ----------
    or_summaries : dict[str, pd.DataFrame]
        Keys are model names; values are OR summary DataFrames
        (output of logistic_or_summary). Must contain:
        variable, OR, CI_low, CI_high, p_value.
    """
    _setup_plot_theme()

    # Union of all variables, sorted by first model's OR
    first_df = next(iter(or_summaries.values()))
    all_vars = (
        first_df[first_df["variable"] != "const"]
        .sort_values("OR")["variable"]
        .tolist()
    )

    palette = sns.color_palette("tab10", len(or_summaries))
    model_names = list(or_summaries.keys())
    n_models = len(model_names)
    offsets = np.linspace(-0.25, 0.25, n_models)

    fig, ax = plt.subplots(figsize=figsize)
    y_ticks = np.arange(len(all_vars))

    for m_idx, (model_name, df) in enumerate(or_summaries.items()):
        df = df[df["variable"] != "const"].copy()
        color = palette[m_idx]
        offset = offsets[m_idx]

        for v_idx, var in enumerate(all_vars):
            row = df[df["variable"] == var]
            if row.empty:
                continue
            row = row.iloc[0]
            sig = row["p_value"] < 0.05
            ax.errorbar(
                x=row["OR"],
                y=v_idx + offset,
                xerr=[[row["OR"] - row["CI_low"]], [row["CI_high"] - row["OR"]]],
                fmt="o" if sig else "D",
                color=color,
                ecolor=color,
                elinewidth=1.4,
                capsize=3,
                markersize=6 if sig else 5,
                alpha=1.0 if sig else 0.55,
            )

    ax.axvline(1, linestyle="--", color="gray", linewidth=1.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(all_vars, fontsize=10)
    ax.set_xlabel("Odds Ratio (linear scale)", fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold")

    legend_elements = [
        Line2D([0], [0], marker="o", color=palette[i], linestyle="None",
               label=name, markersize=8)
        for i, name in enumerate(model_names)
    ] + [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               label="Circle = significant (p<0.05)", markersize=7),
        Line2D([0], [0], marker="D", color="gray", linestyle="None",
               label="Diamond = not significant", markersize=6, alpha=0.55),
    ]
    ax.legend(handles=legend_elements, frameon=True, loc="lower right", fontsize=9)
    _style_axis(ax)
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



def plot_pred_proba_hist(
    ax,
    df: pd.DataFrame,
    col: str,
    target_col: str,
    bins: int = 30,
    kde: bool = True,
    stat: str = "density",
    element: str = "step",
    title: Optional[str] = None,
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
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel(stat.title(), fontsize=11)
    ax.set_title(title or f"Predicted probabilities — {col}", fontsize=12, weight="bold")
    _style_axis(ax)