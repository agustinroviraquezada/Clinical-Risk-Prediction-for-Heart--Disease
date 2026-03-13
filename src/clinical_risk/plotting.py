import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from typing import Optional
import math
import os 

from clinical_risk.utils import (_validate_columns_exist,
                   _validate_numeric_columns,
                   _validate_target)


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

