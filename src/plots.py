"""Plotting functions"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg") # Use non-interactive backend to skip plt.show() and allow saving figures without rendering.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import PERIOD_ORDER
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _quarter_to_timestamp(series: pd.Series) -> pd.Series:
    return series.apply(lambda q: q.to_timestamp(how="end"))


def plot_turkiye_vs_oecd(panel: pd.DataFrame, variable: str, ylabel: str, title: str, path: Path) -> None:
    """Save a Turkiye vs other OECD average time-series plot."""
    plot_data = panel.copy()
    plot_data["date"] = _quarter_to_timestamp(plot_data["quarter"])
    turkiye = plot_data[plot_data["country"] == "TUR"].sort_values("date")
    oecd = (
        plot_data[plot_data["country"] != "TUR"]
        .groupby("date", as_index=False)[variable]
        .mean()
        .sort_values("date")
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(oecd["date"], oecd[variable], color="#457B9D", lw=2.0, label="Other OECD average")
    ax.plot(turkiye["date"], turkiye[variable], color="#C1121F", lw=2.4, label="Turkiye")
    ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"), color="#F4A261", alpha=0.16, label="GFC 2008-09")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), color="#6A4C93", alpha=0.12, label="COVID-19 2020")
    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(panel: pd.DataFrame, path: Path) -> None:
    """Save a Turkiye correlation heatmap including the main project variables."""
    variables = ["unemp", "cpi_yoy", "ipi_yoy", "cci", "gdp_growth"]
    corr = panel.loc[panel["country"] == "TUR", variables].dropna().corr()
    labels = ["Unemployment", "CPI YoY", "IPI YoY", "CCI", "GDP growth"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Turkiye Correlation Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_unemployment_by_period(panel: pd.DataFrame, path: Path) -> None:
    """Save a boxplot of unemployment by macroeconomic period."""
    data = panel[["period", "unemp"]].dropna()
    plot_data = [data.loc[data["period"].astype(str) == period, "unemp"].values for period in PERIOD_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    box = ax.boxplot(plot_data, labels=PERIOD_ORDER, patch_artist=True, medianprops={"color": "black", "linewidth": 1.6})
    colors = ["#6C757D", "#F4A261", "#6A4C93", "#457B9D"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Unemployment rate (%)")
    ax.set_title("Unemployment by Macroeconomic Period")
    plt.xticks(rotation=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fitted_vs_actual(models: dict, path: Path) -> None:
    """Save fitted vs actual unemployment for Turkiye's full regression model."""
    model = models["full"]
    data = models["full_data"].copy()
    data["date"] = _quarter_to_timestamp(data["quarter"])
    data["fitted"] = model.fittedvalues

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data["date"], data["unemp"], color="#C1121F", lw=2.2, label="Actual")
    ax.plot(data["date"], data["fitted"], color="#2D6A4F", lw=2.0, linestyle="--", label="Fitted")
    ax.set_title("Turkiye Unemployment: Actual vs Fitted Full Model")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Unemployment rate (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_core_figures(panel: pd.DataFrame, models: dict, figures_dir: Path) -> None:
    """Save all core figures required for the report."""
    plot_turkiye_vs_oecd(
        panel,
        "unemp",
        "Unemployment rate (%)",
        "Unemployment: Turkiye vs Other OECD Average",
        figures_dir / "turkiye_vs_oecd_unemployment.png",
    )
    plot_turkiye_vs_oecd(
        panel,
        "cpi",
        "Consumer Price Index",
        "CPI: Turkiye vs Other OECD Average",
        figures_dir / "turkiye_vs_oecd_cpi.png",
    )
    plot_correlation_heatmap(panel, figures_dir / "turkiye_correlation_heatmap.png")
    plot_unemployment_by_period(panel, figures_dir / "unemployment_by_period_boxplot.png")
    plot_fitted_vs_actual(models, figures_dir / "turkiye_regression_fitted_vs_actual.png")
