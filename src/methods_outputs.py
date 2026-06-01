"""Methods-section tables and figures for the STAT 250 report.

These outputs document the dataset structure, variable definitions, missingness,
country coverage, macroeconomic period coding, and analysis plan. They are meant
for the Methods/EDA part of the written report, not for hypothesis-test results.
"""

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .features import PERIOD_ORDER
from .utils import save_table

MAIN_VARIABLES = ["unemp", "cpi", "ipi", "cci", "gdp_growth"]
DERIVED_VARIABLES = ["cpi_qoq", "cpi_yoy", "ipi_yoy", "high_inflation", "trend"]


def _quarter_to_timestamp(series: pd.Series) -> pd.Series:
    return series.apply(lambda q: q.to_timestamp(how="end"))


def build_methods_tables(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return report-ready tables for the Methods section."""
    panel = panel.copy()

    overview = pd.DataFrame(
        [
            {"item": "Observation unit", "value": "Country-quarter"},
            {"item": "Countries included", "value": str(panel["country"].nunique())},
            {"item": "Total observations", "value": str(len(panel))},
            {"item": "Date range", "value": f"{panel['quarter'].min()} to {panel['quarter'].max()}"},
            {"item": "Main continuous variables", "value": ", ".join(MAIN_VARIABLES)},
            {"item": "Main categorical/discrete variables", "value": "country, group, year, quarter_num, period, high_inflation"},
            {"item": "High-inflation rule", "value": "1 if quarter-over-quarter CPI growth is greater than 3%; 0 otherwise"},
            {"item": "Turkiye observations", "value": str((panel["country"] == "TUR").sum())},
            {"item": "Other OECD comparison observations", "value": str((panel["country"] != "TUR").sum())},
        ]
    )

    variable_dictionary = pd.DataFrame(
        [
            {"variable": "country", "type": "Categorical", "description": "Three-letter country code", "role_in_report": "Grouping and comparison"},
            {"variable": "country_name", "type": "Categorical", "description": "Readable country name", "role_in_report": "Tables and labels"},
            {"variable": "quarter", "type": "Time/discrete", "description": "Quarterly period from 2005Q1 to 2025Q4", "role_in_report": "Time index"},
            {"variable": "unemp", "type": "Continuous", "description": "Unemployment rate", "role_in_report": "Main response in mean tests, ANOVA, and regression"},
            {"variable": "cpi", "type": "Continuous", "description": "Consumer price index", "role_in_report": "Inflation-related analysis"},
            {"variable": "ipi", "type": "Continuous", "description": "Industrial production index", "role_in_report": "Production comparison"},
            {"variable": "cci", "type": "Continuous", "description": "Consumer confidence index", "role_in_report": "Regression predictor"},
            {"variable": "gdp_growth", "type": "Continuous", "description": "Quarterly GDP growth rate", "role_in_report": "Regression predictor"},
            {"variable": "cpi_qoq", "type": "Continuous/derived", "description": "Quarter-over-quarter CPI percentage change", "role_in_report": "Creates high-inflation indicator"},
            {"variable": "cpi_yoy", "type": "Continuous/derived", "description": "Year-over-year CPI percentage change", "role_in_report": "Regression predictor"},
            {"variable": "ipi_yoy", "type": "Continuous/derived", "description": "Year-over-year IPI percentage change", "role_in_report": "Regression predictor"},
            {"variable": "high_inflation", "type": "Binary/derived", "description": "Equals 1 when cpi_qoq > 3%, otherwise 0", "role_in_report": "Proportion tests"},
            {"variable": "period", "type": "Categorical/derived", "description": "Normal, GFC 2008-09, COVID-19 2020, or Recovery 2021-22", "role_in_report": "ANOVA grouping variable"},
            {"variable": "trend", "type": "Discrete/derived", "description": "Sequential quarter number within country", "role_in_report": "Regression control"},
            {"variable": "group", "type": "Categorical/derived", "description": "Turkiye or Other OECD", "role_in_report": "Two-sample tests and plots"},
        ]
    )

    missingness = (
        panel[MAIN_VARIABLES + DERIVED_VARIABLES]
        .isna()
        .sum()
        .rename("missing")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    missingness["non_missing"] = len(panel) - missingness["missing"]
    missingness["missing_pct"] = (missingness["missing"] / len(panel) * 100).round(2)
    missingness = missingness[["variable", "non_missing", "missing", "missing_pct"]]

    country_coverage = (
        panel.groupby(["country", "country_name"], observed=True)
        .agg(
            first_quarter=("quarter", "min"),
            last_quarter=("quarter", "max"),
            observations=("quarter", "size"),
            missing_unemp=("unemp", lambda s: int(s.isna().sum())),
            missing_cpi=("cpi", lambda s: int(s.isna().sum())),
            missing_ipi=("ipi", lambda s: int(s.isna().sum())),
            missing_cci=("cci", lambda s: int(s.isna().sum())),
            missing_gdp_growth=("gdp_growth", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    period_coding = (
        panel.groupby("period", observed=True)
        .agg(
            observations=("quarter", "size"),
            first_quarter=("quarter", "min"),
            last_quarter=("quarter", "max"),
            countries=("country", "nunique"),
        )
        .reset_index()
    )
    period_coding["period"] = pd.Categorical(period_coding["period"], categories=PERIOD_ORDER, ordered=True)
    period_coding = period_coding.sort_values("period").reset_index(drop=True)

    analysis_plan = pd.DataFrame(
        [
            {
                "rq": "RQ1",
                "question_short": "Is Turkiye's mean unemployment different from 6.5%?",
                "response": "unemp",
                "method": "One-sample t-test",
                "main_assumptions_checked": "continuous response; large-sample robustness; independence discussed",
                "fallback_or_note": "Interpret as benchmark comparison, not causal evidence",
            },
            {
                "rq": "RQ2",
                "question_short": "Did Turkiye's COVID-period mean IPI differ from other OECD countries?",
                "response": "ipi",
                "method": "Two-sample t-test / Welch t-test",
                "main_assumptions_checked": "normality; equality of variances with Levene test; independent groups",
                "fallback_or_note": "Use Welch version if variances are unequal",
            },
            {
                "rq": "RQ3",
                "question_short": "Was Turkiye's high-inflation quarter ratio greater than 50%?",
                "response": "high_inflation",
                "method": "One-sample proportion z-test",
                "main_assumptions_checked": "binary outcome; adequate success/failure counts",
                "fallback_or_note": "High inflation is defined as cpi_qoq > 3%",
            },
            {
                "rq": "RQ4",
                "question_short": "Did Turkiye's high-inflation ratio differ from other OECD countries?",
                "response": "high_inflation",
                "method": "Two-sample proportion z-test",
                "main_assumptions_checked": "binary outcome; adequate counts in both groups",
                "fallback_or_note": "Repeated country-quarter observations make independence approximate",
            },
            {
                "rq": "RQ5",
                "question_short": "Do unemployment rates differ across macroeconomic periods?",
                "response": "unemp",
                "method": "One-way ANOVA / Welch ANOVA with post-hoc comparisons",
                "main_assumptions_checked": "normality; variance equality with Levene test",
                "fallback_or_note": "Use Games-Howell after Welch ANOVA when variances are unequal",
            },
            {
                "rq": "RQ6",
                "question_short": "Which variables are associated with Turkiye's unemployment?",
                "response": "unemp",
                "method": "Simple and multiple OLS regression",
                "main_assumptions_checked": "linearity; residual normality; heteroskedasticity; autocorrelation",
                "fallback_or_note": "Use HAC/Newey-West standard errors; interpret as association only",
            },
        ]
    )

    return {
        "methods_dataset_overview": overview,
        "methods_variable_dictionary": variable_dictionary,
        "methods_missingness_summary": missingness,
        "methods_country_coverage": country_coverage,
        "methods_period_coding": period_coding,
        "methods_analysis_plan": analysis_plan,
    }


def plot_observation_coverage(panel: pd.DataFrame, path: Path) -> None:
    """Save a bar chart of available observations by country."""
    coverage = (
        panel.groupby(["country", "country_name"], observed=True)
        .size()
        .reset_index(name="observations")
        .sort_values("observations", ascending=False)
    )
    labels = coverage["country_name"].tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, coverage["observations"])
    ax.set_title("Quarterly Observation Coverage by Country")
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of country-quarter observations")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_missingness(panel: pd.DataFrame, path: Path) -> None:
    """Save a compact missingness bar chart for analysis variables."""
    miss = panel[MAIN_VARIABLES + DERIVED_VARIABLES].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(miss.index, miss.values)
    ax.set_title("Missing Values in Analysis Variables")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Missing observations")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_period_timeline(panel: pd.DataFrame, path: Path) -> None:
    """Save a timeline showing the macroeconomic period coding used in ANOVA."""
    quarters = pd.Series(sorted(panel["quarter"].unique()))
    timeline = pd.DataFrame({"quarter": quarters})
    # Each quarter has the same period rule, so take the first match from panel.
    period_map = panel.drop_duplicates("quarter").set_index("quarter")["period"]
    timeline["period"] = timeline["quarter"].map(period_map).astype(str)
    timeline["date"] = _quarter_to_timestamp(timeline["quarter"])
    period_to_y = {period: i for i, period in enumerate(PERIOD_ORDER)}
    timeline["y"] = timeline["period"].map(period_to_y)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.scatter(timeline["date"], timeline["y"], s=28)
    ax.set_yticks(list(period_to_y.values()))
    ax.set_yticklabels(list(period_to_y.keys()))
    ax.set_xlabel("Quarter")
    ax.set_title("Macroeconomic Period Classification Used in the Analysis")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_analysis_flowchart(path: Path) -> None:
    """Save a simple flowchart of the data-to-analysis workflow."""
    steps = [
        "Raw OECD\nindicator files",
        "Quarterly\naggregation",
        "Country-quarter\npanel merge",
        "Derived variables\n(CPI/ IPI growth, periods)",
        "EDA + assumption\nchecks",
        "Six required\nstatistical analyses",
    ]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    xs = [0.08, 0.24, 0.40, 0.58, 0.76, 0.92]
    y = 0.55
    for i, (x, step) in enumerate(zip(xs, steps)):
        ax.text(
            x,
            y,
            step,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black"),
            transform=ax.transAxes,
        )
        if i < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.07, y),
                xytext=(x + 0.07, y),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.2),
            )
    ax.set_title("Methods Workflow", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_methods_outputs(panel: pd.DataFrame, tables_dir: Path, figures_dir: Path) -> dict[str, list[Path]]:
    """Save all Methods-section tables and figures; return written paths."""
    written_tables: list[Path] = []
    for name, table in build_methods_tables(panel).items():
        out = tables_dir / f"{name}.csv"
        save_table(table, out)
        written_tables.append(out)

    figure_paths = [
        figures_dir / "methods_observation_coverage_by_country.png",
        figures_dir / "methods_missingness_by_variable.png",
        figures_dir / "methods_period_timeline.png",
        figures_dir / "methods_analysis_flowchart.png",
    ]
    plot_observation_coverage(panel, figure_paths[0])
    plot_missingness(panel, figure_paths[1])
    plot_period_timeline(panel, figure_paths[2])
    plot_analysis_flowchart(figure_paths[3])

    # A short plain-text guide saved with outputs so group members know what to use.
    guide = textwrap.dedent(
        """
        Methods-section output guide
        ============================

        Recommended tables for the report:
        1. methods_dataset_overview.csv: compact table for data source, unit, period, sample size, and variable classes.
        2. methods_variable_dictionary.csv: use a shortened version if page space allows; otherwise move to appendix or omit.
        3. methods_analysis_plan.csv: best Methods table because it connects each research question to the required STAT250 method and assumption checks.
        4. methods_period_coding.csv: useful if the ANOVA period labels need clarification.
        5. methods_missingness_summary.csv and methods_country_coverage.csv: keep as support tables; include only if missingness/coverage is discussed.

        Recommended figures for the report:
        1. methods_analysis_flowchart.png: optional, useful at the beginning of Methods if there is space.
        2. methods_period_timeline.png: useful before the ANOVA description.
        3. methods_observation_coverage_by_country.png: include only if you want to justify the sample structure.
        4. methods_missingness_by_variable.png: usually omit from the main report unless missingness is important.

        The report should not include Python code. Submit code separately as required by STAT250.
        """
    ).strip()
    guide_path = tables_dir / "methods_output_guide.txt"
    guide_path.write_text(guide, encoding="utf-8")

    return {"tables": written_tables + [guide_path], "figures": figure_paths}
