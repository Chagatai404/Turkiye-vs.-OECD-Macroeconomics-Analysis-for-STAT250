"""Complete OECD macroeconomic indicators analysis for the project"""

import pandas as pd

from .clean_data import build_quarterly_panel
from .config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TABLES_DIR
from .eda import descriptive_tables
from .features import add_analysis_features, make_regression_data
from .hypothesis_tests import run_all_hypothesis_tests
from .load_data import load_raw_datasets
from .plots import save_core_figures
from .regression import fit_turkiye_regressions
from .utils import ensure_directories, ensure_quarter_period, save_table


PANEL_PATH = PROCESSED_DATA_DIR / "clean_oecd_quarterly_panel.csv"
REGRESSION_PATH = PROCESSED_DATA_DIR / "clean_oecd_regression_data.csv"


def _load_or_build_panel() -> pd.DataFrame:
    """
    Load processed data, or build it if the processed CSV is absent.
    """
    if PANEL_PATH.exists():
        panel = pd.read_csv(PANEL_PATH)
        panel["quarter"] = ensure_quarter_period(panel["quarter"])
        return add_analysis_features(panel)

    print("Processed panel was not found. Building it from raw OECD files...")
    raw = load_raw_datasets(RAW_DATA_DIR)
    panel = add_analysis_features(build_quarterly_panel(raw))
    ensure_directories(PROCESSED_DATA_DIR)
    panel.to_csv(PANEL_PATH, index=False)
    make_regression_data(panel).to_csv(REGRESSION_PATH, index=False)
    return panel


def _print_research_questions() -> None:
    print("Research Questions")
    print("=" * 72)
    print("RQ1: Is Turkiye's average unemployment rate over 2005-2025 significantly different from 6.5%?")
    print("RQ2: During COVID-19, did Turkiye's average IPI differ from the other OECD comparison countries?")
    print("RQ3: Did Turkiye experience high inflation in more than half of observed quarters?")
    print("RQ4: Is Turkiye's high-inflation quarter ratio different from the other OECD comparison countries?")
    print("RQ5: Do unemployment rates differ across macroeconomic periods in the selected OECD panel?")
    print("RQ6: Which macroeconomic variables are associated with Turkiye's unemployment rate?")
    print()


def _print_hypothesis_summary(summary: pd.DataFrame) -> None:
    display_cols = ["rq", "test", "statistic", "p_value", "decision"]
    print("Hypothesis Test Summary")
    print("=" * 72)
    print(summary.reindex(columns=display_cols).round(4).to_string(index=False))
    print()
    print("Interpretation note: these tests use observational macroeconomic data, so the")
    print("results are described as associations or evidence consistent with differences,")
    print("not as causal proof.")
    print()


def _save_regression_outputs(models: dict, regression_summary: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    save_table(regression_summary, TABLES_DIR / "rq6_regression_diagnostics.csv")
    save_table(coefficients, TABLES_DIR / "rq6_regression_coefficients.csv")
    (TABLES_DIR / "rq6_simple_regression_summary.txt").write_text(models["simple"].summary().as_text(), encoding="utf-8")
    (TABLES_DIR / "rq6_full_regression_summary.txt").write_text(models["full"].summary().as_text(), encoding="utf-8")


def main() -> None:
    ensure_directories(PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR)
    _print_research_questions()

    panel = _load_or_build_panel()
    regression_data = make_regression_data(panel)
    regression_data.to_csv(REGRESSION_PATH, index=False)

    print("Quarterly Panel Overview")
    print("=" * 72)
    print(f"Panel shape: {panel.shape[0]} rows x {panel.shape[1]} columns")
    print(f"Date range: {panel['quarter'].min()} to {panel['quarter'].max()}")
    print(f"Countries: {', '.join(sorted(panel['country'].unique()))}")
    print()

    for name, table in descriptive_tables(panel).items():
        save_table(table, TABLES_DIR / f"{name}.csv")

    hypothesis_summary, supporting_tables = run_all_hypothesis_tests(panel)
    save_table(hypothesis_summary, TABLES_DIR / "hypothesis_tests_summary.csv")
    for name, table in supporting_tables.items():
        save_table(table, TABLES_DIR / f"{name}.csv")
    _print_hypothesis_summary(hypothesis_summary)

    print("RQ6 Regression Models")
    print("=" * 72)
    print("Simple model: unemp ~ ipi_yoy")
    print("Full model: unemp ~ cpi_yoy + ipi_yoy + cci + gdp_growth + trend")
    print("Standard errors use HAC/Newey-West with maxlags=4 when supported.")
    print()
    models, regression_summary, coefficients = fit_turkiye_regressions(regression_data)
    _save_regression_outputs(models, regression_summary, coefficients)
    columns = ["model", "n", "r_squared", "adj_r_squared", "f_p_value", "covariance", "durbin_watson"]
    print(regression_summary[columns].round(4).to_string(index=False))
    print()
    print("Full-model coefficients:")
    print(coefficients[coefficients["model"] == "full"].round(4).to_string(index=False))
    print()

    save_core_figures(panel, models, FIGURES_DIR)

    print("Saved Outputs")
    print("=" * 72)
    print(f"Processed panel: {PANEL_PATH}")
    print(f"Regression data: {REGRESSION_PATH}")
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("Analysis complete.")
