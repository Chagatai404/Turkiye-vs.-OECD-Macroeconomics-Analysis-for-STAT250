"""Analysis features for EDA, hypothesis tests, ANOVA, and regression."""

import numpy as np
import pandas as pd

from .utils import ensure_quarter_period


PERIOD_ORDER = ["Normal", "GFC 2008-09", "COVID-19 2020", "Recovery 2021-22"] # ordered macroeconomic periods for analysis


def assign_period(quarter: pd.Period) -> str:
    """
    Assign a macroeconomic period label to a quarterly observation.
    """
    year = quarter.year
    if year in (2008, 2009):
        return "GFC 2008-09"
    if year == 2020:
        return "COVID-19 2020"
    if year in (2021, 2022):
        return "Recovery 2021-22"
    return "Normal"


def add_analysis_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived variables used throughout the statistical analysis.
    """
    required = {"country", "quarter", "unemp", "cpi", "ipi", "cci", "gdp_growth"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required columns for feature engineering: {missing}")

    featured = panel.copy()
    featured["quarter"] = ensure_quarter_period(featured["quarter"])
    featured = featured.sort_values(["country", "quarter"]).reset_index(drop=True) #create a chronological order within each country 

    featured["year"] = featured["quarter"].dt.year
    featured["quarter_num"] = featured["quarter"].dt.quarter
    featured["period"] = featured["quarter"].apply(assign_period)
    featured["period"] = pd.Categorical(featured["period"], categories=PERIOD_ORDER, ordered=True)

    grouped = featured.groupby("country", group_keys=False)
    # Year-over-year (YoY) growth rates for CPI and IPI, calculated as percentage change from the same quarter in the previous year.
    featured["cpi_yoy"] = grouped["cpi"].pct_change(4) * 100
    featured["ipi_yoy"] = grouped["ipi"].pct_change(4) * 100
    featured["cpi_qoq"] = grouped["cpi"].pct_change() * 100 # Quarter-over-quarter (QoQ) growth rate for CPI, calculated as percentage change from the previous quarter.
    featured["high_inflation"] = (featured["cpi_qoq"] > 3.0).astype(int)
    featured["high_unemp"] = (featured["unemp"] > 6.5).astype(int)
    featured["trend"] = grouped.cumcount() + 1
    featured["is_turkiye"] = (featured["country"] == "TUR").astype(int)
    featured["group"] = np.where(featured["country"] == "TUR", "Turkiye", "Other OECD")
    return featured


def make_regression_data(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Return regression-ready observations with complete model variables.
    """
    required = ["unemp", "cpi_yoy", "ipi_yoy", "cci", "gdp_growth", "trend"]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(f"panel is missing required regression columns: {missing}")
    return panel.dropna(subset=required).copy()

