"""EDA summary tables for the OECD macro indicators project."""

import pandas as pd


ANALYSIS_VARIABLES = ["unemp", "cpi", "ipi", "cci", "gdp_growth"]


def descriptive_tables(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Create descriptive summary tables for EDA.
    """
    turkiye = panel[panel["country"] == "TUR"]
    other = panel[panel["country"] != "TUR"]
    return {
        "turkiye_descriptive_stats": turkiye[ANALYSIS_VARIABLES].describe().T.reset_index(names="variable"),
        "other_oecd_descriptive_stats": other[ANALYSIS_VARIABLES].describe().T.reset_index(names="variable"),
        "period_means": panel.groupby(["period"], observed=True)[ANALYSIS_VARIABLES].mean().reset_index(),
        "country_means": panel.groupby(["country", "country_name"], observed=True)[ANALYSIS_VARIABLES].mean().reset_index(),
    }

