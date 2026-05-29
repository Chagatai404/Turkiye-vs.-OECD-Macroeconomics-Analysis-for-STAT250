"""Cleaning and merging functions for the OECD quarterly panel."""

from functools import reduce

import pandas as pd

from .config import COUNTRIES, COUNTRY_NAMES, END_YEAR, REGIONS, START_YEAR


REQUIRED_MONTHLY_COLUMNS = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE"}
REQUIRED_GDP_COLUMNS = {"REF_AREA", "TIME_PERIOD", "OBS_VALUE", "TRANSACTION", "TRANSFORMATION"}


def _validate_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def clean_monthly_indicator(
    df: pd.DataFrame,
    value_name: str,
    countries: list[str],
    filters: dict[str, str] | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Clean monthly OECD indicator data (all except GDP) and aggregate it to quarterly frequency.
    """
    _validate_columns(df, REQUIRED_MONTHLY_COLUMNS, value_name)

    cleaned = df[df["REF_AREA"].isin(countries)].copy()
    for column, expected_value in (filters or {}).items():
        if column not in cleaned.columns:
            raise ValueError(f"{value_name} filter column '{column}' is not present in the dataset.")
        cleaned = cleaned[cleaned[column] == expected_value]

    cleaned["date"] = pd.to_datetime(cleaned["TIME_PERIOD"], format="%Y-%m", errors="coerce")
    cleaned[value_name] = pd.to_numeric(cleaned["OBS_VALUE"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", value_name])
    cleaned["quarter"] = cleaned["date"].dt.to_period("Q")

    quarterly = (
        cleaned.groupby(["REF_AREA", "quarter"], as_index=False)[value_name]
        .agg(agg)
        .rename(columns={"REF_AREA": "country"})
    )
    return quarterly[["country", "quarter", value_name]]


def clean_quarterly_gdp(df: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """
    Clean OECD quarterly GDP growth data.
    """
    _validate_columns(df, REQUIRED_GDP_COLUMNS, "gdp")

    gdp = df[(df["REF_AREA"].isin(countries)) & (df["TRANSACTION"] == "B1GQ")].copy()
    transformations = set(gdp["TRANSFORMATION"].dropna().unique())
    if "GY" in transformations:
        transformation = "GY"
    elif "G1" in transformations:
        transformation = "G1"
        print("Warning: GDP year-over-year growth (GY) was not found; using quarter-on-quarter growth (G1).")
    else:
        raise ValueError("GDP data must contain TRANSFORMATION == 'GY' or 'G1' for transaction B1GQ.")

    gdp = gdp[gdp["TRANSFORMATION"] == transformation].copy()
    gdp["quarter"] = pd.PeriodIndex(gdp["TIME_PERIOD"].astype(str), freq="Q")
    gdp["gdp_growth"] = pd.to_numeric(gdp["OBS_VALUE"], errors="coerce")
    gdp = gdp.dropna(subset=["gdp_growth"])

    gdp = (
        gdp.groupby(["REF_AREA", "quarter"], as_index=False)["gdp_growth"]
        .mean()
        .rename(columns={"REF_AREA": "country"})
    )
    return gdp[["country", "quarter", "gdp_growth"]]


def _available_filters(df: pd.DataFrame, requested: dict[str, str]) -> dict[str, str]:
    """
    Apply optional filters only when the corresponding OECD column exists.
    """
    return {column: value for column, value in requested.items() if column in df.columns}


def build_quarterly_panel(raw_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the cleaned quarterly OECD panel containing all five indicators.
    """
    required_keys = {"cpi", "ipi", "unemployment", "cci", "gdp"}
    missing = sorted(required_keys - set(raw_datasets))
    if missing:
        raise ValueError(f"raw_datasets is missing required datasets: {missing}")

    cpi = clean_monthly_indicator(raw_datasets["cpi"], "cpi", COUNTRIES)
    ipi = clean_monthly_indicator(
        raw_datasets["ipi"],
        "ipi",
        COUNTRIES,
        filters=_available_filters(raw_datasets["ipi"], {"ACTIVITY": "C", "ADJUSTMENT": "Y"}), #filter manufacturing and seasonally adjusted data
    )
    unemp = clean_monthly_indicator(raw_datasets["unemployment"], "unemp", COUNTRIES)
    cci = clean_monthly_indicator(raw_datasets["cci"], "cci", COUNTRIES)
    gdp = clean_quarterly_gdp(raw_datasets["gdp"], COUNTRIES)

    panel = reduce(
        lambda left, right: left.merge(right, on=["country", "quarter"], how="inner"),
        [unemp, cpi, ipi, cci, gdp],
    )

    panel["country_name"] = panel["country"].map(COUNTRY_NAMES)
    panel["region"] = panel["country"].map(REGIONS)
    panel["year"] = panel["quarter"].dt.year
    panel["quarter_num"] = panel["quarter"].dt.quarter
    panel = panel[panel["year"].between(START_YEAR, END_YEAR)].copy()
    panel = panel.sort_values(["country", "quarter"]).reset_index(drop=True)

    columns = [
        "country",
        "country_name",
        "region",
        "quarter",
        "year",
        "quarter_num",
        "unemp",
        "cpi",
        "ipi",
        "cci",
        "gdp_growth",
    ]
    return panel[columns]

