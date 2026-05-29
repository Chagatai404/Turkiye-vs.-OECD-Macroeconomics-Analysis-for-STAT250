"""Small utility functions."""

from pathlib import Path

import pandas as pd


def ensure_directories(*paths: Path) -> None:
    """
    Create output directories if they do not already exist.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def ensure_quarter_period(series: pd.Series) -> pd.Series:
    """
    Return a pandas Period series with quarterly frequency.
    """
    if isinstance(series.dtype, pd.PeriodDtype):
        return series
    text = series.astype(str).str.replace("Q", "-Q", regex=False)
    text = text.str.replace("--", "-", regex=False)
    return pd.PeriodIndex(text, freq="Q").to_series(index=series.index)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame as CSV, creating the parent directory first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

