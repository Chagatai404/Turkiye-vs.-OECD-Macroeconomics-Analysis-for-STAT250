"""Build the cleaned quarterly OECD panel and regression datasets."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clean_data import build_quarterly_panel
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features import add_analysis_features, make_regression_data
from src.load_data import load_raw_datasets
from src.utils import ensure_directories


def main() -> None:
    ensure_directories(PROCESSED_DATA_DIR)
    raw = load_raw_datasets(RAW_DATA_DIR)
    panel = build_quarterly_panel(raw)
    panel = add_analysis_features(panel) # add derived features for analysis
    regression_data = make_regression_data(panel)

    panel_path = PROCESSED_DATA_DIR / "clean_oecd_quarterly_panel.csv"
    regression_path = PROCESSED_DATA_DIR / "clean_oecd_regression_data.csv"
    panel.to_csv(panel_path, index=False)
    regression_data.to_csv(regression_path, index=False)

    # Console summary of the dataset build results
    print("STAT 250 OECD quarterly dataset build")
    print("=" * 48)
    print(f"Panel shape: {panel.shape[0]} rows x {panel.shape[1]} columns")
    print(f"Regression-ready shape: {regression_data.shape[0]} rows x {regression_data.shape[1]} columns")
    print("\nCountry counts:")
    print(panel["country"].value_counts().sort_index().to_string())
    print("\nMissing values:")
    print(panel.isna().sum().loc[lambda s: s > 0].to_string() or "No missing values in the panel.")
    print("\nDate range by country:")
    ranges = panel.groupby("country")["quarter"].agg(["min", "max", "count"]).reset_index()
    print(ranges.to_string(index=False))
    print(f"\nSaved: {panel_path}")
    print(f"Saved: {regression_path}")


if __name__ == "__main__": #only run the analysis if this file is executed as a script, not if imported as a module.
    main()

