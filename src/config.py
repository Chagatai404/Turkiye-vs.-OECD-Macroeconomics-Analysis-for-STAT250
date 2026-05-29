"""Configuration for our project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

COUNTRIES = ["TUR", "USA", "DEU", "FRA", "GBR", "JPN", "KOR", "POL", "MEX", "ITA"]

COUNTRY_NAMES = {
    "TUR": "Turkiye",
    "USA": "United States",
    "DEU": "Germany",
    "FRA": "France",
    "GBR": "United Kingdom",
    "JPN": "Japan",
    "KOR": "South Korea",
    "POL": "Poland",
    "MEX": "Mexico",
    "ITA": "Italy",
}

REGIONS = {
    "TUR": "Europe",
    "USA": "Americas",
    "DEU": "Europe",
    "FRA": "Europe",
    "GBR": "Europe",
    "JPN": "Asia-Pacific",
    "KOR": "Asia-Pacific",
    "POL": "Europe",
    "MEX": "Americas",
    "ITA": "Europe",
}

START_YEAR = 2005
END_YEAR = 2025

RAW_FILE_CANDIDATES = { # our "data/raw" contains only the first names.
    "cpi": ["cpi.csv", "OECD CPI 2000 - 2025.csv"],
    "ipi": ["ipi.csv", "OECD IPI 2000-2025.csv"],
    "unemployment": ["unemployment.csv", "unep.csv", "OECD Unemployment Rate 2000-2025.csv"],
    "cci": ["cci.csv", "OECD Composite Consumer Confidence 2000-2025.csv"],
    "gdp": ["gdp.csv", "OECD GDP Growth 2000 - 2025.csv"],
}

