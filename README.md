# STAT 250 OECD Macroeconomic Indicators Project

Our project analyzes quarterly macroeconomic indicators for selected OECD countries, with a focus on Turkiye. Monthly CPI, IPI, unemployment, and CCI data are aggregated to quarterly means, then merged with quarterly GDP growth.

## Project Structure

- `data/raw/`: contains raw OECD CSV files *(OECD, 2026) accessed from https://data-explorer.oecd.org*
- `data/processed/`: contains cleaned quarterly panel and regression-ready CSV files
- `src/`: contains modules for loading, cleaning, features, testing, regression, EDA, and plotting 
- `scripts/01_build_dataset.py`: builds the processed datasets
- `stat250_project.py`: runs EDA, hypothesis tests, ANOVA, regression, and output generation through the main function in `src/analysis.py`
- `scripts/02_run_analysis(main).py`: compatibility wrapper for the analysis script (practically same with `stat250_project.py`)
- `outputs/figures/`: contains saved PNG figures
- `outputs/tables/`: contains saved CSV and TXT tables

## How To Run

Install the required Python packages if they are not already available:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then run the project from the project root:

```bash
.venv\Scripts\python scripts\01_build_dataset.py
.venv\Scripts\python stat250_project.py
```

If you are using an already activated environment, the equivalent commands are:

```bash
python -m pip install -r requirements.txt
python scripts/01_build_dataset.py
python stat250_project.py
```

The first script creates:

- `data/processed/clean_oecd_quarterly_panel.csv`
- `data/processed/clean_oecd_regression_data.csv`

The analysis script saves report-ready figures and tables under `outputs/`.

## Research Questions

1. Is Turkiye's average unemployment rate over 2005-2025 significantly different from the OECD reference value of 6.5%?
2. During COVID-19, did Turkiye's average Industrial Production Index differ from the average of the other OECD comparison countries?
3. Did Turkiye experience high inflation in more than half of observed quarters, where high inflation is quarterly CPI growth greater than 3%?
4. Is Turkiye's high-inflation quarter ratio significantly different from the ratio for the other OECD comparison countries?
5. Do unemployment rates across selected OECD countries differ significantly across macroeconomic periods?
6. For Turkiye, which macroeconomic variables are associated with unemployment?

**NOTE:** We avoided causal language in the regression section because the analysis is based on observational macroeconomic time-series data.
