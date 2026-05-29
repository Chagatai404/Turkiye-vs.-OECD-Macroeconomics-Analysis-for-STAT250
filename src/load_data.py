"""Raw data loading helpers."""

from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, RAW_FILE_CANDIDATES


def _resolve_data_file(raw_data_dir: Path, candidates: list[str]) -> Path: # Search for the first existing file among candidates from config, in both the raw data directory and project root.
    search_dirs = [Path(raw_data_dir), PROJECT_ROOT]
    for directory in search_dirs:
        for filename in candidates:
            path = directory / filename
            if path.exists():
                return path
    tried = ", ".join(str(directory / name) for directory in search_dirs for name in candidates)
    raise FileNotFoundError(f"Could not find any matching data file. Tried: {tried}")


def load_raw_datasets(raw_data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the five raw OECD datasets."""
    datasets = {}
    for key, candidates in RAW_FILE_CANDIDATES.items():
        path = _resolve_data_file(Path(raw_data_dir), candidates)
        datasets[key] = pd.read_csv(path, low_memory=False)
    return datasets

