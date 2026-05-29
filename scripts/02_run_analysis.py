"""
Compatibility wrapper for the OECD analysis script. 
The main function is defined in src/analysis.py, but this script allows it to be run directly from the project root with `python scripts/02_run_analysis.py`.
This is an old version of the script that was renamed to "stat250_project.py" in the project root to avoid confusion with the main analysis script. We kept it here for reference.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import main


if __name__ == "__main__":
    main()
