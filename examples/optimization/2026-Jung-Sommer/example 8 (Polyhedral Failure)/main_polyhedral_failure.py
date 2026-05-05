import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_polyhedral_failure import run_experiment


if __name__ == "__main__":
    methods = [
        "SPEG",
        "Adam",
        "BFGS-E",
        "BFGS-S",
        "BFGS-W",
        "BFGS-A",
        "S-BFGS-E",
        "S-BFGS-S",
        "S-BFGS-W",
        "S-BFGS-A",
    ]

    run_experiment(
        methods=methods,
        trials=20,
        iteration=200,
        pdf=False,
        show=False,
    )
