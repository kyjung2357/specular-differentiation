import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_max_regression import run_experiment


if __name__ == "__main__":
    trials = 20
    iteration = 2000

    methods = [
        "SPEG",
        "GD",
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
        file_number=13,
        trials=trials,
        iteration=iteration,
        samples=250,
        features=30,
        lambda1=1e-2,
        lambda2=1e-3,
        outlier_scale=3.0,
        pdf=False,
        show=False,
    )
