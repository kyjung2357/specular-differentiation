import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_hinge_svm import run_experiment


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
        file_number=11,
        trials=trials,
        iteration=iteration,
        samples=200,
        features=20,
        c_hinge=1.0,
        lambda2=1e-2,
        pdf=False,
        show=False,
    )
