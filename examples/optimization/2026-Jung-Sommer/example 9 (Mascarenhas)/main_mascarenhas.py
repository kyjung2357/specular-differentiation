import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_mascarenhas import run_experiment, save_reference_visualizations


if __name__ == "__main__":
    methods = [
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
        max_iter=30,
        tol=1e-12,
        pdf=False,
        show=False,
    )
    save_reference_visualizations(pdf=False, show=False)
