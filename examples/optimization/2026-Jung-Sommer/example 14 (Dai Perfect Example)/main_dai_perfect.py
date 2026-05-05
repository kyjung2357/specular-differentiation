import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_dai_perfect import DEFAULT_METHODS, run_experiment


if __name__ == "__main__":
    run_experiment(
        methods=DEFAULT_METHODS,
        max_iter=100,
        tol=1e-12,
        pdf=False,
        show=False,
    )
