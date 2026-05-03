import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_line_search import run_experiment


if __name__ == "__main__":
    methods = ["S-BFGS-A", "S-BFGS-W", "S-BFGS-S"]

    run_experiment(
        methods=methods,
        file_number=1,
        trials=20,
        iteration=300,
        m=50,
        n=100,
        lambda1=100.0,
        lambda2=1.0,
        alpha_0=1.0,
        c_1=1e-4,
        c_2=0.3,
        rho=0.5,
        max_line_iter=20,
        safeguard=1e-10,
        pdf=False,
        show=False,
    )
