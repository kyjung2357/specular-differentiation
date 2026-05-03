import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_grasp_force import run_experiment


if __name__ == "__main__":
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS", "S-BFGS"]

    for lam in [100.0, 10000.0]:
        run_experiment(
            methods=methods,
            trials=20,
            iteration=1000,
            c_points=3,
            k=4,
            mu=0.5,
            lam=lam,
            line_search="armijo",
            safeguard=1e-10,
            pdf=False,
            show=False,
        )
