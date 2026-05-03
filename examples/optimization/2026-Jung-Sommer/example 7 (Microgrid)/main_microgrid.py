import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_microgrid import run_experiment


if __name__ == "__main__":
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS", "S-BFGS"]

    for c_dis, c_chg in [(5.0, 2.0), (500.0, 200.0)]:
        run_experiment(
            methods=methods,
            trials=20,
            iteration=1000,
            n=24,
            c_dis=c_dis,
            c_chg=c_chg,
            line_search="armijo",
            safeguard=1e-10,
            pdf=False,
            show=False,
        )
