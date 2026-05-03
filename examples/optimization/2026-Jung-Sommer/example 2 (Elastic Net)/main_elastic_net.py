import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_elastic_net import run_experiment


if __name__ == "__main__":
    trials = 20
    iteration = 10000
    line_search = "armijo"
    safeguard = 1e-10

    # Figure 5
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS", "S-BFGS"]

    run_experiment(
        methods=methods,
        file_number=5,
        trials=trials,
        iteration=iteration,
        m=500,
        n=100,
        lambda1=100.0,
        lambda2=1.0,
        line_search=line_search,
        safeguard=safeguard,
        pdf=False,
        show=False,
    )