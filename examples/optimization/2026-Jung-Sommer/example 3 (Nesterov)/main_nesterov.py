import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_nesterov import run_experiment


if __name__ == "__main__":
    trials = 20
    iteration = 10000
    line_search = "armijo"
    safeguard = 1e-10

    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS", "S-BFGS"]

    experiments = [
        (1, 2.0, 2.0, "smooth"),
        (2, 2.0, 1.0, "q_nonsmooth"),
        (3, 1.0, 1.0, "p_q_nonsmooth"),
    ]

    for file_number, p, q, label in experiments:
        for n in [2, 5, 10, 50, 100]:
            run_experiment(
                methods=methods,
                file_number=file_number,
                trials=trials,
                iteration=iteration,
                n=n,
                p=p,
                q=q,
                label=label,
                line_search=line_search,
                safeguard=safeguard,
                pdf=False,
                show=False,
            )
