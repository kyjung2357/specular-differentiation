import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_nesterov import run_experiment


if __name__ == "__main__":
    trials = 100
    iteration = 10000

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

    experiments = [
        (1, 2.0, 2.0, "smooth"),
        (2, 2.0, 1.0, "q_nonsmooth"),
        (3, 1.0, 1.0, "p_q_nonsmooth"),
    ]

    for file_number, p, q, label in experiments:
        for n in [2, 3, 4, 5, 10, 50, 100]:
            run_experiment(
                methods=methods,
                file_number=file_number,
                trials=trials,
                iteration=iteration,
                n=n,
                p=p,
                q=q,
                label=label,
                pdf=False,
                show=False,
            )
