import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from analysis_line_search import run_experiment


if __name__ == "__main__":
    methods = ["S-BFGS-E", "S-BFGS-S", "S-BFGS-W", "S-BFGS-A"]

if __name__ == "__main__":
    methods = ["S-BFGS-E", "S-BFGS-S", "S-BFGS-W", "S-BFGS-A"]

    run_experiment(
        "elastic_net",
        methods=methods,
        file_number=1,
        trials=10,
        iteration=1000,
        m=50,
        n=100,
        lambda1=100.0,
        lambda2=1.0,
        t_0=1e-3,
        c_1=1e-4,
        c_2=0.5,
        rho=0.5,
        max_line_iter=50,
        pdf=False,
        show=False,
    )

    run_experiment(
        "polyhedral_max",
        methods=methods,
        file_number=2,
        trials=10,
        iteration=1000,
        m=50,
        n=20,
        lambda1=10.0,
        lambda2=1.0,
        t_0=1e-2,
        c_1=1e-4,
        c_2=0.5,
        rho=0.5,
        max_line_iter=50,
        pdf=False,
        show=False,
    )

    run_experiment(
        "hinge_quadratic",
        methods=methods,
        file_number=3,
        trials=10,
        iteration=1000,
        m=80,
        n=30,
        lambda1=10.0,
        lambda2=1.0,
        t_0=1e-2,
        c_1=1e-4,
        c_2=0.5,
        rho=0.5,
        max_line_iter=50,
        pdf=False,
        show=False,
    )
