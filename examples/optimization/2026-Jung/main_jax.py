import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from analysis_jax import run_experiment


if __name__ == "__main__":
    trials = 20

    # Figure 3
    methods = ["SPEG-s", "SPEG-g", "GD", "Adam", "BFGS"]

    run_experiment(
        methods=methods,
        file_number=3,
        trials=trials,
        iteration=10000,
        m=50,
        n=100,
        lambda1=0.01,
        lambda2=1.0,
        pdf=True,
        show=False
    )
    
    # Figure 5
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]

    run_experiment(
        methods=methods,
        file_number=5,
        trials=trials,
        iteration=10000,
        m=500,
        n=100,
        lambda1=100.0,
        lambda2=1.0,
        pdf=True,
        show=False
    )
    
    # Figure 6
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]

    run_experiment(
        methods=methods,
        file_number=6,
        trials=trials,
        iteration=10000,
        m=500,
        n=100,
        lambda1=0.0,
        lambda2=0.0,
        pdf=True,
        show=False
    )
