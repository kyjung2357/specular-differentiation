import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from analysis import run_experiment, run_single_trial

if __name__ == '__main__':
    trials = 20
    iteration = 10000

    methods = ["SPEG-s", "SPEG-g", "GD", "Adam", "BFGS"]

    run_experiment(
        methods=methods,
        file_number=1,
        trials=trials,
        iteration=iteration,
        m=50,
        n=100,
        lambda1=0.01,
        lambda2=1.0,
        pdf=True
    )

    methods = ["SPEG-s", "SPEG-g", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]

    run_experiment(
        methods=methods,
        file_number=2,
        trials=trials,
        iteration=iteration,
        m=500,
        n=100,
        lambda1=0.0,
        lambda2=0.0,
        pdf=False
    )

    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]

    run_experiment(
        methods=methods,
        file_number=3,
        trials=trials,
        iteration=iteration,
        m=500,
        n=100,
        lambda1=0.0,
        lambda2=0.0,
        pdf=True
    )

    # ==========================================
    # For users who want to run many experiments
    # ==========================================
    #
    # methods = ["SPEG-s", "SPEG-g", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]    
    # M = [500, 500]
    # N = [100, 100]
    # Lambda1 = [100.0, 0.0]
    # Lambda2 = [1.0, 0.0]

    # for file_id in range(1, len(M) + 1):
    #     print(f"=== Experiment {file_id} Start ===")
    #     m, n, lambda1, lambda2 = M[file_id - 1], N[file_id - 1], Lambda1[file_id - 1], Lambda2[file_id - 1]
    
    #     run_experiment(
    #         methods=methods,
    #         file_number=file_id,
    #         trials=trials,
    #         iteration=iteration,
    #         m=m,
    #         n=n,
    #         lambda1=lambda1,
    #         lambda2=lambda2,
    #         pdf=False
    #     )
    
    # for file_id in range(1, len(M) + 1):
    #     print(f"=== Experiment {file_id} Start ===")
    #     m, n, lambda1, lambda2 = M[file_id - 1], N[file_id - 1], Lambda1[file_id - 1], Lambda2[file_id - 1]
    
        