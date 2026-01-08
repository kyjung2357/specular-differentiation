import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from analysis import run_experiment, run_single_trial

if __name__ == '__main__':
    m, n = 500, 100
    iteration = 10000
    trials = 20
    
    file_id = "5"
    lambda1 = 100.0
    lambda2 = 1.0

    print(f"=== Experiment {file_id} Start ===")
    run_experiment(
        file_number=file_id, 
        trials=trials, 
        iteration=iteration, 
        m=m, n=n, 
        lambda1=lambda1, lambda2=lambda2
    )