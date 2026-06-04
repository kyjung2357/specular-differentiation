import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
OPTIMIZATION_DIR = CURRENT_DIR.parents[0]
PACKAGE_ROOT = CURRENT_DIR.parents[2]

for path in (PACKAGE_ROOT, OPTIMIZATION_DIR, CURRENT_DIR):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)

sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(OPTIMIZATION_DIR))
sys.path.insert(0, str(PACKAGE_ROOT))

import specular
from specular.optimization.solver import adam, bfgs, gradient_descent, specular_gradient, stochastic_specular_gradient, hybrid_specular_gradient
from tools import *

specular.change_backend("cpu_numpy")

# ============================================================================--
# 1. Single Trial Execution
# ============================================================================--
def run_single_trial(args):
    trial_idx, m, n, lambda1, lambda2, iteration, methods = args
    
    np.random.seed(trial_idx) 
    torch.manual_seed(trial_idx)
    torch.set_num_threads(1)
    
    A_np = np.random.randn(m, n)
    b_np = np.random.randn(m)
    x_0 = np.random.randn(n)
    
    A_torch = torch.tensor(A_np, dtype=torch.float32)
    b_torch = torch.tensor(b_np, dtype=torch.float32)

    def f(x):
        x = np.atleast_1d(x)
        return (1/(2*m))*np.sum((A_np @ x - b_np)**2) + (lambda2/2)*np.sum(x**2) + lambda1*np.sum(np.abs(x))

    def f_torch(x_tensor):
        residual = A_torch @ x_tensor - b_torch
        loss_term = (1/(2*m))*torch.sum(residual**2)
        l2_regularization = (lambda2/2)*torch.sum(x_tensor**2)    
        l1_regularization = lambda1*torch.sum(torch.abs(x_tensor)) 
        return loss_term + l2_regularization + l1_regularization

    def f_stochastic(x, j=False):
        x = np.asarray(x)
        if j is False: return f(x)
        
        if x.ndim == 1:
            term_data = (np.dot(A_np[j], x) - b_np[j])**2
            term_reg2 = np.sum(x**2)
            term_reg1 = np.sum(np.abs(x))
        else:
            term_data = (x @ A_np[j] - b_np[j])**2
            term_reg2 = np.sum(x**2, axis=1)
            term_reg1 = np.sum(np.abs(x), axis=1)

        return 0.5 * term_data + (lambda2/2) * term_reg2 + lambda1 * term_reg1

    def make_component(j):
        def f_component(x):
            return f_stochastic(x, j)
        return f_component

    f_components = [make_component(j) for j in range(m)]

    trial_results = {}
    trial_times = {}

    # ==== Specular gradient methods ====
    
    # SPEG with square summable step size
    if "SPEG" in methods:
        res_obj = specular_gradient(
            objective_function=f, initial_point=x_0, step_size='square_summable_not_summable', a=4.0, b=0.0, tol=1e-10, max_iter=iteration, print_bar=True
        )
        _, res, runtime = res_obj.get_history()
        trial_results["SPEG"] = ensure_length(res, iteration)
        trial_times["SPEG"] = runtime

    # SPEG with square summable step size
    if "SPEG-s" in methods:
        res_obj = specular_gradient(
            objective_function=f, initial_point=x_0, step_size='square_summable_not_summable', a=4.0, b=0.0, tol=1e-10, max_iter=iteration, print_bar=True
        )
        _, res, runtime = res_obj.get_history()
        trial_results["SPEG-s"] = ensure_length(res, iteration)
        trial_times["SPEG-s"] = runtime
    
    # SPEG with geometric step size
    if "SPEG-g" in methods:
        res_obj = specular_gradient(
            objective_function=f, initial_point=x_0, step_size='geometric_series', a=1.0, r=0.5, tol=1e-10, max_iter=iteration, print_bar=True
        )
        _, res, runtime = res_obj.get_history()
        trial_results["SPEG-g"] = ensure_length(res, iteration)
        trial_times["SPEG-g"] = runtime

    # S-SPEG
    if "S-SPEG" in methods:
        res_obj = stochastic_specular_gradient(
            objective_function=f, initial_point=x_0, step_size='square_summable_not_summable', a=4.0, b=0.0, f_j=f_components, tol=1e-10, max_iter=iteration, print_bar=True
        )
        _, res, runtime = res_obj.get_history()
        trial_results["S-SPEG"] = ensure_length(res, iteration)
        trial_times["S-SPEG"] = runtime
    
    # H-SPEG
    if "H-SPEG" in methods:
        res_obj = hybrid_specular_gradient(
            objective_function=f, initial_point=x_0, step_size='square_summable_not_summable', a=4.0, b=0.0, f_j=f_components, switch_iter=10, tol=1e-10, max_iter=iteration, print_bar=True
        )
        _, res, runtime = res_obj.get_history()
        trial_results["H-SPEG"] = ensure_length(res, iteration)
        trial_times["H-SPEG"] = runtime

    # ==== Classical Methods ====

    # Gradient Descent
    if "GD" in methods:
        res_obj = gradient_descent(
            objective_function=f, initial_point=x_0, step_size='constant', a=0.001, max_iter=iteration
        ) # Changed f_torch to f since gradient_descent now natively supports callables in numpy
        _, res, runtime = res_obj.get_history()
        trial_results["GD"] = ensure_length(res, iteration)
        trial_times["GD"] = runtime

    # Adam
    if "Adam" in methods:
        res_obj = adam(
            objective_function=f, initial_point=x_0, step_size='constant', a=0.01, max_iter=iteration
        )
        _, res, runtime = res_obj.get_history()
        trial_results["Adam"] = ensure_length(res, iteration)
        trial_times["Adam"] = runtime

    # BFGS
    if "BFGS" in methods:
        res_obj = bfgs(
            objective_function=f, initial_point=x_0, max_iter=iteration, tol=1e-6, skip_on_fail=True
        )
        _, res, runtime = res_obj.get_history()
        trial_results["BFGS"] = ensure_length(res, iteration)
        trial_times["BFGS"] = runtime
    
    return trial_results, trial_times

# ============================================================================--
# 2. Main Analysis Logic
# ============================================================================--
def run_experiment(methods, file_number, trials, iteration, m, n, lambda1, lambda2, pdf=False, show=False):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, lambda_1={lambda1}, lambda_2={lambda2}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}

    tasks = [(i, m, n, lambda1, lambda2, iteration, methods) for i in range(trials)]
    
    num_workers = min(os.cpu_count(), trials) # type: ignore
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_trial, task) for task in tasks] # type: ignore
        
        for future in tqdm(as_completed(futures), total=trials, desc="Processing Trials", leave=False):
            try:
                t_res, t_time = future.result()
                
                for method in methods:
                    if method in t_res:
                        all_results[method].append(t_res[method])
                    
                    if method in t_time:
                        running_times[method].append(t_time[method])
                        
            except Exception as e:
                import traceback
                print(f"\n[Error] Trial failed: {e}")
                traceback.print_exc()

    # ==== Visualization & Analysis ====
    print("\n[Analysis]")
    print(" Generating plots and tables")
 
    report_results(all_results, running_times, file_number, m, n, lambda1, lambda2, iteration, CURRENT_DIR, pdf=pdf, show=show)