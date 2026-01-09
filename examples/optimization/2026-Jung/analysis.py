import specular
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method
from tools import ensure_length, format_sci_latex, save_table_to_txt

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

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

    trial_results = {}
    trial_times = {}

    step_size_squ = specular.StepSize(name='square_summable_not_summable', parameters=[4.0, 0.0])
    step_size_geo = specular.StepSize(name='geometric_series', parameters=[1.0, 0.5])

    # ==== Specular gradient methods ====
    
    # SPEG with square summable step size
    if "SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f, x_0=x_0, step_size=step_size_squ, tol=1e-10, max_iter=iteration, print_bar=True
        ).history()
        trial_results["SPEG"] = ensure_length(res, iteration)
        trial_times["SPEG"] = runtime

    # SPEG with square summable step size
    if "SPEG-s" in methods:
        _, res, runtime = specular.gradient_method(
            f=f, x_0=x_0, step_size=step_size_squ, tol=1e-10, max_iter=iteration, print_bar=True
        ).history()
        trial_results["SPEG-s"] = ensure_length(res, iteration)
        trial_times["SPEG-s"] = runtime
    
    # SPEG with geometric step size
    if "SPEG-g" in methods:
        _, res, runtime = specular.gradient_method(
            f=f, x_0=x_0, step_size=step_size_geo, tol=1e-10, max_iter=iteration, print_bar=True
        ).history()
        trial_results["SPEG-g"] = ensure_length(res, iteration)
        trial_times["SPEG-g"] = runtime

    # SSPEG
    if "S-SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f, x_0=x_0, step_size=step_size_squ, form='stochastic', tol=1e-10, max_iter=iteration, f_j=f_stochastic, m=m, print_bar=True # type: ignore
        ).history()
        trial_results["S-SPEG"] = ensure_length(res, iteration)
        trial_times["S-SPEG"] = runtime
    
    # HSPEG
    if "H-SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f, x_0=x_0, step_size=step_size_squ, form='hybrid', tol=1e-10, max_iter=iteration, f_j=f_stochastic, m=m,switch_iter=10, print_bar=True # type: ignore
        ).history()
        trial_results["H-SPEG"] = ensure_length(res, iteration)
        trial_times["H-SPEG"] = runtime

    # ==== Classical Methods ====

    # Gradient Descent
    if "GD" in methods:
        constant_step_size = specular.StepSize(name='constant', parameters=0.001)
        _, res, runtime = gradient_descent_method(
            f_torch=f_torch, x_0=x_0, step_size=constant_step_size, max_iter=iteration
        ).history()
        trial_results["GD"] = ensure_length(res, iteration)
        trial_times["GD"] = runtime

    # Adam
    if "Adam" in methods:
        _, res, runtime = Adam(
            f_torch=f_torch, x_0=x_0, step_size=0.01, max_iter=iteration
        ).history()
        trial_results["Adam"] = ensure_length(res, iteration)
        trial_times["Adam"] = runtime

    # BFGS
    if "BFGS" in methods:
        _, res, runtime = BFGS(
            f_np=f, x_0=x_0, max_iter=iteration, tol=1e-6
        ).history()
        trial_results["BFGS"] = ensure_length(res, iteration)
        trial_times["BFGS"] = runtime
    
    return trial_results, trial_times

# ============================================================================--
# 2. Main Analysis Logic
# ============================================================================--
def run_experiment(methods, file_number, trials, iteration, m, n, lambda1, lambda2, pdf=False):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, λ1={lambda1}, λ2={lambda2}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}
    summary_stats = {}

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

    colors = {'SPEG': 'red', 'SPEG-s': 'red', 'SPEG-g': 'brown', 'S-SPEG': 'blue', 'H-SPEG': 'purple', 'GD': 'orange', 'Adam': 'green', 'BFGS': 'black'}
    
    plt.figure(figsize=(6, 3))

    for name, results_list in all_results.items():
        if not results_list: continue

        df = pd.DataFrame(results_list).T
        df.columns = [f'trial_{j+1}' for j in range(len(results_list))]
        
        min_vals = df.min(axis=0)
        mean_curve = df.mean(axis=1)
        median_curve = df.median(axis=1)
        std_curve = df.std(axis=1)

        summary_stats[name] = {
            'Mean': min_vals.mean(),
            'Median': min_vals.median(),
            'Standard deviation': min_vals.std()
        }

        x_data = df.index + 1
        
        plt.plot(x_data, mean_curve, label=name, color=colors.get(name, 'black'), linewidth=1.5)
        
        plt.plot(x_data, median_curve, color=colors.get(name, 'black'), linestyle='--', alpha=0.5, linewidth=1)
        
        plt.fill_between(
            x_data, 
            (mean_curve - std_curve), 
            (mean_curve + std_curve),
            color=colors.get(name, 'black'), 
            alpha=0.15
        )

    # ==== Summary & Save ====
    print("\n[Running Time Summary]")
    for name, times in running_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"{name:5s} : {avg_time:.5f} sec")

    print("\n[Final Performance Summary]")
    summary_df = pd.DataFrame(summary_stats).T
    display_df = summary_df.copy()

    for col in display_df.columns:
        display_df[col] = display_df[col].apply(format_sci_latex)
        
    pd.options.display.float_format = '{:.4e}'.format
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'figures'), exist_ok=True)
    
    path_txt = os.path.join(base_dir, f'tables/table{file_number}-{m}-{n}-{lambda1}-{lambda2}.txt')
    path_fig = os.path.join(base_dir, f'figures/figure{file_number}-{m}-{n}-{lambda1}-{lambda2}.{"pdf" if pdf else "png"}')

    with open(os.path.join(base_dir, path_txt), "w", encoding="utf-8") as table_file:
        table_file.write(display_df.to_latex(escape=False))

    print(summary_df)

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(x_k)$", fontsize=10)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, iteration)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

    plt.tight_layout() 
    plt.savefig(path_fig, dpi=1000, bbox_inches='tight')