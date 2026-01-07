import specular
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method
from tools import ensure_length, format_sci_latex, save_table_to_txt

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

# -----------------------------------------------------------
# 1. Single Trial Execution
# -----------------------------------------------------------
def run_single_trial(trial_idx, m, n, lambda1, lambda2, iteration):
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

    step_size_obj = specular.StepSize(name='square_summable_not_summable', parameters=[4.0, 0.0])

    # ==== Specular gradient methods ====
    
    # SPEG
    start_time = time.time()
    _, res = specular.gradient_method(
        f=f, x_0=x_0, step_size=step_size_obj, tol=1e-10, max_iter=iteration, print_bar=True
    ).history()
    trial_results["SPEG"] = ensure_length(res, iteration)
    trial_times["SPEG"] = time.time() - start_time

    # SSPEG
    start_time = time.time()
    _, res = specular.gradient_method(
        f=f, x_0=x_0, step_size=step_size_obj, form='stochastic', tol=1e-10, max_iter=iteration, f_j=f_stochastic, m=m, print_bar=True # type: ignore
    ).history()
    trial_results["SSPEG"] = ensure_length(res, iteration)
    trial_times["SSPEG"] = time.time() - start_time
    
    # # HSPEG
    start_time = time.time()
    _, res = specular.gradient_method(
        f=f, x_0=x_0, step_size=step_size_obj, form='hybrid', tol=1e-10, max_iter=iteration, f_j=f_stochastic, m=m,switch_iter=10, print_bar=True # type: ignore
    ).history()
    trial_results["HSPEG"] = ensure_length(res, iteration)
    trial_times["HSPEG"] = time.time() - start_time

    # ==== Classical Methods ====

    # Gradient Descent
    start_time = time.time()
    _, res = gradient_descent_method(
        f_torch=f_torch, x_0=x_0, step_size=step_size_obj, max_iter=iteration
    ).history()
    trial_results["GD"] = ensure_length(res, iteration)
    trial_times["GD"] = time.time() - start_time

    # Adam
    start_time = time.time()
    _, res = Adam(
        f_torch=f_torch, x_0=x_0, step_size=0.01, max_iter=iteration
    ).history()
    trial_results["Adam"] = ensure_length(res, iteration)
    trial_times["Adam"] = time.time() - start_time

    # BFGS
    start_time = time.time()
    _, res = BFGS(
        f_np=f, x_0=x_0, max_iter=iteration, tol=1e-6
    ).history()
    trial_results["BFGS"] = ensure_length(res, iteration)
    trial_times["BFGS"] = time.time() - start_time

    return trial_results, trial_times

# -----------------------------------------------------------
# 2. Main Analysis Logic
# -----------------------------------------------------------
def run_experiment(file_number, trials, iteration, part_of_iteration, m, n, lambda1, lambda2):
    
    # 결과 저장소
    all_results = {"SPEG": [], "GD": [], "SSPEG": [], "Adam": [], "HSPEG": [], "BFGS": []}
    running_times = {"SPEG": [], "GD": [], "SSPEG": [], "Adam": [], "HSPEG": [], "BFGS": []}
    summary_stats = {}

    print(f"Starting {trials} trials | m={m}, n={n}, λ1={lambda1}, λ2={lambda2}")

    tasks = [(i, m, n, lambda1, lambda2, iteration) for i in range(trials)]
    
    # 병렬 처리
    num_workers = min(os.cpu_count(), trials) # type: ignore
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_trial, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=trials, desc="Progress"):
            try:
                t_res, t_time = future.result()
                for key in all_results.keys():
                    all_results[key].append(t_res[key])
                    running_times[key].append(t_time[key])
            except Exception as e:
                print(f"Trial failed: {e}")

    # --- Visualization & Analysis ---
    colors = {'SPEG': 'red', 'GD': 'orange', 'SSPEG': 'blue', 'Adam': 'skyblue', 'HSPEG': 'purple', 'BFGS': 'black'}
    
    # GridSpec Plot
    plt.figure(figsize=(7.5, 3))

    for name, results_list in all_results.items():
        if not results_list: continue

        df = pd.DataFrame(results_list).T
        df.columns = [f'trial_{j+1}' for j in range(len(results_list))]
        
        # 통계 계산
        min_vals = df.min(axis=0)
        mean_curve = df.mean(axis=1)
        std_curve = df.std(axis=1)

        # 요약 통계 저장
        summary_stats[name] = {
            'Avg Min': min_vals.mean(),
            'Std Min': min_vals.std()
        }

        # Plot Data 준비
        x_data = df.index + 1
        x_part = x_data[:part_of_iteration]
        mean_part = mean_curve.iloc[:part_of_iteration]
        std_part = std_curve.iloc[:part_of_iteration]

        # 하단 그래프 (전체 구간)
        plt.plot(mean_part, label=name, color=colors[name])
        plt.fill_between(
            x_data, 
            (mean_curve - std_curve), 
            (mean_curve + std_curve),
            color=colors[name], 
            alpha=0.15
        )

    # --- Summary & Save ---
    print("\n[Running Time Summary]")
    for name, times in running_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"{name:5s} : {avg_time:.5f} sec")

    print("\n[Final Performance Summary]")
    summary_df = pd.DataFrame(summary_stats).T
    pd.options.display.float_format = '{:.4e}'.format
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'figures'), exist_ok=True)

    path_txt = os.path.join(base_dir, f'tables/table_{file_number}.txt')
    path_fig = os.path.join(base_dir, f'figures/figure_{file_number}.png')

    save_table_to_txt(
        summary_df, 
        filename=path_txt,
        precision=4,
        formatting="exponential"
    )

    print(summary_df)

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(x_k)$", fontsize=10)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, iteration)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

    plt.tight_layout() 
    plt.savefig(path_fig, dpi=300, bbox_inches='tight')
    plt.show()