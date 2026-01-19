import specular 
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

# ------------------------------------------------------------------------------
# 1. Single Trial Function 
# ------------------------------------------------------------------------------
def run_single_trial_lambda(args):
    trial_idx, m, n, lambda1, lambda2, iteration = args
    
    np.random.seed(trial_idx) 
    torch.manual_seed(trial_idx)
    
    A_np = np.random.randn(m, n)
    b_np = np.random.randn(m)
    x_0 = np.random.randn(n)

    def f(x):
        x = np.atleast_1d(x)
        return (1/(2*m))*np.sum((A_np @ x - b_np)**2) + (lambda2/2)*np.sum(x**2) + lambda1*np.sum(np.abs(x))

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

    step_size_squ = specular.StepSize(name='square_summable_not_summable', parameters=[4.0, 0.0])
    
    trial_results = {}

    # 1. SPEG
    _, res, _ = specular.gradient_method(
        f=f, x_0=x_0, step_size=step_size_squ, tol=1e-10, max_iter=iteration, print_bar=False
    ).history()
    trial_results["SPEG"] = res

    # 2. S-SPEG
    _, res, _ = specular.gradient_method(
        f=f, x_0=x_0, step_size=step_size_squ, form='stochastic', 
        tol=1e-10, max_iter=iteration, f_j=f_stochastic, m=m, print_bar=False
    ).history()
    trial_results["S-SPEG"] = res

    # 3. H-SPEG
    _, res, _ = specular.gradient_method(
        f=f, x_0=x_0, step_size=step_size_squ, form='hybrid', 
        tol=1e-10, max_iter=iteration, f_j=f_stochastic, m=m, switch_iter=10, print_bar=False
    ).history()
    trial_results["H-SPEG"] = res

    return trial_results

# ------------------------------------------------------------------------------
# 2. Experiment & Plotting Function
# ------------------------------------------------------------------------------
def run_experiment(file_number, trials, iteration, m, n, pdf=False):
    lambda_settings = [
        (0.1, 0.1),   
        (1.0, 1.0),   
        (10.0, 10.0)  
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(6, 3)) 
    plt.subplots_adjust(wspace=0.3)
    
    colors = {'SPEG': 'red', 'S-SPEG': 'blue', 'H-SPEG': 'purple'}
    labels = {'SPEG': 'SPEG', 'S-SPEG': 'S-SPEG', 'H-SPEG': 'H-SPEG'}
    
    for idx, (lambda1, lambda2) in enumerate(lambda_settings):
        ax = axes[idx]
        print(f"\n[Processing Case {idx+1}] lambda1={lambda1}, lambda2={lambda2}")

        tasks = [(i, m, n, lambda1, lambda2, iteration) for i in range(trials)]
        
        all_results = {"SPEG": [], "S-SPEG": [], "H-SPEG": []}

        num_workers = min(os.cpu_count(), trials) # type: ignore
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_single_trial_lambda, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=trials, desc=f"Simulating Case {idx+1}", leave=False):
                try:
                    t_res = future.result()
                    for method in all_results.keys():
                        if method in t_res:
                            full_res = t_res[method]
                            if len(full_res) < iteration:
                                full_res = np.pad(full_res, (0, iteration - len(full_res)), 'edge')
                            elif len(full_res) > iteration:
                                full_res = full_res[:iteration]
                            all_results[method].append(full_res)
                except Exception as e:
                    print(f"Error: {e}")

        x_data = np.arange(1, iteration + 1)

        for name in ["SPEG", "S-SPEG", "H-SPEG"]: 
            results_list = all_results[name]
            if not results_list: continue

            df = pd.DataFrame(results_list).T 
            
            mean_curve = df.mean(axis=1)
            median_curve = df.median(axis=1)
            std_curve = df.std(axis=1)
            
            ax.plot(x_data, mean_curve, label=labels[name], color=colors[name], linewidth=1.5)
            
            ax.plot(x_data, median_curve, color=colors[name], linestyle='--', alpha=0.6, linewidth=1)
            
            ax.fill_between(
                x_data, 
                (mean_curve - std_curve), 
                (mean_curve + std_curve),
                color=colors[name], 
                alpha=0.15
            )

        ax.set_title(r"$\lambda_1=%.1f, \lambda_2=%.1f$" % (lambda1, lambda2), fontsize=11)
        ax.set_xlabel(r"Iterations $k$", fontsize=10)
        
        if idx == 0:
            ax.set_ylabel(r"Objective function value $f(\mathbf{x}_k)$", fontsize=10)
            
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, iteration)

    handles, labels_legend = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels_legend, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, borderaxespad=0.)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, 'figures'), exist_ok=True)
    path_fig = os.path.join(base_dir, f'figures/lambda-comparison-figure{file_number}-{m}-{n}.{"pdf" if pdf else "png"}')
    plt.savefig(path_fig, dpi=1000, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    run_experiment(file_number=1, trials=20, iteration=10000, m=100, n=100, pdf=True)
    run_experiment(file_number=2, trials=20, iteration=10000, m=50, n=100, pdf=True)
    run_experiment(file_number=3, trials=20, iteration=10000, m=100, n=50, pdf=True)