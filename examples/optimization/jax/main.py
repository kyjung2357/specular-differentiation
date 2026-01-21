import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import specular.jax as sjax
from specular.optimization.step_size import StepSize
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def ensure_length(data, length):
    data = list(data)
    if len(data) == 0: return [0.0] * length 
    if len(data) < length: return data + [data[-1]] * (length - len(data))
    else: return data[:length]
    
def format_sci_latex(x):
    if isinstance(x, str): return x
    if x == 0: return "0"
    s = "{:.4e}".format(x) 
    base, exponent = s.split('e')
    return fr"${base} \times 10^{{{int(exponent)}}}$"
# ==========================================
# 1. Objective function for only JAX
# ==========================================
def run_jax_experiment(methods, file_number,trials, iteration=100, m=500, n=100, lambda1=0.1, lambda2=1.0, pdf=False, show=False):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, λ1={lambda1}, λ2={lambda2}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}
    summary_stats = {}

    for trial in tqdm(range(trials), desc="Trials"):
        key = jax.random.PRNGKey(trial)
        k1, k2, k3 = jax.random.split(key, 3)
        
        A = jax.random.normal(k1, (m, n))
        b = jax.random.normal(k2, (m,))
        x_0 = jax.random.normal(k3, (n,))

        def f(x):
            residual = jnp.dot(A, x) - b
            loss = (1/(2*m)) * jnp.sum(residual**2)
            reg = (lambda2/2) * jnp.sum(x**2) + lambda1 * jnp.sum(jnp.abs(x))
            return loss + reg

        def f_np(x):
            x = np.atleast_1d(x)
            return (1/(2*m))*np.sum((A_np @ x - b_np)**2) + (lambda2/2)*np.sum(x**2) + lambda1*np.sum(np.abs(x))
    
        # Stochastic Component
        def f_j(x, idx):
            term_data = (jnp.dot(A[idx], x) - b[idx])**2
            term_reg = (lambda2/2) * jnp.sum(x**2) + lambda1 * jnp.sum(jnp.abs(x))
            return 0.5 * term_data + term_reg

        A_np = np.random.randn(m, n)
        b_np = np.random.randn(m)
        x_0_np = np.random.randn(n)

        A_torch = torch.tensor(A_np, dtype=torch.float32)
        b_torch = torch.tensor(b_np, dtype=torch.float32)

        def f_torch(x_tensor):
            residual = A_torch @ x_tensor - b_torch
            loss_term = (1/(2*m))*torch.sum(residual**2)
            l2_regularization = (lambda2/2)*torch.sum(x_tensor**2)    
            l1_regularization = lambda1*torch.sum(torch.abs(x_tensor)) 
            return loss_term + l2_regularization + l1_regularization
    
        step_size = StepSize('square_summable_not_summable', [4.0, 0.0])

        # ==== Specular gradient methods ====

        # SPEG
        if "SPEG" in methods:
            _, res, runtime = sjax.gradient_method(f, x_0, step_size, form='specular gradient', max_iter=iteration).history()
            all_results["SPEG"].append(ensure_length(res, iteration))
            running_times["SPEG"].append(runtime)

        # S-SPEG
        if "S-SPEG" in methods:
            _, res, runtime = sjax.gradient_method(f, x_0, step_size, form='stochastic', f_j=f_j, m=m, max_iter=iteration, seed=trial).history()
            all_results["S-SPEG"].append(ensure_length(res, iteration))
            running_times["S-SPEG"].append(runtime)

        # H-SPEG
        if "H-SPEG" in methods:
            _, res, runtime = sjax.gradient_method(f, x_0, step_size, form='hybrid', f_j=f_j, m=m, switch_iter=10, max_iter=iteration, seed=trial).history()
            all_results["H-SPEG"].append(ensure_length(res, iteration))
            running_times["H-SPEG"].append(runtime)

        # ==== Classical Methods ====
        if "GD" in methods:
            constant_step_size = StepSize(name='constant', parameters=0.001)
            _, res, runtime = gradient_descent_method(
                f_torch=f_torch, x_0=x_0_np, step_size=constant_step_size, max_iter=iteration
            ).history()
            all_results["GD"].append(ensure_length(res, iteration))
            running_times["GD"].append(runtime)

        # Adam
        if "Adam" in methods:
            _, res, runtime = Adam(
                f_torch=f_torch, x_0=x_0_np, step_size=0.01, max_iter=iteration
            ).history()
            all_results["Adam"].append(ensure_length(res, iteration))
            running_times["Adam"].append(runtime)

        # BFGS
        if "BFGS" in methods:
            _, res, runtime = BFGS(
                f_np=f_np, x_0=x_0_np, max_iter=iteration, tol=1e-6
            ).history()
            all_results["BFGS"].append(ensure_length(res, iteration))
            running_times["BFGS"].append(runtime)
    # ==========================================
    # 2. Results
    # ==========================================
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

    with open(path_txt, "w", encoding="utf-8") as table_file:
        table_file.write(display_df.to_latex(escape=False))

    print(summary_df)

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(\mathbf{x}_k)$", fontsize=10)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, iteration)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

    plt.tight_layout() 
    plt.savefig(path_fig, dpi=1000, bbox_inches='tight')

    if show:
        plt.show()


if __name__ == "__main__":
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]

    run_jax_experiment(methods, file_number=5, trials=20, iteration=10000, m=500, n=100, lambda1=100.0, lambda2=1.0, pdf=False, show=False)

    run_jax_experiment(methods, file_number=6, trials=20, iteration=10000, m=500, n=100, lambda1=0.0, lambda2=0.0, pdf=False, show=False)