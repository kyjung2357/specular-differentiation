import sys
import os
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import specular.jax as sjax
from specular.optimization.step_size import StepSize
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method
from tools import *

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ==========================================
# 1. Objective function for only JAX
# ==========================================
def run_jax_experiment(methods, file_number,trials, iteration=100, m=500, n=100, lambda1=0.1, lambda2=1.0, pdf=False, show=False):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, λ1={lambda1}, λ2={lambda2}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}

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

    report_results(all_results, running_times, file_number, m, n, lambda1, lambda2, iteration, current_dir, pdf=pdf, show=show)

if __name__ == "__main__":
    methods = ["SPEG", "S-SPEG", "H-SPEG", "GD", "Adam", "BFGS"]

    run_jax_experiment(methods, file_number=5, trials=20, iteration=10000, m=500, n=100, lambda1=100.0, lambda2=1.0, pdf=False, show=False)

    run_jax_experiment(methods, file_number=6, trials=20, iteration=10000, m=500, n=100, lambda1=0.0, lambda2=0.0, pdf=False, show=False)