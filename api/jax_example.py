import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

import specular.jax as sjax
from specular.optimization.step_size import StepSize

jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ==========================================
# 1. Objective function for only JAX
# ==========================================
def run_jax_experiment(trials=10, iteration=100, m=500, n=100, lambda1=0.1, lambda2=1.0):
    
    methods = ["SPEG", "S-SPEG", "H-SPEG"]
    all_results = {m: [] for m in methods}
    all_times = {m: [] for m in methods}

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

        # Stochastic Component
        def f_j(x, idx):
            term_data = (jnp.dot(A[idx], x) - b[idx])**2
            term_reg = (lambda2/2) * jnp.sum(x**2) + lambda1 * jnp.sum(jnp.abs(x))
            return 0.5 * term_data + term_reg

        step_size = StepSize('square_summable_not_summable', [4.0, 0.0])

        # --- 1. SPEG (Deterministic) ---
        res = sjax.gradient_method(f, x_0, step_size, form='specular gradient', max_iter=iteration)
        _, hist_f, runtime = res.history()
        all_results["SPEG"].append(np.array(hist_f))
        all_times["SPEG"].append(runtime)

        # --- 2. S-SPEG (Stochastic) ---
        res_s = sjax.gradient_method(f, x_0, step_size, form='stochastic', f_j=f_j, m=m, max_iter=iteration, seed=trial)
        _, hist_f_s, runtime_s = res_s.history()
        all_results["S-SPEG"].append(np.array(hist_f_s))
        all_times["S-SPEG"].append(runtime_s)

        # --- 3. H-SPEG (Hybrid) ---
        res_h = sjax.gradient_method(f, x_0, step_size, form='hybrid', f_j=f_j, m=m, switch_iter=10, max_iter=iteration, seed=trial)
        _, hist_f_h, runtime_h = res_h.history()
        all_results["H-SPEG"].append(np.array(hist_f_h))
        all_times["H-SPEG"].append(runtime_h)

    # ==========================================
    # 2. Results
    # ==========================================
    print("\n[Avg Running Time]")
    for name in methods:
        print(f"{name}: {np.mean(all_times[name]):.4f}s")

    plt.figure(figsize=(8, 5))
    for name in methods:
        data = np.array(all_results[name])
        mean_curve = np.mean(data, axis=0)
        std_curve = np.std(data, axis=0)
        
        x_axis = np.arange(len(mean_curve))
        plt.plot(x_axis, mean_curve, label=name)
        plt.fill_between(x_axis, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)

    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.xscale('log')
    plt.ylabel('Objective Value')
    plt.title(f'JAX Backend Optimization (m={m}, n={n})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    run_jax_experiment(trials=20, iteration=1000, m=500, n=100, lambda1=100.0, lambda2=1.0)