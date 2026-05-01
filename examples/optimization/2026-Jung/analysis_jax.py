import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.dirname(current_dir)
repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

for path in (repo_root, tools_dir):
    if path not in sys.path:
        sys.path.insert(0, path)

import specular
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method
from tools import ensure_length, report_results

specular.change_backend("cpu_jax")


def run_experiment(
    methods,
    file_number,
    trials,
    iteration,
    m,
    n,
    lambda1,
    lambda2,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, lambda1={lambda1}, lambda2={lambda2}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}

    step_size_squ = specular.StepSize(
        name="square_summable_not_summable",
        parameters=[4.0, 0.0],
    )
    step_size_geo = specular.StepSize(
        name="geometric_series",
        parameters=[1.0, 0.5],
    )

    for trial in tqdm(range(trials), desc="Trials"):
        np.random.seed(trial)
        torch.manual_seed(trial)
        torch.set_num_threads(1)

        A_np = np.random.randn(m, n)
        b_np = np.random.randn(m)
        x_0_np = np.random.randn(n)

        A = jnp.asarray(A_np, dtype=float)
        b = jnp.asarray(b_np, dtype=float)
        x_0 = jnp.asarray(x_0_np, dtype=float)

        A_torch = torch.tensor(A_np, dtype=torch.float32)
        b_torch = torch.tensor(b_np, dtype=torch.float32)

        def f(x):
            residual = A @ x - b
            return (
                (1.0 / (2.0 * m)) * jnp.sum(residual**2)
                + (lambda2 / 2.0) * jnp.sum(x**2)
                + lambda1 * jnp.sum(jnp.abs(x))
            )

        def f_j(x, idx):
            residual = jnp.dot(A[idx], x) - b[idx]
            return (
                0.5 * residual**2
                + (lambda2 / 2.0) * jnp.sum(x**2)
                + lambda1 * jnp.sum(jnp.abs(x))
            )

        def f_np(x):
            x = np.atleast_1d(x)
            return (
                (1.0 / (2.0 * m)) * np.sum((A_np @ x - b_np) ** 2)
                + (lambda2 / 2.0) * np.sum(x**2)
                + lambda1 * np.sum(np.abs(x))
            )

        def f_torch(x_tensor):
            residual = A_torch @ x_tensor - b_torch
            return (
                (1.0 / (2.0 * m)) * torch.sum(residual**2)
                + (lambda2 / 2.0) * torch.sum(x_tensor**2)
                + lambda1 * torch.sum(torch.abs(x_tensor))
            )

        if "SPEG" in methods:
            _, res, runtime = specular.gradient_method(
                f=f,
                x_0=x_0,
                step_size=step_size_squ,
                form="specular gradient",
                tol=1e-10,
                max_iter=iteration,
                print_bar=False,
            ).history()
            all_results["SPEG"].append(ensure_length(res, iteration))
            running_times["SPEG"].append(runtime)

        if "SPEG-s" in methods:
            _, res, runtime = specular.gradient_method(
                f=f,
                x_0=x_0,
                step_size=step_size_squ,
                form="specular gradient",
                tol=1e-10,
                max_iter=iteration,
                print_bar=False,
            ).history()
            all_results["SPEG-s"].append(ensure_length(res, iteration))
            running_times["SPEG-s"].append(runtime)

        if "SPEG-g" in methods:
            _, res, runtime = specular.gradient_method(
                f=f,
                x_0=x_0,
                step_size=step_size_geo,
                form="specular gradient",
                tol=1e-10,
                max_iter=iteration,
                print_bar=False,
            ).history()
            all_results["SPEG-g"].append(ensure_length(res, iteration))
            running_times["SPEG-g"].append(runtime)

        if "S-SPEG" in methods:
            _, res, runtime = specular.gradient_method(
                f=f,
                x_0=x_0,
                step_size=step_size_squ,
                form="stochastic",
                tol=1e-10,
                max_iter=iteration,
                f_j=f_j,
                m=m,
                print_bar=False,
            ).history()
            all_results["S-SPEG"].append(ensure_length(res, iteration))
            running_times["S-SPEG"].append(runtime)

        if "H-SPEG" in methods:
            _, res, runtime = specular.gradient_method(
                f=f,
                x_0=x_0,
                step_size=step_size_squ,
                form="hybrid",
                tol=1e-10,
                max_iter=iteration,
                f_j=f_j,
                m=m,
                switch_iter=10,
                print_bar=False,
            ).history()
            all_results["H-SPEG"].append(ensure_length(res, iteration))
            running_times["H-SPEG"].append(runtime)

        if "GD" in methods:
            constant_step_size = specular.StepSize(
                name="constant",
                parameters=0.001,
            )
            _, res, runtime = gradient_descent_method(
                f_torch=f_torch,
                x_0=x_0_np,
                step_size=constant_step_size,
                max_iter=iteration,
            ).history()
            all_results["GD"].append(ensure_length(res, iteration))
            running_times["GD"].append(runtime)

        if "Adam" in methods:
            _, res, runtime = Adam(
                f_torch=f_torch,
                x_0=x_0_np,
                step_size=0.01,
                max_iter=iteration,
            ).history()
            all_results["Adam"].append(ensure_length(res, iteration))
            running_times["Adam"].append(runtime)

        if "BFGS" in methods:
            _, res, runtime = BFGS(
                f_np=f_np,
                x_0=x_0_np,
                max_iter=iteration,
                tol=1e-6,
            ).history()
            all_results["BFGS"].append(ensure_length(res, iteration))
            running_times["BFGS"].append(runtime)

    print("\n[Analysis]")
    print(" Generating plots and tables")

    report_results(
        all_results,
        running_times,
        file_number,
        m,
        n,
        lambda1,
        lambda2,
        iteration,
        current_dir,
        pdf=pdf,
        show=show,
    )