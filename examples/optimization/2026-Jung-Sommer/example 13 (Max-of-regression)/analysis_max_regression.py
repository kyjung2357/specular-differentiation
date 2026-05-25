import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parents[3]

for path in (PACKAGE_ROOT, CURRENT_DIR):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

import specular
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method
from tools import ensure_length, report_results

specular.change_backend("cpu_numpy")


BFGS_LINE_SEARCH_RULES = {
    "BFGS-E": "exact",
    "BFGS-S": "strong_wolfe",
    "BFGS-W": "wolfe",
    "BFGS-A": "armijo",
}

S_BFGS_LINE_SEARCH_RULES = {
    "S-BFGS-E": "exact",
    "S-BFGS-S": "strong_wolfe",
    "S-BFGS-W": "wolfe",
    "S-BFGS-A": "armijo",
}

LINE_SEARCH_RULES = {
    **BFGS_LINE_SEARCH_RULES,
    **S_BFGS_LINE_SEARCH_RULES,
}


def make_correlated_design(rng, samples, features, correlation=0.9):
    index = np.arange(features)
    covariance = correlation ** np.abs(index[:, None] - index[None, :])
    cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(features))
    return rng.standard_normal((samples, features)) @ cholesky.T


def run_single_trial(args):
    (
        trial_idx,
        samples,
        features,
        lambda1,
        lambda2,
        outlier_scale,
        iteration,
        methods,
        trials,
    ) = args

    rng = np.random.default_rng(trial_idx)
    np.random.seed(trial_idx)
    torch.manual_seed(trial_idx)
    torch.set_num_threads(1)

    A_np = make_correlated_design(rng, samples, features)
    true_x = rng.standard_normal(features)
    mask = rng.random(features) < 0.35
    true_x = true_x * mask
    b_np = A_np @ true_x + 0.05 * rng.standard_normal(samples)
    outlier_count = max(1, samples // 10)
    outlier_index = rng.choice(samples, size=outlier_count, replace=False)
    b_np[outlier_index] += outlier_scale * rng.choice([-1.0, 1.0], size=outlier_count)
    x_0 = rng.normal(scale=0.5, size=features)

    A_torch = torch.tensor(A_np, dtype=torch.float32)
    b_torch = torch.tensor(b_np, dtype=torch.float32)

    def f(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        residual = A_np @ x - b_np
        max_residual = np.max(np.abs(residual))
        l1_regularization = lambda1 * np.sum(np.abs(x))
        l2_regularization = 0.5 * lambda2 * np.dot(x, x)
        return float(max_residual + l1_regularization + l2_regularization)

    def f_torch(x_tensor):
        residual = A_torch @ x_tensor - b_torch
        max_residual = torch.max(torch.abs(residual))
        l1_regularization = lambda1 * torch.sum(torch.abs(x_tensor))
        l2_regularization = 0.5 * lambda2 * torch.sum(x_tensor**2)
        return max_residual + l1_regularization + l2_regularization

    trial_results = {}
    trial_times = {}
    trial_label = f"{trial_idx + 1:0{len(str(trials))}d}"

    step_size_squ = specular.StepSchedule(
        name="square_summable_not_summable",
        parameters=[0.75, 0.0],
    )

    if "SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            tol=1e-12,
            max_iter=iteration,
            fill_iteration=True,
            print_bar=False,
        ).history()
        trial_results["SPEG"] = ensure_length(res, iteration)
        trial_times["SPEG"] = runtime

    for method, rule in BFGS_LINE_SEARCH_RULES.items():
        if method in methods:
            try:
                _, res, runtime = BFGS(
                    f_np=f,
                    x_0=x_0,
                    max_iter=iteration,
                    fill_iteration=True,
                    tol=1e-12,
                    line_search=rule,
                ).history()
            except Exception as exc:
                print(f"[Trial {trial_label}] {method} failed: {exc}", flush=True)
                continue

            trial_results[method] = ensure_length(res, iteration)
            trial_times[method] = runtime

    for method, rule in S_BFGS_LINE_SEARCH_RULES.items():
        if method in methods:
            try:
                _, res, runtime = specular.BFGS_method(
                    f=f,
                    x_0=x_0,
                    tol=1e-12,
                    max_iter=iteration,
                    fill_iteration=True,
                    line_search=rule,
                    print_bar=False,
                ).history()
            except Exception as exc:
                print(f"[Trial {trial_label}] {method} failed: {exc}", flush=True)
                continue

            trial_results[method] = ensure_length(res, iteration)
            trial_times[method] = runtime

    if "GD" in methods:
        constant_step_size = specular.StepSchedule(name="constant", parameters=0.005)
        _, res, runtime = gradient_descent_method(
            f_torch=f_torch,
            x_0=x_0,
            step_size=constant_step_size,
            max_iter=iteration,
        ).history()
        trial_results["GD"] = ensure_length(res, iteration)
        trial_times["GD"] = runtime

    if "Adam" in methods:
        _, res, runtime = Adam(
            f_torch=f_torch,
            x_0=x_0,
            step_size=0.01,
            max_iter=iteration,
        ).history()
        trial_results["Adam"] = ensure_length(res, iteration)
        trial_times["Adam"] = runtime

    return trial_results, trial_times


def run_experiment(
    methods,
    file_number,
    trials,
    iteration,
    samples,
    features,
    lambda1,
    lambda2,
    outlier_scale,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment Start] Number: {file_number}")
    print("Problem: max-of-regression")
    print(
        f"Settings: samples={samples}, features={features}, "
        f"lambda_1={lambda1}, lambda_2={lambda2}, outlier_scale={outlier_scale}"
    )
    active_bfgs_rules = {
        method: LINE_SEARCH_RULES[method]
        for method in methods
        if method in BFGS_LINE_SEARCH_RULES
    }
    active_s_bfgs_rules = {
        method: LINE_SEARCH_RULES[method]
        for method in methods
        if method in S_BFGS_LINE_SEARCH_RULES
    }
    print(f"BFGS settings: line_search={active_bfgs_rules}")
    print(f"S-BFGS settings: line_search={active_s_bfgs_rules}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}

    tasks = [
        (i, samples, features, lambda1, lambda2, outlier_scale, iteration, methods, trials)
        for i in range(trials)
    ]

    num_workers = min(os.cpu_count() or 1, trials)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_trial, task) for task in tasks]

        print()
        for future in tqdm(
            as_completed(futures),
            total=trials,
            desc="Processing Trials",
            leave=False,
        ):
            try:
                t_res, t_time = future.result()

                for method in methods:
                    if method in t_res:
                        all_results[method].append(t_res[method])

                    if method in t_time:
                        running_times[method].append(t_time[method])

            except Exception as exc:
                import traceback

                print(f"\n[Error] Trial failed: {exc}")
                traceback.print_exc()

    print("\n[Analysis]")
    print(" Generating plots and tables")

    suffix = f"{samples}-{features}-{lambda1}-{lambda2}-{outlier_scale}"
    report_results(
        all_results,
        running_times,
        file_number,
        "Max-of-regression",
        suffix,
        trials,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
