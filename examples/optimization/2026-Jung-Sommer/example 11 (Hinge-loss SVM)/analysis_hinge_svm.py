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


def make_correlated_design(rng, samples, features, correlation=0.85):
    index = np.arange(features)
    covariance = correlation ** np.abs(index[:, None] - index[None, :])
    cholesky = np.linalg.cholesky(covariance + 1e-10 * np.eye(features))
    return rng.standard_normal((samples, features)) @ cholesky.T


def run_single_trial(args):
    (
        trial_idx,
        samples,
        features,
        c_hinge,
        lambda2,
        iteration,
        methods,
        trials,
    ) = args

    rng = np.random.default_rng(trial_idx)
    np.random.seed(trial_idx)
    torch.manual_seed(trial_idx)
    torch.set_num_threads(1)

    X_np = make_correlated_design(rng, samples, features)
    true_w = rng.standard_normal(features)
    true_w /= max(np.linalg.norm(true_w), 1e-12)
    scores = X_np @ true_w + 0.25 * rng.standard_normal(samples)
    y_np = np.where(scores >= np.median(scores), 1.0, -1.0)
    X_np = X_np + 0.25 * y_np[:, None] * true_w
    x_0 = rng.normal(scale=0.1, size=features + 1)

    X_torch = torch.tensor(X_np, dtype=torch.float32)
    y_torch = torch.tensor(y_np, dtype=torch.float32)

    def split(theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return theta[:-1], theta[-1]

    def f(theta):
        w, bias = split(theta)
        margins = y_np * (X_np @ w + bias)
        hinge = np.maximum(0.0, 1.0 - margins)
        return c_hinge * np.mean(hinge) + 0.5 * lambda2 * np.dot(w, w)

    def f_torch(theta):
        w = theta[:-1]
        bias = theta[-1]
        margins = y_torch * (X_torch @ w + bias)
        hinge = torch.relu(1.0 - margins)
        return c_hinge * torch.mean(hinge) + 0.5 * lambda2 * torch.sum(w**2)

    trial_results = {}
    trial_times = {}
    trial_label = f"{trial_idx + 1:0{len(str(trials))}d}"

    step_size_squ = specular.StepSchedule(
        name="square_summable_not_summable",
        parameters=[1.0, 0.0],
    )

    if "SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            tol=1e-12,
            max_iter=iteration,
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
                    line_search=rule,
                    print_bar=False,
                ).history()
            except Exception as exc:
                print(f"[Trial {trial_label}] {method} failed: {exc}", flush=True)
                continue

            trial_results[method] = ensure_length(res, iteration)
            trial_times[method] = runtime

    if "GD" in methods:
        constant_step_size = specular.StepSchedule(name="constant", parameters=0.01)
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
    c_hinge,
    lambda2,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment Start] Number: {file_number}")
    print("Problem: hinge-loss SVM")
    print(f"Settings: samples={samples}, features={features}, C={c_hinge}, lambda_2={lambda2}")
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
        (i, samples, features, c_hinge, lambda2, iteration, methods, trials)
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

    suffix = f"{samples}-{features}-{c_hinge}-{lambda2}"
    report_results(
        all_results,
        running_times,
        file_number,
        "Hinge-loss SVM",
        suffix,
        trials,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
