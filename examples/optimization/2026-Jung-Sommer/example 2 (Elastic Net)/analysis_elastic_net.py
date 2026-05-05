import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

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


def run_single_trial(args):
    (
        trial_idx,
        m,
        n,
        lambda1,
        lambda2,
        iteration,
        methods,
        trials,
    ) = args

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
        residual = A_np @ x - b_np
        loss_term = (1 / (2 * m)) * np.sum(residual**2)
        l2_regularization = (lambda2 / 2) * np.sum(x**2)
        l1_regularization = lambda1 * np.sum(np.abs(x))
        return loss_term + l2_regularization + l1_regularization

    def f_torch(x_tensor):
        residual = A_torch @ x_tensor - b_torch
        loss_term = (1 / (2 * m)) * torch.sum(residual**2)
        l2_regularization = (lambda2 / 2) * torch.sum(x_tensor**2)
        l1_regularization = lambda1 * torch.sum(torch.abs(x_tensor))
        return loss_term + l2_regularization + l1_regularization

    def f_stochastic(x, j=False):
        x = np.asarray(x)
        if j is False:
            return f(x)

        if x.ndim == 1:
            term_data = (np.dot(A_np[j], x) - b_np[j]) ** 2
            term_reg2 = np.sum(x**2)
            term_reg1 = np.sum(np.abs(x))
        else:
            term_data = (x @ A_np[j] - b_np[j]) ** 2
            term_reg2 = np.sum(x**2, axis=1)
            term_reg1 = np.sum(np.abs(x), axis=1)

        return 0.5 * term_data + (lambda2 / 2) * term_reg2 + lambda1 * term_reg1

    trial_results = {}
    trial_times = {}
    trial_label = f"{trial_idx + 1:0{len(str(trials))}d}"

    step_size_squ = specular.StepSchedule(
        name="square_summable_not_summable",
        parameters=[4.0, 0.0],
    )
    step_size_geo = specular.StepSchedule(
        name="geometric_series",
        parameters=[1.0, 0.5],
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

    if "SPEG-s" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            tol=1e-12,
            max_iter=iteration,
            print_bar=False,
        ).history()
        trial_results["SPEG-s"] = ensure_length(res, iteration)
        trial_times["SPEG-s"] = runtime

    if "SPEG-g" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_geo,
            tol=1e-12,
            max_iter=iteration,
            print_bar=False,
        ).history()
        trial_results["SPEG-g"] = ensure_length(res, iteration)
        trial_times["SPEG-g"] = runtime

    if "S-SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            form="stochastic",
            tol=1e-12,
            max_iter=iteration,
            f_j=f_stochastic,
            m=m,
            print_bar=False,
        ).history()
        trial_results["S-SPEG"] = ensure_length(res, iteration)
        trial_times["S-SPEG"] = runtime

    if "H-SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            form="hybrid",
            tol=1e-12,
            max_iter=iteration,
            f_j=f_stochastic,
            m=m,
            switch_iter=10,
            print_bar=False,
        ).history()
        trial_results["H-SPEG"] = ensure_length(res, iteration)
        trial_times["H-SPEG"] = runtime

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
            except Exception as e:
                print(f"[Trial {trial_label}] {method} failed: {e}", flush=True)
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
            except Exception as e:
                print(f"[Trial {trial_label}] {method} failed: {e}", flush=True)
                continue

            trial_results[method] = ensure_length(res, iteration)
            trial_times[method] = runtime

    if "GD" in methods:
        constant_step_size = specular.StepSchedule(name="constant", parameters=0.001)
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
    m,
    n,
    lambda1,
    lambda2,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, lambda_1={lambda1}, lambda_2={lambda2}")
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
        (i, m, n, lambda1, lambda2, iteration, methods, trials)
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

            except Exception as e:
                import traceback

                print(f"\n[Error] Trial failed: {e}")
                traceback.print_exc()

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
        trials,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
