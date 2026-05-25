import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

CURRENT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CURRENT_DIR.parents[0]
REPO_ROOT = CURRENT_DIR.parents[3]

for path in (REPO_ROOT, CURRENT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
    trial_idx, iteration, methods, trials = args

    np.random.seed(trial_idx)
    needs_torch = any(method in methods for method in ("GD", "Adam"))
    if needs_torch:
        import torch

        torch.manual_seed(trial_idx)
        torch.set_num_threads(1)

    def f(x):
        x = np.atleast_1d(x)
        term1 = 2.0 * np.abs(x[0]) + x[1]
        term2 = 3.0 * x[1]
        return float(np.maximum(term1, term2))

    def f_torch(x_tensor):
        import torch

        term1 = 2.0 * torch.abs(x_tensor[0]) + x_tensor[1]
        term2 = 3.0 * x_tensor[1]
        return torch.maximum(term1, term2)

    def f_stochastic(x, j=False):
        x = np.atleast_1d(x)

        if j is False:
            return f(x)
        if j == 0:
            return float(2.0 * np.abs(x[0]) + x[1])
        return float(3.0 * x[1])

    x_0 = np.random.uniform(low=-5.0, high=5.0, size=2)

    trial_results = {}
    trial_times = {}
    trial_label = f"{trial_idx + 1:0{len(str(trials))}d}"

    step_size_squ = specular.StepSchedule(
        name="square_summable_not_summable",
        parameters=[4.0, 0.0],
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

    if "S-SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            form="stochastic",
            tol=1e-12,
            max_iter=iteration,
            fill_iteration=True,
            f_j=f_stochastic,
            m=2,
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
            fill_iteration=True,
            f_j=f_stochastic,
            m=2,
            switch_iter=min(10, iteration),
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
                    fill_iteration=True,
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
                    fill_iteration=True,
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
    trials=20,
    iteration=200,
    pdf=False,
    show=False,
):
    print("\n[Experiment 8] Lewis-Overton Polyhedral Failure Example")
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
        (i, iteration, methods, trials)
        for i in range(trials)
    ]

    num_workers = min(os.cpu_count() or 1, trials)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_trial, task) for task in tasks]

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

    report_results(
        all_results,
        running_times,
        "Polyhedral",
        2,
        2,
        "NA",
        "NA",
        trials,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
        yscale="symlog",
        title="Lewis-Overton polyhedral example",
    )
