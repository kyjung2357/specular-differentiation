import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

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
    trial_idx, c_points, k, mu, lam, iteration, methods, trials = args

    np.random.seed(trial_idx)
    needs_torch = any(method in methods for method in ("GD", "Adam"))
    if needs_torch:
        torch.manual_seed(trial_idx)
        torch.set_num_threads(1)

    dim = 3 * c_points
    f_0 = np.random.randn(dim)
    N_mat = np.random.randn(dim, k)
    x_0 = np.zeros(k)

    if needs_torch:
        f_0_torch = torch.tensor(f_0, dtype=torch.float32)
        N_torch = torch.tensor(N_mat, dtype=torch.float32)

    def f(x):
        x = np.atleast_1d(x)
        force = f_0 + N_mat @ x
        max_normal = -np.inf
        penalty = 0.0
        for i in range(c_points):
            f_n = force[3 * i]
            f_t = force[3 * i + 1 : 3 * i + 3]
            max_normal = max(max_normal, f_n)
            penalty += lam * max(0.0, np.linalg.norm(f_t) - mu * f_n)
        return float(max_normal + penalty)

    def f_torch(x_tensor):
        import torch

        force = f_0_torch + N_torch @ x_tensor
        normal_forces = force[0::3]
        max_normal = torch.max(normal_forces)
        penalty = torch.tensor(0.0, dtype=x_tensor.dtype)
        for i in range(c_points):
            f_n = force[3 * i]
            f_t = force[3 * i + 1 : 3 * i + 3]
            penalty = penalty + lam * torch.clamp(torch.linalg.norm(f_t) - mu * f_n, min=0.0)
        return max_normal + penalty

    def f_stochastic(x, j=False):
        x = np.atleast_1d(x)
        force = f_0 + N_mat @ x

        if j is False:
            return f(x)
        if j == 0:
            return float(np.max(force[0::3]))

        contact_idx = j - 1
        f_n = force[3 * contact_idx]
        f_t = force[3 * contact_idx + 1 : 3 * contact_idx + 3]
        return float(lam * max(0.0, np.linalg.norm(f_t) - mu * f_n))

    trial_results = {}
    trial_times = {}
    component_count = c_points + 1
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
            f_j=f_stochastic,
            m=component_count,
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
            m=component_count,
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
    trials=20,
    iteration=1000,
    c_points=3,
    k=4,
    mu=0.5,
    lam=10000.0,
    pdf=False,
    show=False,
):
    print(
        f"\n[Experiment 6] Grasp Force: contacts={c_points}, "
        f"k={k}, mu={mu}, lambda={lam}"
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
        (i, c_points, k, mu, lam, iteration, methods, trials)
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
        "GraspForce",
        c_points,
        k,
        mu,
        lam,
        trials,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
        title="Grasp force optimization",
    )
