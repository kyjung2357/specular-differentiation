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


def run_single_trial(args):
    trial_idx, n, c_dis, c_chg, iteration, methods, line_search, safeguard = args

    np.random.seed(trial_idx)
    needs_torch = any(method in methods for method in ("GD", "Adam"))
    if needs_torch:
        import torch

        torch.manual_seed(trial_idx)
        torch.set_num_threads(1)

    C_grid = np.random.uniform(1.0, 3.0, n)
    P_load = np.random.uniform(10.0, 50.0, n)
    x_0 = np.zeros(n)

    if needs_torch:
        C_grid_torch = torch.tensor(C_grid, dtype=torch.float32)
        P_load_torch = torch.tensor(P_load, dtype=torch.float32)

    def f(p):
        p = np.atleast_1d(p)
        grid_cost = np.sum(C_grid * (P_load - p) ** 2)
        dis_cost = c_dis * np.sum(np.maximum(0.0, p))
        chg_cost = c_chg * np.sum(np.maximum(0.0, -p))
        return float(grid_cost + dis_cost + chg_cost)

    def f_torch(p_tensor):
        import torch

        grid_cost = torch.sum(C_grid_torch * (P_load_torch - p_tensor) ** 2)
        dis_cost = c_dis * torch.sum(torch.clamp(p_tensor, min=0.0))
        chg_cost = c_chg * torch.sum(torch.clamp(-p_tensor, min=0.0))
        return grid_cost + dis_cost + chg_cost

    def f_stochastic(p, j=False):
        p = np.atleast_1d(p)

        if j is False:
            return f(p)

        grid_cost = C_grid[j] * (P_load[j] - p[j]) ** 2
        dis_cost = c_dis * np.maximum(0.0, p[j])
        chg_cost = c_chg * np.maximum(0.0, -p[j])
        return float(grid_cost + dis_cost + chg_cost)

    trial_results = {}
    trial_times = {}

    step_size_squ = specular.StepSize(
        name="square_summable_not_summable",
        parameters=[4.0, 0.0],
    )

    if "SPEG" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            tol=1e-10,
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
            tol=1e-10,
            max_iter=iteration,
            f_j=f_stochastic,
            m=n,
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
            tol=1e-10,
            max_iter=iteration,
            f_j=f_stochastic,
            m=n,
            switch_iter=min(10, iteration),
            print_bar=False,
        ).history()
        trial_results["H-SPEG"] = ensure_length(res, iteration)
        trial_times["H-SPEG"] = runtime

    if "S-BFGS" in methods:
        _, res, runtime = specular.BFGS_method(
            f=f,
            x_0=x_0,
            tol=1e-10,
            max_iter=iteration,
            line_search=line_search,
            safeguard=safeguard,
            print_bar=False,
        ).history()
        trial_results["S-BFGS"] = ensure_length(res, iteration)
        trial_times["S-BFGS"] = runtime

    if "GD" in methods:
        constant_step_size = specular.StepSize(name="constant", parameters=0.001)
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

    if "BFGS" in methods:
        _, res, runtime = BFGS(
            f_np=f,
            x_0=x_0,
            max_iter=iteration,
            tol=1e-6,
        ).history()
        trial_results["BFGS"] = ensure_length(res, iteration)
        trial_times["BFGS"] = runtime

    return trial_results, trial_times


def run_experiment(
    methods,
    trials=20,
    iteration=1000,
    n=24,
    c_dis=500.0,
    c_chg=200.0,
    line_search="armijo",
    safeguard=1e-10,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment 7] Microgrid: n={n}, c_dis={c_dis}, c_chg={c_chg}")
    print(f"S-BFGS settings: line_search={line_search}, safeguard={safeguard}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}

    tasks = [
        (i, n, c_dis, c_chg, iteration, methods, line_search, safeguard)
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
        "Microgrid",
        n,
        c_dis,
        c_chg,
        "NA",
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
        title="Microgrid dispatch",
    )
