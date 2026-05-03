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
REPO_ROOT = CURRENT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import specular
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method
from tools import ensure_length, report_results

specular.change_backend("cpu_numpy")


def abs_power_np(x, exponent):
    return np.abs(x) ** exponent


def abs_power_torch(x_tensor, exponent):
    import torch

    return torch.abs(x_tensor) ** exponent


def nesterov_objective_np(x, p, q):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    term1 = 0.25 * abs_power_np(x[0] - 1.0, p)
    residual = x[1:] - 2.0 * abs_power_np(x[:-1], p) + 1.0
    return float(term1 + np.sum(abs_power_np(residual, q)))


def nesterov_objective_torch(x_tensor, p, q):
    import torch

    term1 = 0.25 * abs_power_torch(x_tensor[0] - 1.0, p)
    residual = x_tensor[1:] - 2.0 * abs_power_torch(x_tensor[:-1], p) + 1.0
    return term1 + torch.sum(abs_power_torch(residual, q))


def nesterov_component_np(x, j, p, q):
    x = np.atleast_1d(np.asarray(x, dtype=float))

    if j == 0:
        return float(0.25 * abs_power_np(x[0] - 1.0, p))

    residual = x[j] - 2.0 * abs_power_np(x[j - 1], p) + 1.0
    return float(abs_power_np(residual, q))


def run_single_trial(args):
    (
        trial_idx,
        n,
        p,
        q,
        iteration,
        methods,
        line_search,
        safeguard,
    ) = args

    np.random.seed(trial_idx)
    needs_torch = any(method in methods for method in ("GD", "Adam"))
    if needs_torch:
        import torch

        torch.manual_seed(trial_idx)
        torch.set_num_threads(1)

    x_0 = np.random.uniform(low=-2.0, high=2.0, size=n)

    def f(x):
        return nesterov_objective_np(x, p, q)

    def f_torch(x_tensor):
        return nesterov_objective_torch(x_tensor, p, q)

    def f_stochastic(x, j=False):
        if j is False:
            return f(x)
        return nesterov_component_np(x, j, p, q)

    trial_results = {}
    trial_times = {}

    step_size_squ = specular.StepSize(
        name="square_summable_not_summable",
        parameters=[4.0, 0.0],
    )
    step_size_geo = specular.StepSize(
        name="geometric_series",
        parameters=[1.0, 0.5],
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

    if "SPEG-s" in methods:
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            tol=1e-10,
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
            tol=1e-10,
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
            tol=1e-10,
            max_iter=iteration,
            f_j=f_stochastic,
            m=n,
            print_bar=False,
        ).history()
        trial_results["S-SPEG"] = ensure_length(res, iteration)
        trial_times["S-SPEG"] = runtime

    if "H-SPEG" in methods:
        switch_iter = min(10, iteration)
        _, res, runtime = specular.gradient_method(
            f=f,
            x_0=x_0,
            step_size=step_size_squ,
            form="hybrid",
            tol=1e-10,
            max_iter=iteration,
            f_j=f_stochastic,
            m=n,
            switch_iter=switch_iter,
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
    file_number,
    trials,
    iteration,
    n,
    p,
    q,
    label=None,
    line_search="armijo",
    safeguard=1e-10,
    pdf=False,
    show=False,
):
    function_label = label or f"Nesterov_N{p:g}{q:g}"
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Function: N_{{{p:g},{q:g}}} ({function_label})")
    print(f"Settings: n={n}, trials={trials}, iterations={iteration}")
    print(f"S-BFGS settings: line_search={line_search}, safeguard={safeguard}")

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}

    tasks = [
        (i, n, p, q, iteration, methods, line_search, safeguard)
        for i in range(trials)
    ]

    num_workers = min(os.cpu_count() or 1, trials)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_single_trial, task) for task in tasks]

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
        n,
        p,
        q,
        function_label,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
