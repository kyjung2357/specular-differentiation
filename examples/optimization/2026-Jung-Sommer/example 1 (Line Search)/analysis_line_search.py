import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import specular
from tools import ensure_length, report_results

specular.change_backend("cpu_numpy")


LINE_SEARCH_RULES = {
    "S-BFGS-E": "exact",
    "S-BFGS-S": "strong_wolfe",
    "S-BFGS-W": "wolfe",
    "S-BFGS-A": "armijo",
}


def run_single_trial(args):
    (
        trial_idx,
        objective_name,
        m,
        n,
        lambda1,
        lambda2,
        iteration,
        methods,
        t_0,
        c_1,
        c_2,
        rho,
        max_line_iter,
    ) = args

    np.random.seed(trial_idx)

    rng = np.random.default_rng(trial_idx)
    x_0 = rng.normal(size=n)

    if objective_name == "elastic_net":
        A_np = rng.normal(size=(m, n))
        b_np = rng.normal(size=m)

        def f(x):
            x = np.atleast_1d(x)
            residual = A_np @ x - b_np
            loss_term = (1 / (2 * m)) * np.sum(residual**2)
            l2_regularization = (lambda2 / 2) * np.sum(x**2)
            l1_regularization = lambda1 * np.sum(np.abs(x))
            return float(1e-8 + loss_term + l2_regularization + l1_regularization)

    elif objective_name == "polyhedral_max":
        A_np = rng.normal(size=(m, n))
        A_np /= np.maximum(np.linalg.norm(A_np, axis=1, keepdims=True), 1e-12)
        b_np = rng.uniform(-0.5, 0.5, size=m)

        def f(x):
            x = np.atleast_1d(x)
            return float(
                1e-8
                + 0.05 * np.sum(x**2)
                + np.maximum(0.0, np.max(A_np @ x - b_np))
            )

    elif objective_name == "hinge_quadratic":
        A_np = rng.normal(size=(m, n))
        A_np /= np.maximum(np.linalg.norm(A_np, axis=1, keepdims=True), 1e-12)
        b_np = rng.uniform(-0.5, 0.5, size=m)

        def f(x):
            x = np.atleast_1d(x)
            hinge = np.maximum(0.0, A_np @ x - b_np)
            return float(1e-8 + (lambda2 / 2) * np.sum(x**2) + lambda1 * np.mean(hinge))

    else:
        raise ValueError(f"Unknown objective_name: {objective_name}")

    trial_results = {}
    trial_times = {}
    failures = {}

    for method in methods:
        line_search_name = LINE_SEARCH_RULES[method]
        line_search = specular.LineSearch(
            name=line_search_name,
            t_0=t_0,
            c_1=c_1,
            c_2=c_2,
            rho=rho,
            max_iter=max_line_iter,
            raise_on_fail=True,
        )

        try:
            _, res, runtime = specular.BFGS_method(
                f=f,
                x_0=x_0,
                tol=1e-12,
                max_iter=iteration,
                fill_iteration=True,
                line_search=line_search,
                print_bar=False,
            ).history()
            trial_results[method] = ensure_length(res, iteration)
            trial_times[method] = runtime
        except Exception as exc:
            failures[method] = str(exc)

    return trial_results, trial_times, failures


def run_experiment(
    objective_name,
    methods,
    file_number,
    trials,
    iteration,
    m,
    n,
    lambda1,
    lambda2,
    t_0=1.0,
    c_1=1e-4,
    c_2=0.9,
    rho=0.5,
    max_line_iter=20,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Objective: {objective_name}")
    print(f"Settings: m={m}, n={n}, lambda_1={lambda1}, lambda_2={lambda2}")
    print(
        "Line search settings: "
        f"t_0={t_0}, c_1={c_1}, c_2={c_2}, rho={rho}, "
        f"max_line_iter={max_line_iter}"
    )

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}
    failure_counts = {method: 0 for method in methods}
    failure_examples = {method: [] for method in methods}

    tasks = [
        (
            i,
            objective_name,
            m,
            n,
            lambda1,
            lambda2,
            iteration,
            methods,
            t_0,
            c_1,
            c_2,
            rho,
            max_line_iter,
        )
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
                t_res, t_time, t_failures = future.result()

                for method in methods:
                    if method in t_res:
                        all_results[method].append(t_res[method])

                    if method in t_time:
                        running_times[method].append(t_time[method])

                    if method in t_failures:
                        failure_counts[method] += 1
                        if len(failure_examples[method]) < 3:
                            failure_examples[method].append(t_failures[method])

            except Exception as e:
                import traceback

                print(f"\n[Error] Trial failed: {e}")
                traceback.print_exc()

    print("\n[Failure Examples]")
    for method, examples in failure_examples.items():
        if examples:
            print(f"{method}:")
            for example in examples:
                print(f"  - {example}")

    print("\n[Analysis]")
    print(" Generating plots and tables")

    report_results(
        all_results,
        running_times,
        failure_counts,
        file_number,
        m,
        n,
        lambda1,
        lambda2,
        iteration,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
