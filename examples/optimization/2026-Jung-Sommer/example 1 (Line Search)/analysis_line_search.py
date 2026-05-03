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
    "S-BFGS-A": "armijo",
    "S-BFGS-W": "wolfe",
    "S-BFGS-S": "strong_wolfe",
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
        alpha_0,
        c_1,
        c_2,
        rho,
        max_line_iter,
        safeguard,
    ) = args

    np.random.seed(trial_idx)

    A_np = np.random.randn(m, n)
    b_np = np.random.randn(m)
    x_0 = np.random.randn(n)

    def f(x):
        x = np.atleast_1d(x)
        residual = A_np @ x - b_np
        loss_term = (1 / (2 * m)) * np.sum(residual**2)
        l2_regularization = (lambda2 / 2) * np.sum(x**2)
        l1_regularization = lambda1 * np.sum(np.abs(x))
        return loss_term + l2_regularization + l1_regularization

    trial_results = {}
    trial_times = {}
    failures = {}

    for method in methods:
        line_search_name = LINE_SEARCH_RULES[method]
        line_search = specular.LineSearch(
            name=line_search_name,
            alpha_0=alpha_0,
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
                tol=1e-10,
                max_iter=iteration,
                line_search=line_search,
                safeguard=safeguard,
                print_bar=False,
            ).history()
            trial_results[method] = ensure_length(res, iteration)
            trial_times[method] = runtime
        except Exception as exc:
            failures[method] = str(exc)

    return trial_results, trial_times, failures


def run_experiment(
    methods,
    file_number,
    trials,
    iteration,
    m,
    n,
    lambda1,
    lambda2,
    alpha_0=1.0,
    c_1=1e-4,
    c_2=0.9,
    rho=0.5,
    max_line_iter=20,
    safeguard=1e-10,
    pdf=False,
    show=False,
):
    print(f"\n[Experiment Start] Number: {file_number}")
    print(f"Settings: m={m}, n={n}, lambda_1={lambda1}, lambda_2={lambda2}")
    print(
        "Line search settings: "
        f"alpha_0={alpha_0}, c_1={c_1}, c_2={c_2}, rho={rho}, "
        f"max_line_iter={max_line_iter}, safeguard={safeguard}"
    )

    all_results = {method: [] for method in methods}
    running_times = {method: [] for method in methods}
    failure_counts = {method: 0 for method in methods}
    failure_examples = {method: [] for method in methods}

    tasks = [
        (
            i,
            m,
            n,
            lambda1,
            lambda2,
            iteration,
            methods,
            alpha_0,
            c_1,
            c_2,
            rho,
            max_line_iter,
            safeguard,
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
