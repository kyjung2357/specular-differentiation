import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import specular
from specular.optimization.classical_solver import BFGS
from tools import ensure_length, plot_comparison, print_summary

specular.change_backend("cpu_numpy")


DEFAULT_METHODS = [
    "Dai sequence",
    "SPEG",
    "BFGS-E",
    "BFGS-S",
    "BFGS-W",
    "BFGS-A",
    "S-BFGS-E",
    "S-BFGS-S",
    "S-BFGS-W",
    "S-BFGS-A",
]

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

LINE_SEARCH_OPTIONS = {
    "alpha_0": 1.0,
    "c_1": 1e-4,
    "c_2": 0.9,
    "rho": 0.5,
    "max_line_iter": 60,
    "max_alpha": 20.0,
    "raise_on_fail": False,
}


SQRT2 = np.sqrt(2.0)
THETA_1 = 0.25 * np.pi
THETA_2 = 0.75 * np.pi
DECAY_T = (3.0 * SQRT2 - 1.0 - np.sqrt(31.0 - 20.0 * SQRT2)) / 2.0


def rotation(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )


R1 = rotation(THETA_1)
R2 = rotation(THETA_2)
M_MATRIX = np.block(
    [
        [R1, np.zeros((2, 2))],
        [np.zeros((2, 2)), DECAY_T * R2],
    ]
)


def initial_step():
    gamma_1 = (17.0 - 8.0 * SQRT2) * DECAY_T + (-17.0 + 9.0 * SQRT2)
    tau_1 = (-17.0 + 9.0 * SQRT2) * DECAY_T + (17.0 - 9.0 * SQRT2)
    return np.array([SQRT2, 0.0, gamma_1, tau_1], dtype=float)


def initial_point():
    a_1 = -0.5 * SQRT2
    b_1 = -1.0 - 0.5 * SQRT2
    p_1 = ((-15720.0 + 4019.0 * SQRT2) * DECAY_T + (32931.0 - 17534.0 * SQRT2)) / 4633.0
    q_1 = ((4376.0 - 1977.0 * SQRT2) * DECAY_T + (9563.0 - 9433.0 * SQRT2)) / 4633.0
    return np.array([a_1, b_1, p_1, q_1], dtype=float)


def initial_inverse_hessian():
    h_bar = np.zeros((4, 4), dtype=float)
    h_bar[0, 0] = (-3690.0 - 13280.0 * SQRT2) * DECAY_T + (79982.0 - 13694.0 * SQRT2)
    h_bar[0, 1] = (4474.0 + 1590.0 * SQRT2) * DECAY_T - (1990.0 + 11308.0 * SQRT2)
    h_bar[0, 2] = (1428.0 + 18496.0 * SQRT2) * DECAY_T - (33966.0 - 11118.0 * SQRT2)
    h_bar[0, 3] = (-23256.0 + 952.0 * SQRT2) * DECAY_T + (25092.0 - 2108.0 * SQRT2)
    h_bar[1, 1] = (-1954.0 - 10928.0 * SQRT2) * DECAY_T + (65266.0 - 15580.0 * SQRT2)
    h_bar[1, 2] = -10268.0 * SQRT2 * DECAY_T - (5134.0 - 10268.0 * SQRT2)
    h_bar[1, 3] = (13600.0 + 5508.0 * SQRT2) * DECAY_T - (12512.0 + 9996.0 * SQRT2)
    h_bar[2, 2] = (78234.0 - 415769.0 * SQRT2) * DECAY_T + (235654.0 + 163183.0 * SQRT2)
    h_bar[2, 3] = (-875432.0 + 576963.0 * SQRT2) * DECAY_T + (943194.0 - 626093.0 * SQRT2)
    h_bar[3, 3] = (83606.0 - 164543.0 * SQRT2) * DECAY_T + (104210.0 + 49521.0 * SQRT2)
    h_bar = h_bar + np.triu(h_bar, 1).T
    return h_bar / 87278.0


def lambda_f_value(x_1, x_2):
    return (
        (15.0 - 10.5 * SQRT2) * x_1**5
        + (6.0 - 4.5 * SQRT2) * (x_1**4 * x_2 + 2.0 * x_1**2 * x_2**3)
        + (-15.0 + 10.0 * SQRT2) * x_1**3
        + (-1.0 + SQRT2) * (6.0 * x_1**2 * x_2 + x_2**3)
        + (3.75 - 1.875 * SQRT2) * x_1
        - 1.125 * SQRT2 * x_2
    )


def lambda_f_grad(x_1, x_2):
    grad_x = (
        5.0 * (15.0 - 10.5 * SQRT2) * x_1**4
        + (6.0 - 4.5 * SQRT2) * (4.0 * x_1**3 * x_2 + 4.0 * x_1 * x_2**3)
        + 3.0 * (-15.0 + 10.0 * SQRT2) * x_1**2
        + 12.0 * (-1.0 + SQRT2) * x_1 * x_2
        + (3.75 - 1.875 * SQRT2)
    )
    grad_y = (
        (6.0 - 4.5 * SQRT2) * (x_1**4 + 6.0 * x_1**2 * x_2**2)
        + (-1.0 + SQRT2) * (6.0 * x_1**2 + 3.0 * x_2**2)
        - 1.125 * SQRT2
    )
    return np.array([grad_x, grad_y], dtype=float)


def lambda_g_value(x_1, x_2):
    coefficient = -(3.0 - 2.0 * SQRT2) / 4.0
    return coefficient * (2.0 * x_1**2 - 1.0) * (2.0 * x_1**2 - (3.0 + 2.0 * SQRT2)) * x_2


def lambda_g_grad(x_1, x_2):
    coefficient = -(3.0 - 2.0 * SQRT2) / 4.0
    factor = (2.0 * x_1**2 - 1.0) * (2.0 * x_1**2 - (3.0 + 2.0 * SQRT2))
    grad_x = coefficient * x_2 * 4.0 * x_1 * (4.0 * x_1**2 - (4.0 + 2.0 * SQRT2))
    grad_y = coefficient * factor
    return np.array([grad_x, grad_y], dtype=float)


def lambda_c_value(x_1, x_2):
    radius_shift = x_1**2 + x_2**2 - (2.0 + SQRT2)
    return x_1 * radius_shift**2


def lambda_c_grad(x_1, x_2):
    radius_shift = x_1**2 + x_2**2 - (2.0 + SQRT2)
    grad_x = radius_shift**2 + 4.0 * x_1**2 * radius_shift
    grad_y = 4.0 * x_1 * x_2 * radius_shift
    return np.array([grad_x, grad_y], dtype=float)


def rotated_value_and_grad(func, grad_func, x_1, x_2):
    value = func(-x_2, x_1)
    base_grad = grad_func(-x_2, x_1)
    grad = np.array([base_grad[1], -base_grad[0]], dtype=float)
    return value, grad


def lambda_base_value_and_grad(x_1, x_2):
    omega_1 = ((163.0 + 106.0 * SQRT2) * DECAY_T - (195.0 + 129.0 * SQRT2)) / 34.0
    omega_3 = ((57.0 + 33.0 * SQRT2) * DECAY_T + (53.0 + 45.0 * SQRT2)) / 34.0

    value = lambda_f_value(x_1, x_2)
    grad = lambda_f_grad(x_1, x_2)

    value += omega_1 * lambda_g_value(x_1, x_2)
    grad += omega_1 * lambda_g_grad(x_1, x_2)

    rotated_value, rotated_grad = rotated_value_and_grad(lambda_g_value, lambda_g_grad, x_1, x_2)
    value -= omega_3 * rotated_value
    grad -= omega_3 * rotated_grad
    return value, grad


def lambda_dai_value_and_grad(x_1, x_2):
    c_bar_1 = ((-39.0 + 15.0 * SQRT2) * DECAY_T - (2529.0 - 1756.0 * SQRT2)) / 272.0
    c_bar_2 = ((-65.0 + 8.0 * SQRT2) * DECAY_T + (681.0 - 462.0 * SQRT2)) / 272.0

    value, grad = lambda_base_value_and_grad(x_1, x_2)

    value += c_bar_1 * lambda_c_value(x_1, x_2)
    grad += c_bar_1 * lambda_c_grad(x_1, x_2)

    rotated_value, rotated_grad = rotated_value_and_grad(lambda_c_value, lambda_c_grad, x_1, x_2)
    value += c_bar_2 * rotated_value
    grad += c_bar_2 * rotated_grad
    return value, grad


def f_dai(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    lambda_value, _ = lambda_dai_value_and_grad(x[0], x[1])
    mu_value, _ = lambda_dai_value_and_grad(-x[1], x[0])
    return float(lambda_value * x[2] + mu_value * x[3])


def grad_dai(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    lambda_value, lambda_grad = lambda_dai_value_and_grad(x[0], x[1])
    mu_value, mu_base_grad = lambda_dai_value_and_grad(-x[1], x[0])
    mu_grad = np.array([mu_base_grad[1], -mu_base_grad[0]], dtype=float)

    return np.array(
        [
            lambda_grad[0] * x[2] + mu_grad[0] * x[3],
            lambda_grad[1] * x[2] + mu_grad[1] * x[3],
            lambda_value,
            mu_value,
        ],
        dtype=float,
    )


def theoretical_sequence(max_iter):
    x = initial_point()
    step = initial_step()
    variables = [x.copy()]

    for _ in range(max_iter):
        x = x + step
        variables.append(x.copy())
        step = M_MATRIX.dot(step)

    variables = np.array(variables, dtype=float)
    values = np.array([f_dai(x_k) for x_k in variables], dtype=float)
    return variables, values


def run_bfgs_method(method, rule, max_iter, tol):
    result = BFGS(
        f_np=f_dai,
        x_0=initial_point(),
        max_iter=max_iter,
        fill_iteration=True,
        tol=tol,
        line_search=rule,
        grad_np=grad_dai,
        H_0=initial_inverse_hessian(),
        **LINE_SEARCH_OPTIONS,
    )
    variables, values, runtime = result.history()
    return {
        "variables": ensure_length(variables, max_iter + 1),
        "values": ensure_length(values, max_iter + 1),
        "runtime": runtime,
        "stop_reason": result.stop_reason,
    }


def run_s_bfgs_method(method, rule, max_iter, tol):
    result = specular.BFGS_method(
        f=f_dai,
        x_0=initial_point(),
        tol=tol,
        max_iter=max_iter,
        fill_iteration=True,
        line_search=rule,
        H_0=initial_inverse_hessian(),
        print_bar=False,
        **LINE_SEARCH_OPTIONS,
    )
    variables, values, runtime = result.history()
    return {
        "variables": ensure_length(variables, max_iter + 1),
        "values": ensure_length(values, max_iter + 1),
        "runtime": runtime,
        "stop_reason": result.stop_reason,
    }


def run_speg(max_iter, tol):
    step_size = specular.StepSchedule(
        name="square_summable_not_summable",
        parameters=[4.0, 0.0],
    )
    result = specular.gradient_method(
        f=f_dai,
        x_0=initial_point(),
        step_size=step_size,
        tol=tol,
        max_iter=max_iter,
        fill_iteration=True,
        print_bar=False,
    )
    variables, values, runtime = result.history()
    return {
        "variables": ensure_length(variables, max_iter + 1),
        "values": ensure_length(values, max_iter + 1),
        "runtime": runtime,
        "stop_reason": result.stop_reason,
    }


def run_experiment(methods, max_iter=100, tol=1e-12, pdf=False, show=False):
    print("\n[Experiment 14] Dai 2013 perfect-example construction")
    print(f"Settings: max_iter={max_iter}, tol={tol}, t={DECAY_T:.8f}")
    print(f"Line-search parameters: {LINE_SEARCH_OPTIONS}")

    results = {}

    if "Dai sequence" in methods:
        variables, values = theoretical_sequence(max_iter)
        results["Dai sequence"] = {
            "variables": variables,
            "values": values,
            "runtime": 0.0,
            "stop_reason": "paper sequence",
        }

    if "SPEG" in methods:
        try:
            results["SPEG"] = run_speg(max_iter=max_iter, tol=tol)
        except Exception as exc:
            print(f"SPEG failed: {exc}")

    for method, rule in BFGS_LINE_SEARCH_RULES.items():
        if method not in methods:
            continue
        try:
            results[method] = run_bfgs_method(method, rule, max_iter=max_iter, tol=tol)
        except Exception as exc:
            print(f"{method} failed: {exc}")

    for method, rule in S_BFGS_LINE_SEARCH_RULES.items():
        if method not in methods:
            continue
        try:
            results[method] = run_s_bfgs_method(method, rule, max_iter=max_iter, tol=tol)
        except Exception as exc:
            print(f"{method} failed: {exc}")

    print_summary(results)
    plot_comparison(
        results=results,
        base_dir=CURRENT_DIR,
        filename=f"figure14-dai-perfect-{max_iter}",
        title="Dai 2013 Perfect Example",
        pdf=pdf,
        show=show,
    )

    return results
