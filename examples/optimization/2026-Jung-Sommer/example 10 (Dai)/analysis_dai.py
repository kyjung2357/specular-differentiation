import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CURRENT_DIR.parents[0]
PACKAGE_ROOT = CURRENT_DIR.parents[4] / "specular-differentiation"

for path in (EXAMPLES_DIR, CURRENT_DIR, PACKAGE_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

import specular
from specular.optimization.classical_solver import BFGS
from specular.optimization import BFGS_method as S_BFGS
from tools import plot_comparison, plot_theoretical_sequence

specular.change_backend("cpu_numpy")

U1_POINTS = np.array([-87.5, -73.5, 86.5, 87.5, 73.5, -86.5])
XI_VALUES = np.array([8251 / 458, -2847 / 387, -6981 / 212, 8251 / 458, -2847 / 387, -6981 / 212])
GAMMA_VALUES = np.array([55 / 229, -44 / 387, 33 / 106, -55 / 229, 44 / 387, -33 / 106])
DEFAULT_METHODS = [
    "BFGS-E",
    "BFGS-S",
    "BFGS-W",
    "BFGS-A",
    "S-BFGS-E",
    "S-BFGS-S",
    "S-BFGS-W",
    "S-BFGS-A",
]
LINE_SEARCH_RULES = {
    "BFGS-E": "exact",
    "BFGS-S": "strong_wolfe",
    "BFGS-W": "wolfe",
    "BFGS-A": "armijo",
    "S-BFGS-E": "exact",
    "S-BFGS-S": "strong_wolfe",
    "S-BFGS-W": "wolfe",
    "S-BFGS-A": "armijo",
}


def psi(t):
    r = np.abs(t)
    if r <= 0.1:
        return 1.0
    if r >= 0.4:
        return 0.0

    t_norm = (r - 0.1) / 0.3
    return 1.0 - t_norm**3 * (10.0 - 15.0 * t_norm + 6.0 * t_norm**2)


def psi_prime(t):
    r = np.abs(t)
    if r <= 0.1 or r >= 0.4:
        return 0.0

    t_norm = (r - 0.1) / 0.3
    d_norm = -(30.0 * t_norm**2 - 60.0 * t_norm**3 + 30.0 * t_norm**4) / 0.3
    return d_norm * np.sign(t)


def get_xi_gamma(u1):
    xi = 0.0
    gamma = 0.0
    dxi = 0.0
    dgamma = 0.0

    for i in range(6):
        p = psi(u1 - U1_POINTS[i])
        dp = psi_prime(u1 - U1_POINTS[i])

        xi += XI_VALUES[i] * p
        gamma += GAMMA_VALUES[i] * p
        dxi += XI_VALUES[i] * dp
        dgamma += GAMMA_VALUES[i] * dp

    return xi, gamma, dxi, dgamma


def f_dai(x):
    u1, u2 = x[0], x[1]
    xi, gamma, _, _ = get_xi_gamma(u1)
    return float((xi + gamma * u1) * u2)


def f_dai_grad(x):
    u1, u2 = x[0], x[1]
    xi, gamma, dxi, dgamma = get_xi_gamma(u1)

    du1 = (dxi + dgamma * u1 + gamma) * u2
    du2 = xi + gamma * u1

    return np.array([du1, du2])


def initial_point():
    return np.array([-87.5, -229.0 / 44.0])


def initial_inverse_hessian():
    B0 = np.array([[5.0 / 56.0, 0.0], [0.0, 12.0 / 56.0]])
    return np.linalg.inv(B0)


def run_experiment(methods=None, max_iter=25, tol=1e-6, pdf=False, show=False):
    print("\n[Experiment 10] Dai BFGS counterexample")
    methods = DEFAULT_METHODS if methods is None else methods

    x_0 = initial_point()

    runners = {
        "BFGS-E": lambda: BFGS(
            f_np=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-E"],
            grad_np=f_dai_grad,
        ).history(),
        "BFGS-S": lambda: BFGS(
            f_np=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-S"],
            grad_np=f_dai_grad,
        ).history(),
        "BFGS-W": lambda: BFGS(
            f_np=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-W"],
            grad_np=f_dai_grad,
        ).history(),
        "BFGS-A": lambda: BFGS(
            f_np=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-A"],
            grad_np=f_dai_grad,
        ).history(),
        "S-BFGS-E": lambda: S_BFGS(
            f=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-E"],
            print_bar=False,
        ).history(),
        "S-BFGS-S": lambda: S_BFGS(
            f=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-S"],
            print_bar=False,
        ).history(),
        "S-BFGS-W": lambda: S_BFGS(
            f=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-W"],
            print_bar=False,
        ).history(),
        "S-BFGS-A": lambda: S_BFGS(
            f=f_dai,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-A"],
            print_bar=False,
        ).history(),
    }

    results = {}
    for method in methods:
        print(f"Running {method}...")
        try:
            x_hist, f_hist, _ = runners[method]()
            results[method] = {
                "values": np.array(f_hist, dtype=float),
                "variables": np.array(x_hist, dtype=float),
            }
            print(f"  finished in {len(f_hist) - 1} iterations; final loss={f_hist[-1]:.6e}")
        except Exception as exc:
            print(f"  failed: {exc}")

    plot_comparison(
        results,
        CURRENT_DIR,
        filename="comparison_dai_8methods",
        title="Dai counterexample",
        xlim=(-100.0, 100.0),
        ylim=(-10.0, 20.0),
        pdf=pdf,
        show=show,
    )


def save_theoretical_sequence(pdf=False, show=False):
    phi = 3.0 / 14.0
    h = np.zeros(6)
    a = [14.0, 160.0, 1.0]
    b = [1.0, -1.0 / 16.0, 5.0 / 56.0]

    h[0] = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    h[1] = a[0] * b[0] * phi + a[1] * b[1] + a[2] * b[2]
    h[2] = a[0] * b[0] * phi + a[1] * b[1] * phi + a[2] * b[2]
    h[3] = h[0] * phi
    h[4] = h[1] * phi
    h[5] = h[2] * phi

    c = [-3.0, 1.0, -6.0, -3.0, 1.0, -6.0]
    x_seq = []
    f_seq = []

    for k in range(30):
        i = k % 6
        j = k // 6
        factor = -(1.0 / (1.0 - phi))
        if i < 3:
            u2 = factor * h[i] * (phi ** (2 * j))
        else:
            u2 = factor * h[i - 3] * (phi ** (2 * j + 1))

        u1 = U1_POINTS[i]
        x_seq.append([u1, u2])
        f_seq.append(u2 * c[i])

    plot_theoretical_sequence(
        np.array(x_seq, dtype=float),
        np.array(f_seq, dtype=float),
        U1_POINTS,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
