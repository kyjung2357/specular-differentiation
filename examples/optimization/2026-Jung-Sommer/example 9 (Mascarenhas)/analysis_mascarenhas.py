import os
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
from tools import plot_comparison, plot_reference_iterates

print("version of specular-differentiation: ", specular.__version__)
specular.change_backend("cpu_numpy")

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


def psi(c):
    r = np.linalg.norm(c)
    if r <= 0.08:
        return 1.0
    if r >= 0.16:
        return 0.0

    t = (r - 0.08) / 0.08
    return 1.0 - t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def psi_grad(c):
    r = np.linalg.norm(c)
    if r <= 0.08 or r >= 0.16 or r == 0:
        return np.zeros_like(c)

    t = (r - 0.08) / 0.08
    dpsi_dt = -(30.0 * t**2 - 60.0 * t**3 + 30.0 * t**4)
    dpsi_dr = dpsi_dt / 0.08
    return dpsi_dr * (c / r)


def get_octagon_vertices():
    c0 = np.array([3.0 + 2.0 * np.sqrt(2.0), 1.0 + np.sqrt(2.0)]) / 2.0
    theta = np.pi / 4.0
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    vertices = []
    current = c0
    for _ in range(8):
        vertices.append(current)
        current = R @ current
    return vertices


VERTICES = get_octagon_vertices()
H_VEC = np.array([3.0, -1.0, 0.0])
D_VEC = np.array([0.0, 1.0, 0.0])
Q_MAT = np.array(
    [
        [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
        [0.0, 0.0, -1.0],
    ]
)


def q_k(x, k):
    Q_k = np.linalg.matrix_power(Q_MAT, k)
    x_infty_k = np.array([VERTICES[k][0], VERTICES[k][1], 0.0])
    w = x - x_infty_k

    w_Qk_h = w.T @ (Q_k @ H_VEC)
    w_Qk_d = w.T @ (Q_k @ D_VEC)

    return ((-1.0) ** k * w[2]) * (1.0 + w_Qk_h + 1.1 * w_Qk_d**2)


def q_k_grad(x, k):
    Q_k = np.linalg.matrix_power(Q_MAT, k)
    x_infty_k = np.array([VERTICES[k][0], VERTICES[k][1], 0.0])
    w = x - x_infty_k

    w_Qk_h = w.T @ (Q_k @ H_VEC)
    w_Qk_d = w.T @ (Q_k @ D_VEC)

    term1 = 1.0 + w_Qk_h + 1.1 * w_Qk_d**2
    term2 = (-1.0) ** k * w[2]

    da_dw = Q_k @ H_VEC
    db_dw = Q_k @ D_VEC

    grad_w = term2 * (da_dw + 2.2 * w_Qk_d * db_dw)
    grad_w[2] += ((-1.0) ** k) * term1
    return grad_w


def f_mascarenhas(x):
    c = x[:2]

    val = 0.0
    for j, c_j in enumerate(VERTICES):
        val += psi(c - c_j) * q_k(x, j)

    prod_term = 1.0
    for c_j in VERTICES:
        prod_term *= 1.0 - psi(2.0 * (c - c_j))

    return float(val + 2.0 * prod_term)


def f_mascarenhas_grad(x):
    c = x[:2]
    sum_grad = np.zeros(3)

    for j, c_j in enumerate(VERTICES):
        psi_val = psi(c - c_j)
        q_val = q_k(x, j)

        grad_psi_c = psi_grad(c - c_j)
        grad_psi_full = np.array([grad_psi_c[0], grad_psi_c[1], 0.0])
        q_grad_full = q_k_grad(x, j)

        sum_grad += psi_val * q_grad_full + q_val * grad_psi_full

    prod_grad = np.zeros(3)
    for j, c_j in enumerate(VERTICES):
        term_grad_c = -2.0 * psi_grad(2.0 * (c - c_j))
        term_grad_full = np.array([term_grad_c[0], term_grad_c[1], 0.0])

        other_prod = 1.0
        for m, c_m in enumerate(VERTICES):
            if m != j:
                other_prod *= 1.0 - psi(2.0 * (c - c_m))

        prod_grad += other_prod * term_grad_full

    return sum_grad + 2.0 * prod_grad


def initial_point():
    x_infty_0 = np.array([VERTICES[0][0], VERTICES[0][1], 0.0])
    e_z = np.array([0.0, 0.0, 1.0])
    return x_infty_0 + e_z


def initial_inverse_hessian():
    B0_part1 = (np.sqrt(2.0) / 5.0) * np.array(
        [
            [11.0, -7.0, 12.0],
            [-7.0, 9.0, 6.0],
            [12.0, 6.0, 4.0],
        ]
    )
    B0_part2 = (1.0 / 5.0) * np.array(
        [
            [3.0, -11.0, 16.0],
            [-11.0, 7.0, 8.0],
            [16.0, 8.0, 2.0],
        ]
    )
    return np.linalg.inv(B0_part1 - B0_part2)


def run_experiment(methods, max_iter=50, tol=1e-12, pdf=False, show=False):
    print("\n[Experiment 9] Mascarenhas BFGS counterexample")

    x_0 = initial_point()
    H_0 = initial_inverse_hessian()

    runners = {
        "BFGS-E": lambda: BFGS(
            f_np=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-E"],
            grad_np=f_mascarenhas_grad,
            H_0=H_0,
            max_line_iter=60,
            max_alpha=20.0,
        ).history(),
        "BFGS-S": lambda: BFGS(
            f_np=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-S"],
            grad_np=f_mascarenhas_grad,
        ).history(),
        "BFGS-W": lambda: BFGS(
            f_np=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-W"],
            grad_np=f_mascarenhas_grad,
        ).history(),
        "BFGS-A": lambda: BFGS(
            f_np=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["BFGS-A"],
            grad_np=f_mascarenhas_grad,
        ).history(),
        "S-BFGS-E": lambda: S_BFGS(
            f=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-E"],
            H_0=H_0,
            max_line_iter=60,
            max_alpha=20.0,
            print_bar=False,
        ).history(),
        "S-BFGS-S": lambda: S_BFGS(
            f=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-S"],
            print_bar=False,
        ).history(),
        "S-BFGS-W": lambda: S_BFGS(
            f=f_mascarenhas,
            x_0=x_0,
            max_iter=max_iter,
            tol=tol,
            line_search=LINE_SEARCH_RULES["S-BFGS-W"],
            print_bar=False,
        ).history(),
        "S-BFGS-A": lambda: S_BFGS(
            f=f_mascarenhas,
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
        filename="comparison_mascarenhas_8methods",
        title="Mascarenhas counterexample",
        xlim=(-4.0, 4.0),
        ylim=(-4.0, 4.0),
        pdf=pdf,
        show=show,
    )


def save_reference_visualizations(pdf=False, show=False):
    plot_reference_iterates(CURRENT_DIR, pdf=pdf, show=show)
