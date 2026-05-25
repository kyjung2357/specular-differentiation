import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = CURRENT_DIR.parents[0]
PACKAGE_ROOT = CURRENT_DIR.parents[3]

for path in (PACKAGE_ROOT, EXAMPLES_DIR, CURRENT_DIR):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

import specular
from specular.calculation import gradient as specular_gradient
from specular.optimization.line_search import LineSearch
from tools import plot_comparison, plot_curvature_diagnostics, plot_theoretical_sequence

print("version of specular-differentiation: ", specular.__version__)
specular.change_backend("cpu_numpy")

U1_POINTS = np.array([-87.5, -73.5, 86.5, 87.5, 73.5, -86.5])
XI_VALUES = np.array([8251 / 458, -2847 / 387, -6981 / 212, 8251 / 458, -2847 / 387, -6981 / 212])
GAMMA_VALUES = np.array([55 / 229, -44 / 387, 33 / 106, -55 / 229, 44 / 387, -33 / 106])
DEFAULT_METHODS = [
    "SPEG",
    "BFGS-D",
    "S-BFGS-D",
    "BFGS-E",
    "S-BFGS-E",
    "BFGS-A",
    "S-BFGS-A",
]
LINE_SEARCH_RULES = {
    "BFGS-E": "exact",
    "BFGS-W": "wolfe",
    "BFGS-A": "armijo",
    "S-BFGS-E": "exact",
    "S-BFGS-W": "wolfe",
    "S-BFGS-A": "armijo",
}


def dai_center_distance(x):
    x_arr = np.asarray(x, dtype=float)
    return float(np.min(np.abs(x_arr[0] - U1_POINTS)))


def dai_theoretical_sequence(num_points=30):
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

    for k in range(num_points):
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

    return np.array(x_seq, dtype=float), np.array(f_seq, dtype=float)


def alpha_matching_step(target_step, direction):
    pivot = int(np.argmax(np.abs(direction)))

    if abs(direction[pivot]) <= np.finfo(float).eps:
        return np.nan, np.nan

    alpha = float(target_step[pivot] / direction[pivot])
    residual = np.linalg.norm(target_step - alpha * direction) / max(1.0, np.linalg.norm(target_step))
    return alpha, residual


def normalized_curvature(y, s):
    denominator = np.linalg.norm(y) * np.linalg.norm(s)

    if denominator <= np.finfo(float).eps:
        return np.nan

    return float(np.dot(y, s) / denominator)


def run_dai_prescribed_bfgs(max_iter=25, tol=1e-12, c_1=1e-4, c_2=0.9):
    x_reference, _ = dai_theoretical_sequence(max_iter + 1)
    H = initial_inverse_hessian()
    x = x_reference[0].copy()
    g = f_dai_grad(x)

    x_history = [x.copy()]
    f_history = [float(f_dai(x))]
    diagnostics = {
        "alpha": [],
        "direction_residual": [],
        "wolfe": [],
        "ys": [],
        "normalized_curvature": [],
        "step_norm": [],
        "gradient_norm": [float(np.linalg.norm(g))],
        "support_distance": [dai_center_distance(x)],
        "bfgs_update": [],
        "stop_reason": "max_iter reached",
    }

    n = x.size
    I = np.eye(n)

    for k in range(1, max_iter + 1):
        direction = -H.dot(g)
        target_step = x_reference[k] - x
        alpha, direction_residual = alpha_matching_step(target_step, direction)

        s = target_step
        x_new = x_reference[k].copy()

        f_current = f_history[-1]
        f_new = float(f_dai(x_new))
        g_new = f_dai_grad(x_new)
        initial_slope = float(np.dot(g, s))
        trial_slope = float(np.dot(g_new, s))
        wolfe = (
            np.isfinite(alpha)
            and alpha > 0.0
            and initial_slope < 0.0
            and f_new <= f_current + c_1 * initial_slope
            and trial_slope >= c_2 * initial_slope
        )

        y = g_new - g
        ys = float(np.dot(y, s))
        curvature = normalized_curvature(y, s)
        diagnostics["alpha"].append(alpha)
        diagnostics["direction_residual"].append(direction_residual)
        diagnostics["wolfe"].append(wolfe)
        diagnostics["ys"].append(ys)
        diagnostics["normalized_curvature"].append(curvature)
        diagnostics["step_norm"].append(float(np.linalg.norm(s)))
        diagnostics["gradient_norm"].append(float(np.linalg.norm(g_new)))
        diagnostics["support_distance"].append(dai_center_distance(x_new))

        if not np.isfinite(ys) or ys == 0.0:
            diagnostics["bfgs_update"].append(False)
            diagnostics["stop_reason"] = "zero or non-finite curvature"
            x_history.append(x_new.copy())
            f_history.append(f_new)
            return x_history, f_history, diagnostics

        rho = 1.0 / ys
        V = I - rho * np.outer(s, y)
        H = V.dot(H).dot(V.T) + rho * np.outer(s, s)

        x = x_new
        g = g_new
        x_history.append(x.copy())
        f_history.append(f_new)
        diagnostics["bfgs_update"].append(True)

    return x_history, f_history, diagnostics


def run_dai_prescribed_s_bfgs(alpha_schedule, max_iter=25, tol=1e-12, h=1e-6, zero_tol=1e-8):
    H = initial_inverse_hessian()
    x = initial_point()
    computation = specular_gradient(
        f=f_dai,
        x=x,
        h=h,
        zero_tol=zero_tol,
        quasi_Fermat=True,
        monotonicity=False,
    )
    spec_grad = np.asarray(computation[0], dtype=float).reshape(-1)

    x_history = [x.copy()]
    f_history = [float(f_dai(x))]
    diagnostics = {
        "alpha": [],
        "ys": [],
        "normalized_curvature": [],
        "step_norm": [],
        "gradient_norm": [float(np.linalg.norm(spec_grad))],
        "bfgs_update": [],
        "support_distance": [dai_center_distance(x)],
        "stop_reason": "max_iter reached",
    }

    n = x.size
    I = np.eye(n)
    step_count = max_iter

    for k in range(step_count):
        if np.linalg.norm(spec_grad) < tol:
            x_history.append(x.copy())
            f_history.append(float(f_dai(x)))
            diagnostics["alpha"].append(np.nan)
            diagnostics["ys"].append(np.nan)
            diagnostics["normalized_curvature"].append(np.nan)
            diagnostics["step_norm"].append(0.0)
            diagnostics["gradient_norm"].append(float(np.linalg.norm(spec_grad)))
            diagnostics["bfgs_update"].append(False)
            diagnostics["support_distance"].append(dai_center_distance(x))
            diagnostics["stop_reason"] = "specular gradient norm below tolerance"
            continue

        direction = -H.dot(spec_grad)
        initial_slope = float(np.dot(spec_grad, direction))
        if initial_slope >= 0.0:
            H = I.copy()
            direction = -spec_grad

        if k >= len(alpha_schedule):
            x_history.append(x.copy())
            f_history.append(float(f_dai(x)))
            diagnostics["alpha"].append(np.nan)
            diagnostics["ys"].append(np.nan)
            diagnostics["normalized_curvature"].append(np.nan)
            diagnostics["step_norm"].append(0.0)
            diagnostics["gradient_norm"].append(float(np.linalg.norm(spec_grad)))
            diagnostics["bfgs_update"].append(False)
            diagnostics["support_distance"].append(dai_center_distance(x))
            diagnostics["stop_reason"] = "alpha schedule exhausted"
            continue

        alpha = float(alpha_schedule[k])
        if not np.isfinite(alpha) or alpha <= 0.0:
            x_history.append(x.copy())
            f_history.append(float(f_dai(x)))
            diagnostics["alpha"].append(alpha)
            diagnostics["ys"].append(np.nan)
            diagnostics["normalized_curvature"].append(np.nan)
            diagnostics["step_norm"].append(0.0)
            diagnostics["gradient_norm"].append(float(np.linalg.norm(spec_grad)))
            diagnostics["bfgs_update"].append(False)
            diagnostics["support_distance"].append(dai_center_distance(x))
            diagnostics["stop_reason"] = "non-positive prescribed alpha"
            continue

        s = alpha * direction
        x_new = x + s

        computation_new = specular_gradient(
            f=f_dai,
            x=x_new,
            h=h,
            zero_tol=zero_tol,
            quasi_Fermat=True,
            monotonicity=False,
        )
        spec_grad_new = np.asarray(computation_new[0], dtype=float).reshape(-1)

        y = spec_grad_new - spec_grad
        ys = float(np.dot(y, s))
        curvature = normalized_curvature(y, s)
        diagnostics["alpha"].append(alpha)
        diagnostics["ys"].append(ys)
        diagnostics["normalized_curvature"].append(curvature)
        diagnostics["step_norm"].append(float(np.linalg.norm(s)))
        diagnostics["gradient_norm"].append(float(np.linalg.norm(spec_grad_new)))
        diagnostics["support_distance"].append(dai_center_distance(x_new))

        if not np.isfinite(ys) or ys == 0.0:
            diagnostics["bfgs_update"].append(False)
            diagnostics["stop_reason"] = "zero or non-finite curvature"
            x_history.append(x_new.copy())
            f_history.append(float(f_dai(x_new)))
            return x_history, f_history, diagnostics

        rho = 1.0 / ys
        V = I - rho * np.outer(s, y)
        H = V.dot(H).dot(V.T) + rho * np.outer(s, s)

        x = x_new
        spec_grad = spec_grad_new
        x_history.append(x.copy())
        f_history.append(float(f_dai(x)))
        diagnostics["bfgs_update"].append(True)

    return x_history, f_history, diagnostics


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


def dai_specular_gradient(x, h=1e-6, zero_tol=1e-8):
    computation = specular_gradient(
        f=f_dai,
        x=x,
        h=h,
        zero_tol=zero_tol,
        quasi_Fermat=True,
        monotonicity=False,
    )
    return np.asarray(computation[0], dtype=float).reshape(-1)


def run_dai_line_search_bfgs(
    gradient_f,
    line_search_name,
    max_iter=25,
    tol=1e-12,
    max_line_iter=20,
    max_alpha=1e8,
):
    H = initial_inverse_hessian()
    x = initial_point()
    g = np.asarray(gradient_f(x), dtype=float).reshape(-1)

    x_history = [x.copy()]
    f_history = [float(f_dai(x))]
    diagnostics = {
        "alpha": [],
        "ys": [],
        "normalized_curvature": [],
        "step_norm": [],
        "gradient_norm": [float(np.linalg.norm(g))],
        "bfgs_update": [],
        "support_distance": [dai_center_distance(x)],
        "stop_reason": "max_iter reached",
    }

    n = x.size
    I = np.eye(n)
    line_search = LineSearch(
        name=line_search_name,
        max_iter=max_line_iter,
        max_alpha=max_alpha,
        f=f_dai,
        gradient_f=gradient_f,
    )

    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            diagnostics["stop_reason"] = "gradient norm below tolerance"
            break

        direction = -H.dot(g)
        initial_slope = float(np.dot(g, direction))

        if initial_slope >= 0.0:
            H = I.copy()
            direction = -g

        try:
            alpha = float(
                line_search(
                    x=x,
                    direction=direction,
                    gradient_current=g,
                )
            )
        except Exception as exc:
            diagnostics["stop_reason"] = str(exc)
            break

        s = alpha * direction
        x_new = x + s
        g_new = np.asarray(gradient_f(x_new), dtype=float).reshape(-1)
        y = g_new - g
        ys = float(np.dot(y, s))
        curvature = normalized_curvature(y, s)

        x_history.append(x_new.copy())
        f_history.append(float(f_dai(x_new)))
        diagnostics["alpha"].append(alpha)
        diagnostics["ys"].append(ys)
        diagnostics["normalized_curvature"].append(curvature)
        diagnostics["step_norm"].append(float(np.linalg.norm(s)))
        diagnostics["gradient_norm"].append(float(np.linalg.norm(g_new)))
        diagnostics["support_distance"].append(dai_center_distance(x_new))

        if not np.isfinite(ys) or ys == 0.0:
            diagnostics["bfgs_update"].append(False)
            diagnostics["stop_reason"] = "zero or non-finite curvature"
            break

        rho = 1.0 / ys
        V = I - rho * np.outer(s, y)
        H = V.dot(H).dot(V.T) + rho * np.outer(s, s)

        x = x_new
        g = g_new
        diagnostics["bfgs_update"].append(True)

    return x_history, f_history, diagnostics


def run_dai_speg(max_iter=25, tol=1e-12, h=1e-6, zero_tol=1e-8):
    x = initial_point()
    step_size = specular.StepSchedule(
        name="square_summable_not_summable",
        parameters=[0.1, 0.0],
    )

    x_history = [x.copy()]
    f_history = [float(f_dai(x))]
    diagnostics = {
        "gradient_norm": [],
        "support_distance": [dai_center_distance(x)],
        "stop_reason": "max_iter reached",
    }

    for k in range(1, max_iter + 1):
        spec_grad = dai_specular_gradient(x, h=h, zero_tol=zero_tol)
        norm = np.linalg.norm(spec_grad)
        diagnostics["gradient_norm"].append(float(norm))

        if not np.isfinite(norm):
            diagnostics["stop_reason"] = "specular gradient norm is not finite"
            break

        if norm == 0.0 or norm < tol:
            diagnostics["stop_reason"] = "specular gradient norm below tolerance"
            break

        x = x - step_size(k) * (spec_grad / norm)
        x_history.append(x.copy())
        f_history.append(float(f_dai(x)))
        diagnostics["support_distance"].append(dai_center_distance(x))

    return x_history, f_history, diagnostics


def initial_point():
    return np.array([-87.5, -229.0 / 44.0])


def initial_inverse_hessian():
    B0 = np.array([[5.0 / 56.0, 0.0], [0.0, 12.0 / 56.0]])
    return np.linalg.inv(B0)


def run_experiment(methods=None, max_iter=25, tol=1e-12, pdf=False, show=False):
    print("\n[Experiment 10] Dai BFGS counterexample")
    print("BFGS-D reconstructs the Dai sequence and records its prescribed alpha values.")
    print("S-BFGS-D uses the same alpha values with specular gradients.")
    methods = DEFAULT_METHODS if methods is None else methods

    prescribed_bfgs = run_dai_prescribed_bfgs(
        max_iter=max_iter,
        tol=tol,
        c_1=7/7480,
        c_2=0.99,
    )
    prescribed_alpha = prescribed_bfgs[2]["alpha"]

    runners = {
        "SPEG": lambda: run_dai_speg(
            max_iter=max_iter,
            tol=tol,
            h=1e-6,
            zero_tol=1e-8,
        ),
        "BFGS-E": lambda: run_dai_line_search_bfgs(
            gradient_f=f_dai_grad,
            line_search_name=LINE_SEARCH_RULES["BFGS-E"],
            max_iter=max_iter,
            tol=tol,
            max_line_iter=60,
            max_alpha=20.0,
        ),
        "BFGS-W": lambda: run_dai_line_search_bfgs(
            gradient_f=f_dai_grad,
            line_search_name=LINE_SEARCH_RULES["BFGS-W"],
            max_iter=max_iter,
            tol=tol,
        ),
        "BFGS-A": lambda: run_dai_line_search_bfgs(
            gradient_f=f_dai_grad,
            line_search_name=LINE_SEARCH_RULES["BFGS-A"],
            max_iter=max_iter,
            tol=tol,
        ),
        "S-BFGS-E": lambda: run_dai_line_search_bfgs(
            gradient_f=dai_specular_gradient,
            line_search_name=LINE_SEARCH_RULES["S-BFGS-E"],
            max_iter=max_iter,
            tol=tol,
            max_line_iter=60,
            max_alpha=20.0,
        ),
        "S-BFGS-W": lambda: run_dai_line_search_bfgs(
            gradient_f=dai_specular_gradient,
            line_search_name=LINE_SEARCH_RULES["S-BFGS-W"],
            max_iter=max_iter,
            tol=tol,
        ),
        "S-BFGS-A": lambda: run_dai_line_search_bfgs(
            gradient_f=dai_specular_gradient,
            line_search_name=LINE_SEARCH_RULES["S-BFGS-A"],
            max_iter=max_iter,
            tol=tol,
        ),
        "BFGS-D": lambda: prescribed_bfgs,
        "S-BFGS-D": lambda: run_dai_prescribed_s_bfgs(
            alpha_schedule=prescribed_alpha,
            max_iter=max_iter,
            tol=tol,
            h=1e-6,
            zero_tol=1e-8,
        ),
    }

    results = {}
    diagnostics_by_method = {}
    for method in methods:
        print(f"Running {method}...")
        try:
            x_hist, f_hist, info = runners[method]()
            results[method] = {
                "values": np.array(f_hist, dtype=float),
                "variables": np.array(x_hist, dtype=float),
            }
            if isinstance(info, dict):
                diagnostics_by_method[method] = info
            print(f"  finished in {len(f_hist) - 1} iterations; final loss={f_hist[-1]:.6e}")
            print(f"  final x={x_hist[-1]}")
            print(f"  distance to Dai support={dai_center_distance(x_hist[-1]):.6e}")
            if isinstance(info, dict) and "stop_reason" in info:
                print(f"  stop reason={info['stop_reason']}")
            if method == "BFGS-D" and isinstance(info, dict):
                wolfe_count = sum(info["wolfe"])
                valid_alpha_count = sum(np.isfinite(info["alpha"]))
                residuals = info["direction_residual"]
                finite_residuals = [value for value in residuals if np.isfinite(value)]
                max_residual = max(finite_residuals) if finite_residuals else 0.0
                print(f"  valid prescribed alpha steps={valid_alpha_count}")
                print(f"  Wolfe-admissible prescribed steps={wolfe_count}/{valid_alpha_count}")
                print(f"  max BFGS-direction residual={max_residual:.6e}")
            if method == "S-BFGS-D" and isinstance(info, dict):
                support_distances = info["support_distance"]
                max_support_distance = max(support_distances) if support_distances else 0.0
                valid_alpha_count = sum(np.isfinite(info["alpha"]))
                print(f"  valid prescribed alpha steps used={valid_alpha_count}")
                print(f"  max distance to Dai support={max_support_distance:.6e}")
        except Exception as exc:
            print(f"  failed: {exc}")

    theoretical_x, theoretical_f = dai_theoretical_sequence(max_iter + 1)
    results["Dai sequence"] = {
        "values": theoretical_f,
        "variables": theoretical_x,
    }

    plot_comparison(
        results,
        CURRENT_DIR,
        filename="comparison_dai_diagnostic",
        title="Dai counterexample",
        xlim=(-100.0, 100.0),
        ylim=(-10.0, 20.0),
        pdf=pdf,
        show=show,
    )
    plot_curvature_diagnostics(
        diagnostics_by_method,
        CURRENT_DIR,
        filename="comparison_dai_diagnostic",
        title="Dai counterexample curvature diagnostics",
        pdf=pdf,
        show=show,
    )


def save_theoretical_sequence(pdf=False, show=False):
    x_seq, f_seq = dai_theoretical_sequence(30)

    plot_theoretical_sequence(
        x_seq,
        f_seq,
        U1_POINTS,
        CURRENT_DIR,
        pdf=pdf,
        show=show,
    )
