"""Compare SE scale choices and RK4 for quadratic decay."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import specular

from _common import maximum_error, rk4


T_0 = 0.0
T = 1.0
U_0 = 1.0
STEP_COUNTS = (
    10,
    18,
    32,
    56,
    100,
    178,
    316,
    562,
    1000,
    1778,
    3162,
    5623,
    10000,
)

METHOD_STYLES = (
    (r"SE ($\sigma_n=1$)", "fixed", "#fcbba1", "s", "-"),
    (r"SE ($\sigma_n=\sigma_\ast$)", "third", "#ef3b2c", "o", "-"),
    (
        r"SE ($\sigma_n=\sigma_{\mathrm{bal}}$)",
        "fourth",
        "#99000d",
        "D",
        "-",
    ),
    ("RK4", "rk4", "#08519c", "v", "--"),
)

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

matplotlib.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.axisbelow": True,
        "lines.dashed_pattern": (3.7, 1.6),
        "lines.dash_capstyle": "butt",
        "lines.scale_dashes": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    }
)


def exact(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + t)


def F(t: float, u: float) -> float:
    del t
    return -(u * u)


def derivatives_of_F(point: np.ndarray) -> np.ndarray:
    """Return L_F F and L_F^2 F at ``point = [t, u]``."""

    _, u = map(float, point)
    return np.array([2.0 * u**3, -6.0 * u**4])


def run_se(n_steps: int, mode: str) -> float:
    common_options = {
        "n_steps": n_steps,
        "atol": 0.0,
        "rtol": 1e-15,
        "max_iter": 2000,
    }
    if mode == "fixed":
        result = specular.ellipse_scheme(
            F,
            T_0,
            T,
            U_0,
            sigma_n=1.0,
            **common_options,
        )
    elif mode == "third":
        result = specular.ellipse_scheme(
            F,
            T_0,
            T,
            U_0,
            third_order=True,
            derivatives_of_F=derivatives_of_F,
            **common_options,
        )
    elif mode == "fourth":
        result = specular.ellipse_scheme(
            F,
            T_0,
            T,
            U_0,
            fourth_order=True,
            derivatives_of_F=derivatives_of_F,
            **common_options,
        )
    else:
        raise ValueError(f"unknown SE mode: {mode}")

    return maximum_error(result.t, result.u, exact)


def run_rk4(n_steps: int) -> float:
    t, u = rk4(F, T_0, T, U_0, n_steps)
    return maximum_error(t, u, exact)


def run_method(n_steps: int, method: str) -> float:
    if method in {"fixed", "third", "fourth"}:
        return run_se(n_steps, method)
    if method == "rk4":
        return run_rk4(n_steps)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    print("Quadratic decay: SE scale choices and RK4")
    print(f"{'method':>36} {'error at h=1e-4':>20}")

    errors_by_method: dict[str, np.ndarray] = {}
    for label, method, *_ in METHOD_STYLES:
        errors = np.array(
            [run_method(n_steps, method) for n_steps in STEP_COUNTS],
            dtype=np.float64,
        )
        errors_by_method[method] = errors
        print(f"{label:>36} {errors[-1]:20.6e}")

    step_sizes = (T - T_0) / np.asarray(STEP_COUNTS, dtype=np.float64)
    figure, ax = plt.subplots(figsize=(5.125, 1.8))
    for label, method, color, marker, linestyle in METHOD_STYLES:
        ax.loglog(
            step_sizes,
            errors_by_method[method],
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=1.0,
            linestyle=linestyle,
            label=label,
        )

    ax.set_xlim(1e-1, 1e-4)
    ax.set_xticks(
        (1e-1, 1e-2, 1e-3, 1e-4),
        (r"$10^{-1}$", r"$10^{-2}$", r"$10^{-3}$", r"$10^{-4}$"),
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Maximum global error")
    ax.grid(color="0.85", linewidth=0.4, which="major")

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.6,
        labelspacing=0.45,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.6)

    figure.subplots_adjust(
        left=0.105,
        right=0.73,
        bottom=0.19,
        top=0.97,
    )
    figure.savefig(
        os.path.join(figures_dir, "quadratic_decay_fourth_order.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
