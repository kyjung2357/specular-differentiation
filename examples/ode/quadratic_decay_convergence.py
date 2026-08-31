"""Compare CN, SE2, SE3, SE4, RK3, and RK4 for u' = -u^2."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import specular

from _common import crank_nicolson, maximum_error, rk3, rk4, uniform_step_count


T_0 = 0.0
T = 1.0
U_0 = 1.0
STEP_SIZES = 1.0e-1 * 2.0 ** (-np.arange(6, dtype=np.float64))
type ErrorSolver = Callable[[float], float]
type Method = tuple[str, ErrorSolver]
type ErrorSeries = tuple[str, np.ndarray]

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


def solve_se(h: float, mode: str) -> float:
    n_steps = uniform_step_count(T_0, T, h)
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


def solve_classical(
    method: Callable[..., tuple[np.ndarray, np.ndarray]],
    h: float,
) -> float:
    n_steps = uniform_step_count(T_0, T, h)
    if method is crank_nicolson:
        t, u = method(
            F,
            T_0,
            T,
            U_0,
            n_steps,
            atol=0.0,
            rtol=1e-15,
            max_iter=2000,
        )
    else:
        t, u = method(F, T_0, T, U_0, n_steps)
    return maximum_error(t, u, exact)


def evaluate(methods: tuple[Method, ...]) -> tuple[ErrorSeries, ...]:
    """Return the error series for the requested methods."""

    return tuple(
        (
            label,
            np.array(
                [solver(float(h)) for h in STEP_SIZES],
                dtype=np.float64,
            ),
        )
        for label, solver in methods
    )


def main() -> None:
    methods: tuple[Method, ...] = (
        ("CN", lambda h: solve_classical(crank_nicolson, h)),
        (r"SE2", lambda h: solve_se(h, "fixed")),
        (r"SE3", lambda h: solve_se(h, "third")),
        (r"SE4", lambda h: solve_se(h, "fourth")),
        ("RK3", lambda h: solve_classical(rk3, h)),
        ("RK4", lambda h: solve_classical(rk4, h)),
    )
    series = evaluate(methods)

    coarse_index = 0
    fine_index = -1
    coarse_h = float(STEP_SIZES[coarse_index])
    fine_h = float(STEP_SIZES[fine_index])
    print("u' = -u^2 on [0, 1]")
    print(f"{'method':<24} {f'error at h={fine_h:.8g}':>24} {'order':>10}")
    for label, errors in series:
        order = math.log(errors[coarse_index] / errors[fine_index]) / math.log(
            coarse_h / fine_h
        )
        print(f"{label:<24} {errors[fine_index]:24.8e} {order:10.6f}")

    figure, ax = plt.subplots(figsize=(5.125, 1.8))
    colors = (
        "#7b3294",
        "#fcbba1",
        "#ef3b2c",
        "#99000d",
        "#238b45",
        "#08519c",
    )
    markers = ("x", "s", "o", "D", ">", "v")
    line_styles = ("--", "-", "-", "-", "--", "--")
    for (label, errors), color, marker, line_style in zip(
        series,
        colors,
        markers,
        line_styles,
        strict=True,
    ):
        ax.loglog(
            STEP_SIZES,
            errors,
            color=color,
            marker=marker,
            markersize=3.0,
            linewidth=1.1,
            linestyle=line_style,
            label=label,
        )

    ax.set_xlim(1.15 * STEP_SIZES[0], STEP_SIZES[-1] / 1.15)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Maximum global error")
    ax.grid(color="0.85", linewidth=0.4, which="major")
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=7.5,
        handlelength=1.7,
        handletextpad=0.6,
        labelspacing=0.45,
        borderpad=0.4,
        markerscale=1.0,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.6)

    figure.subplots_adjust(
        left=0.105,
        right=0.78,
        bottom=0.22,
        top=0.97,
    )
    figure.savefig(
        os.path.join(figures_dir, "quadratic_decay_convergence.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
