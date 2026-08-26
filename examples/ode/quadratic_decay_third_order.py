"""Compare third-order SE with CN and RK3 for u' = -u^2."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import specular

from _common import crank_nicolson, maximum_error, rk3


STEP_COUNTS = (
    10,
    14,
    20,
    28,
    40,
    57,
    80,
    113,
    160,
    226,
    320,
    453,
    640,
    905,
    1280,
    1810,
    2560,
)
type ErrorSolver = Callable[[int], float]
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


def sigma_n(n: int, t_n: float, u_n: float, h_n: float) -> float:
    del n, t_n, h_n
    return u_n * u_n


def solve_SE(n_steps: int) -> float:
    result = specular.ellipse_scheme(
        F,
        0.0,
        1.0,
        1.0,
        n_steps=n_steps,
        sigma_n=sigma_n,
        atol=1e-14,
        rtol=1e-14,
        max_iter=200,
    )
    return maximum_error(result.t, result.u, exact)


def solve_classical(
    method: Callable[..., tuple[np.ndarray, np.ndarray]],
    n_steps: int,
) -> float:
    if method is crank_nicolson:
        t, u = method(
            F,
            0.0,
            1.0,
            1.0,
            n_steps,
            atol=1e-14,
            rtol=1e-14,
            max_iter=200,
        )
    else:
        t, u = method(F, 0.0, 1.0, 1.0, n_steps)
    return maximum_error(t, u, exact)


def evaluate(methods: tuple[Method, ...]) -> tuple[ErrorSeries, ...]:
    """Return the error series for the requested methods."""

    return tuple(
        (
            label,
            np.array(
                [solver(n_steps) for n_steps in STEP_COUNTS],
                dtype=np.float64,
            ),
        )
        for label, solver in methods
    )


def main() -> None:
    methods: tuple[Method, ...] = (
        (r"SE ($\sigma_n=u_n^2$)", solve_SE),
        ("CN", lambda n: solve_classical(crank_nicolson, n)),
        ("RK3", lambda n: solve_classical(rk3, n)),
    )
    series = evaluate(methods)

    coarse_index = STEP_COUNTS.index(1280)
    fine_index = STEP_COUNTS.index(2560)
    print("u' = -u^2 on [0, 1]")
    print(f"{'method':<24} {'error at N=2560':>18} {'order':>10}")
    for label, errors in series:
        order = math.log2(errors[coarse_index] / errors[fine_index])
        print(f"{label:<24} {errors[fine_index]:18.8e} {order:10.6f}")

    step_sizes = 1.0 / np.asarray(STEP_COUNTS, dtype=np.float64)
    figure, ax = plt.subplots(figsize=(5.125, 1.5))
    colors = ("#d7301f", "#7b3294", "#238b45")
    markers = ("o", "x", ">")
    line_styles = ("-", "--", "--")
    for (label, errors), color, marker, line_style in zip(
        series,
        colors,
        markers,
        line_styles,
        strict=True,
    ):
        ax.loglog(
            step_sizes,
            errors,
            color=color,
            marker=marker,
            markersize=3.2,
            linewidth=1.1,
            linestyle=line_style,
            label=label,
        )

    ax.set_xlim(1.15 * step_sizes[0], step_sizes[-1] / 1.15)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Maximum global error")
    ax.grid(color="0.85", linewidth=0.4, which="major")
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        handlelength=1.7,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.6)

    figure.subplots_adjust(
        left=0.105,
        right=0.75,
        bottom=0.15,
        top=0.97,
    )
    figure.savefig(
        os.path.join(figures_dir, "quadratic_decay_third_order.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
