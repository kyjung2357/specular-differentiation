"""Compare SE scale choices and classical methods for u' = 1/u."""

from __future__ import annotations

import os
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import specular

from _common import crank_nicolson, maximum_error, rk3, rk4, uniform_step_count


STEP_COUNTS = np.rint(np.logspace(1.0, 3.0, 17)).astype(np.int64)
STEP_SIZES = 1.0 / STEP_COUNTS.astype(np.float64)
type StepScale = float | Callable[[int, float, float, float], float]
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
    return np.sqrt(1.0 + 2.0 * t)


def F(t: float, u: float) -> float:
    del t
    return 1.0 / u


def mesh_scale(power: float) -> StepScale:
    def sigma_n(n: int, t_n: float, u_n: float, h: float) -> float:
        del n, t_n, u_n
        return h**power

    return sigma_n


def solve_SE(sigma_n: StepScale, h: float) -> float:
    n_steps = uniform_step_count(0.0, 1.0, h)
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
    h: float,
) -> float:
    n_steps = uniform_step_count(0.0, 1.0, h)
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
                [solver(float(h)) for h in STEP_SIZES],
                dtype=np.float64,
            ),
        )
        for label, solver in methods
    )


def main() -> None:
    methods: tuple[Method, ...] = (
        (r"SE2 $(\sigma_n = 1)$", lambda h: solve_SE(1.0, h)),
        (
            r"SE3 $\left(\sigma_n = \sqrt{h}\right)$",
            lambda h: solve_SE(mesh_scale(0.5), h),
        ),
        (r"SE3 $(\sigma_n = h)$", lambda h: solve_SE(mesh_scale(1.0), h)),
        (r"SE3 $\left(\sigma_n = h^{2}\right)$", lambda h: solve_SE(mesh_scale(2.0), h)),
        (r"SE3 $\left(\sigma_n = h^{3}\right)$", lambda h: solve_SE(mesh_scale(3.0), h)),
        (r"SE3 $\left(\sigma_n = h^{4}\right)$", lambda h: solve_SE(mesh_scale(4.0), h)),
        ("CN", lambda h: solve_classical(crank_nicolson, h)),
        ("RK3", lambda h: solve_classical(rk3, h)),
        ("RK4", lambda h: solve_classical(rk4, h)),
    )
    series = evaluate(methods)

    print("u' = 1/u on [0, 1]")
    final_h = float(STEP_SIZES[-1])
    print(f"{'method':<26} {f'error at h={final_h:.8g}':>24}")
    for label, errors in series:
        print(f"{label:<26} {errors[-1]:24.8e}")

    figure, ax = plt.subplots(figsize=(5.125, 2.55))
    colors = (
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#ef3b2c",
        "#cb181d",
        "#99000d",
        "#7b3294",
        "#238b45",
        "#08519c",
    )
    markers = ("o", "s", "^", "D", "P", "X", "x", ">", "v")
    line_styles = ("-", "-", "-", "-", "-", "-", "--", "--", "--")
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
            markersize=3.2,
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
        right=0.75,
        bottom=0.15,
        top=0.97,
    )
    figure.savefig(
        os.path.join(figures_dir, "inverse_equation_small_scale.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
