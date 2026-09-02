"""Compare fixed and diverging SE scales with CN on E3b1 pairs."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import specular

from _common import crank_nicolson, observed_order


STEP_SIZES = np.array(
    (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125),
    dtype=np.float64,
)
FIXED_SCALES = (1.0, 3.0, 10.0)
TRANSITION_TIME = math.atanh(1.0 / math.sqrt(3.0))
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


def exact(t: float) -> float:
    """Return the exact solution ``u(t) = tanh(t)``."""

    return math.tanh(t)


def F(t: float, u: float) -> float:
    del t
    return 1.0 - u * u


def second_transport(t: float) -> float:
    """Return ``L_F^2 F`` along the exact solution."""

    u = exact(t)
    return 2.0 * (1.0 - u * u) * (3.0 * u * u - 1.0)


def e3_pair(h: float) -> tuple[float, float, float, float]:
    """Return consecutive exact points whose E3 coefficient ``A`` vanishes."""

    left = TRANSITION_TIME - h
    right = TRANSITION_TIME

    for _ in range(80):
        midpoint = 0.5 * (left + right)
        residual = second_transport(midpoint) + second_transport(midpoint + h)
        if residual < 0.0:
            left = midpoint
        else:
            right = midpoint

    t_left = 0.5 * (left + right)
    t_right = t_left + h
    return t_left, t_right, exact(t_left), exact(t_right)


def classification_coefficients(h: float) -> tuple[float, float, float, float]:
    """Return ``A``, ``B``, ``C``, and ``cf`` for the selected pair."""

    _, _, u_left, u_right = e3_pair(h)

    def quantities(u: float) -> tuple[float, float, float]:
        field = 1.0 - u * u
        first_transport = -2.0 * u * field
        a = 2.0 * field * (3.0 * u * u - 1.0)
        b = 3.0 * field * first_transport * first_transport
        c = field * field
        return a, b, c

    a, b, c = quantities(u_left)
    d, e, f = quantities(u_right)
    A = a + d
    B = A * (c + f) - b - e
    C = A * c * f - b * f - e * c
    return A, B, C, c * f


def solve_se(h: float, sigma: float) -> float:
    t_left, t_right, u_left, u_right = e3_pair(h)
    result = specular.ellipse_scheme(
        F,
        t_left,
        t_right,
        u_left,
        n_steps=1,
        sigma_n=sigma,
        atol=0.0,
        rtol=1e-15,
        max_iter=1000,
    )
    return abs(float(result.u[-1]) - u_right)


def solve_cn(h: float) -> float:
    t_left, t_right, u_left, u_right = e3_pair(h)
    _, values = crank_nicolson(
        F,
        t_left,
        t_right,
        u_left,
        1,
        atol=0.0,
        rtol=1e-15,
        max_iter=1000,
    )
    return abs(float(values[-1]) - u_right)


def evaluate(methods: tuple[Method, ...]) -> tuple[ErrorSeries, ...]:
    return tuple(
        (
            label,
            np.array([solver(float(h)) for h in STEP_SIZES], dtype=np.float64),
        )
        for label, solver in methods
    )


def main() -> None:
    representative_h = 0.1
    for h in STEP_SIZES:
        A_h, B_h, C_h, cf_h = classification_coefficients(float(h))
        assert abs(A_h) < 1e-14
        assert B_h < 0.0 and C_h < 0.0 and B_h * C_h > 0.0
        assert cf_h > 0.0

    A, B, C, cf = classification_coefficients(representative_h)

    methods: tuple[Method, ...] = tuple(
        (
            r"SE2 $(\sigma_n = 1)$"
            if sigma == 1.0
            else rf"SE $(\sigma_n = {sigma:g})$",
            lambda h, sigma=sigma: solve_se(h, sigma),
        )
        for sigma in FIXED_SCALES
    ) + (
        (r"SE $\left(\sigma_n = h^{-1}\right)$", lambda h: solve_se(h, 1.0 / h)),
        ("CN", solve_cn),
    )
    series = evaluate(methods)

    t_left, t_right, u_left, u_right = e3_pair(representative_h)
    print("u' = 1-u^2: one-step E3b1 pairs")
    print(
        f"h={representative_h:g}, "
        f"(t_n,t_(n+1))=({t_left:.12f},{t_right:.12f}), "
        f"(u_n,u_(n+1))=({u_left:.12f},{u_right:.12f})"
    )
    print(f"A={A:.3e}, B={B:.8f}, C={C:.8f}, BC={B*C:.8f}, cf={cf:.8f}")
    error_heading = f"error at h={STEP_SIZES[-1]:g}"
    print(f"{'method':<22} {error_heading:>22} {'order':>10}")
    for label, errors in series:
        order = observed_order(float(errors[-2]), float(errors[-1]))
        print(f"{label:<22} {errors[-1]:22.8e} {order:10.6f}")

    figure, ax = plt.subplots(figsize=(5.125, 1.8))
    colors = ("#fcae91", "#fb6a4a", "#cb181d", "#99000d", "#7b3294")
    markers = ("o", "s", "^", "D", "x")
    line_styles = ("-", "-", "-", "-", "--")
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
    ax.set_ylabel("One-step error")
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
        left=0.115,
        right=0.73,
        bottom=0.22,
        top=0.97,
    )
    figure.savefig(
        os.path.join(figures_dir, "autonomous_large_scale.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
