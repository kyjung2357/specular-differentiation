"""Compare SE and Crank--Nicolson trajectories on two ellipses."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import specular

from _common import crank_nicolson, maximum_error, uniform_step_count


STEP_SIZE = 0.3
ENDPOINT_MARGIN = 0.01
TIME_SHIFT = 2.25

figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.axisbelow": True,
        "lines.dashed_pattern": (3.7, 1.6),
        "lines.dash_capstyle": "butt",
        "lines.scale_dashes": False,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    }
)

def _upper_arc_mesh(a: float, p: float, h: float) -> tuple[float, float, int]:
    """Return an ``h``-mesh just inside the singular ellipse endpoints."""

    t_0 = max(0.0, p - a + ENDPOINT_MARGIN)
    T = p + a - ENDPOINT_MARGIN
    n_steps = uniform_step_count(t_0, T, h)
    return t_0, T, n_steps


def plot_case(ax: Axes, a: float, b: float, p: float, h: float) -> None:
    """Plot the fitted SE and Crank--Nicolson trajectories for one ellipse."""

    sigma = b / a
    is_SE2 = np.isclose(sigma, 1.0)
    method_label = "SE2" if is_SE2 else r"SE $\left(\sigma_n = \frac{b}{a}\right)$"
    legend_label = (
        r"SE2 $(\sigma_n = 1)$"
        if is_SE2
        else r"SE $\left(\sigma_n = \frac{b}{a}\right)$"
    )
    method_color = "#fc9272" if is_SE2 else "#cb181d"

    def F(t: float, u: float) -> float:
        return -(b * b * (t - p)) / (a * a * u)

    def exact(t: np.ndarray) -> np.ndarray:
        return b * np.sqrt(np.maximum(0.0, 1.0 - ((t - p) / a) ** 2))

    t_0, T, n_steps = _upper_arc_mesh(a, p, h)
    u_0 = float(exact(np.asarray(t_0)))

    SE = specular.ellipse_scheme(
        F,
        t_0,
        T,
        u_0,
        n_steps=n_steps,
        sigma_n=sigma,
        atol=1e-14,
        rtol=1e-14,
        max_iter=1000,
    )
    CN_t, CN_u = crank_nicolson(
        F,
        t_0,
        T,
        u_0,
        n_steps,
        atol=1e-14,
        rtol=1e-14,
        max_iter=1000,
    )

    SE_error = maximum_error(SE.t, SE.u, exact)
    CN_error = maximum_error(CN_t, CN_u, exact)
    ellipse_residual = float(
        np.max(np.abs(((SE.t - p) / a) ** 2 + (SE.u / b) ** 2 - 1.0))
    )
    marker_stride = max(1, n_steps // 10)

    print(
        f"a={a:g}, b={b:g}, h={h:g}: "
        f"{method_label} error={SE_error:.3e}, "
        f"CN error={CN_error:.3e}, "
        f"{method_label} ellipse residual={ellipse_residual:.3e}"
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 1000)
    ax.plot(
        p + a * np.cos(theta),
        b * np.sin(theta),
        color="0.6",
        linestyle="--",
        linewidth=1.0,
        label="Exact Ellipse",
    )
    ax.plot(
        SE.t,
        SE.u,
        color=method_color,
        marker="o",
        markevery=marker_stride,
        markersize=3.2,
        linewidth=1.1,
        label=legend_label,
    )
    ax.plot(
        CN_t,
        CN_u,
        color="#7b3294",
        marker="x",
        markevery=marker_stride,
        markersize=3.8,
        linewidth=1.1,
        linestyle="--",
        label="CN",
    )

    plotted_extent = max(
        a,
        b,
        float(np.max(np.abs(SE.u))),
        float(np.max(np.abs(CN_u))),
    )
    limit = 1.1 * plotted_extent
    ax.set_xlim(p - limit, p + limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"$a={a:g},\ b={b:g},\ h={h:g}$")
    ax.set_xlabel(r"$t$")
    ax.grid(color="0.85", linewidth=0.4)


def main() -> None:
    print("Elliptic exact-tracing calibration")

    figure, axes = plt.subplots(1, 2, figsize=(5.125, 2.6))
    plot_case(axes[0], a=2.26, b=2.26, p=TIME_SHIFT, h=STEP_SIZE)
    plot_case(axes[1], a=2.26, b=1.5, p=TIME_SHIFT, h=STEP_SIZE)
    axes[0].set_ylabel(r"$u$")

    common_radius = max(
        max(
            0.5 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            abs(ax.get_ylim()[0]),
            abs(ax.get_ylim()[1]),
        )
        for ax in axes
    )
    for ax in axes:
        ax.set_xlim(TIME_SHIFT - common_radius, TIME_SHIFT + common_radius)
        ax.set_ylim(-common_radius, common_radius)

    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            handles_by_label.setdefault(label, handle)
    legend_labels = [
        "Exact Ellipse",
        r"SE2 $(\sigma_n = 1)$",
        r"SE $\left(\sigma_n = \frac{b}{a}\right)$",
        "CN",
    ]
    legend = figure.legend(
        [handles_by_label[label] for label in legend_labels],
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.735, 0.5),
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
        left=0.075,
        right=0.72,
        bottom=0.19,
        top=0.86,
        wspace=0.20,
    )
    figure.savefig(
        os.path.join(figures_dir, "ellipse_exactness.pdf"),
        format="pdf",
        dpi=300,
    )
    # plt.show()


if __name__ == "__main__":
    main()
