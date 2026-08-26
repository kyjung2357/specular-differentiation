"""Compare SE and Crank--Nicolson trajectories on two ellipses."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import specular

from _common import crank_nicolson, maximum_error

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

def _upper_arc_mesh(a: float, h: float) -> tuple[float, float, int]:
    """Return an ``h``-mesh just inside the singular ellipse endpoints."""

    t_0 = -a + 0.01
    T = a - 0.01
    n_steps = round((T - t_0) / h)
    return t_0, T, n_steps


def plot_case(ax: Axes, a: float, b: float, h: float) -> None:
    """Plot the fitted SE and Crank--Nicolson trajectories for one ellipse."""

    def F(t: float, u: float) -> float:
        return -(b * b * t) / (a * a * u)

    def exact(t: np.ndarray) -> np.ndarray:
        return b * np.sqrt(np.maximum(0.0, 1.0 - (t / a) ** 2))

    t_0, T, n_steps = _upper_arc_mesh(a, h)
    u_0 = float(exact(np.asarray(t_0)))

    SE = specular.ellipse_scheme(
        F,
        t_0,
        T,
        u_0,
        n_steps=n_steps,
        sigma_n=b / a,
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
        np.max(np.abs((SE.t / a) ** 2 + (SE.u / b) ** 2 - 1.0))
    )
    marker_stride = max(1, n_steps // 10)

    print(
        f"a={a:g}, b={b:g}, h={h:g}: "
        f"SE error={SE_error:.3e}, "
        f"CN error={CN_error:.3e}, "
        f"SE ellipse residual={ellipse_residual:.3e}"
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 1000)
    ax.plot(
        a * np.cos(theta),
        b * np.sin(theta),
        color="0.6",
        linestyle="--",
        linewidth=1.0,
        label="Exact trajectory",
    )
    ax.plot(
        SE.t,
        SE.u,
        color="tab:red",
        marker="o",
        markevery=marker_stride,
        markersize=3.2,
        linewidth=1.1,
        label="SE",
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
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(rf"$a={a:g},\ b={b:g},\ h={h:g}$")
    ax.set_xlabel(r"$t$")
    ax.grid(color="0.85", linewidth=0.4)


def main() -> None:
    print("Elliptic exact-tracing calibration")

    figure, axes = plt.subplots(1, 2, figsize=(5.125, 2.55))
    plot_case(axes[0], a=2.3, b=2.3, h=0.3)
    plot_case(axes[1], a=2.3, b=1.5, h=0.3)
    axes[0].set_ylabel(r"$u$")

    common_limit = max(
        abs(value)
        for ax in axes
        for limits in (ax.get_xlim(), ax.get_ylim())
        for value in limits
    )
    for ax in axes:
        ax.set_xlim(-common_limit, common_limit)
        ax.set_ylim(-common_limit, common_limit)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        frameon=True,
        facecolor="white",
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.6)
    figure.subplots_adjust(
        left=0.095,
        right=0.765,
        bottom=0.19,
        top=0.86,
        wspace=0.20,
    )
    figure.savefig(
        os.path.join(figures_dir, "ellipse_exactness.pdf"),
        format="pdf",
        dpi=300,
        bbox_inches='tight'
    )
    # plt.show()


if __name__ == "__main__":
    main()
