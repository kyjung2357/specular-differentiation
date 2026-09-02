"""Plot the defect quantities cancelled by SE3 and SE4 for ``u' = -u^2``."""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCALES = np.geomspace(1.0e-2, 1.0e2, 401)
STEP_SIZE = 0.3
STEP_INDICES = (0, 1, 2)

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
    """Return the exact solution at ``t``."""

    return 1.0 / (1.0 + t)


def curvature_defect(
    sigma: np.ndarray | float,
    u: float,
) -> np.ndarray | float:
    r"""Return :math:`\mathcal{D}_\sigma(t,u)` for ``u' = -u^2``."""

    sigma_values = np.asarray(sigma)
    u_fourth = u**4
    return -6.0 * u_fourth + 12.0 * u_fourth**2 / (
        sigma_values**2 + u_fourth
    )


def pairwise_optimal_scale(x: float, y: float) -> float:
    """Return the unique SE4 optimal scale for two positive exact values."""

    p = x**4
    q = y**4
    squared_scale = (
        (p - q) ** 2
        + np.sqrt((p - q) ** 4 + 4.0 * p * q * (p + q) ** 2)
    ) / (2.0 * (p + q))
    return float(np.sqrt(squared_scale))


def main() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(5.125, 1.9), sharex=True)
    pointwise_ax, two_point_ax = axes
    colors = ("#fcae91", "#fb6a4a", "#de2d26")
    scale_colors = ("#6baed6", "#3182bd", "#08519c")
    legend_handles = []

    print("u' = -u^2, u(0) = 1: optimal-scale defect cancellation")
    print(
        f"{'n':>3} {'t_n':>10} {'u(t_n)':>12} "
        f"{'sigma_n^(3)':>14} {'sigma_n^(4)':>14}"
    )
    for n, color, scale_color in zip(
        STEP_INDICES,
        colors,
        scale_colors,
        strict=True,
    ):
        t_n = n * STEP_SIZE
        t_next = (n + 1) * STEP_SIZE
        u_n = exact(t_n)
        u_next = exact(t_next)
        sigma_third = u_n**2
        sigma_fourth = pairwise_optimal_scale(u_n, u_next)

        third_scales = np.unique(np.append(SCALES, sigma_third))
        third_values = np.abs(
            np.asarray(curvature_defect(third_scales, u_n), dtype=np.float64)
        )
        third_line, = pointwise_ax.semilogx(
            third_scales,
            third_values,
            color=color,
            linewidth=1.2,
            label=rf"$n={n}$",
            zorder=2,
        )
        pointwise_ax.axvline(
            sigma_third,
            color=scale_color,
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        legend_handles.append(third_line)

        fourth_scales = np.unique(np.append(SCALES, sigma_fourth))
        fourth_values = np.abs(
            np.asarray(curvature_defect(fourth_scales, u_n), dtype=np.float64)
            + np.asarray(
                curvature_defect(fourth_scales, u_next),
                dtype=np.float64,
            )
        )
        two_point_ax.semilogx(
            fourth_scales,
            fourth_values,
            color=color,
            linewidth=1.2,
            zorder=2,
        )
        two_point_ax.axvline(
            sigma_fourth,
            color=scale_color,
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )

        print(
            f"{n:3d} {t_n:10.3f} {u_n:12.8f} "
            f"{sigma_third:14.8f} {sigma_fourth:14.8f}"
        )

    pointwise_ax.set_title("SE3")
    two_point_ax.set_title("SE4")
    pointwise_ax.set_ylim(-0.15, 6.25)
    pointwise_ax.set_yticks((0.0, 1.5, 3.0, 4.5, 6.0))
    two_point_ax.set_ylim(-0.2, 8.5)
    two_point_ax.set_yticks((0.0, 2.0, 4.0, 6.0, 8.0))
    pointwise_ax.set_ylabel(r"$\mathsf{D}^{(3)}_n(\sigma)$")
    two_point_ax.set_ylabel(r"$\mathsf{D}^{(4)}_n(\sigma)$")

    for ax in axes:
        ax.set_xlim(float(SCALES[0]), float(SCALES[-1]))
        ax.set_xlabel(r"$\sigma$")
        ax.grid(color="0.85", linewidth=0.4, which="major")

    legend = figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.54),
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
        left=0.08,
        right=0.82,
        bottom=0.22,
        top=0.88,
        wspace=0.42,
    )
    figure.savefig(
        os.path.join(figures_dir, "quadratic_decay_defect_cancellation.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
