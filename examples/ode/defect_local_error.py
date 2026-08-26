"""Plot the curvature-defect profile for ``u' = -u^2``."""

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


def main() -> None:
    figure, ax = plt.subplots(figsize=(5.125, 1.5))
    colors = ("#de2d26", "#fb6a4a", "#fcae91")
    scale_colors = ("#08519c", "#3182bd", "#6baed6")
    defect_handles = []
    scale_handles = []

    print("u' = -u^2, u(0) = 1: absolute curvature-defect profiles")
    print(f"{'n':>3} {'t_n':>10} {'u(t_n)':>12} {'sigma_n':>12}")
    for n, color, scale_color in zip(
        STEP_INDICES,
        colors,
        scale_colors,
        strict=True,
    ):
        t_n = n * STEP_SIZE
        u_n = exact(t_n)
        sigma_star = u_n * u_n
        profile_scales = np.unique(np.append(SCALES, sigma_star))
        defect_values = np.abs(
            np.asarray(curvature_defect(profile_scales, u_n), dtype=np.float64)
        )

        defect_line, = ax.semilogx(
            profile_scales,
            defect_values,
            color=color,
            linewidth=1.2,
            label=rf"$n={n}$",
            zorder=2,
        )
        scale_line = ax.axvline(
            sigma_star,
            color=scale_color,
            linestyle="--",
            linewidth=0.9,
            label=rf"$\sigma_{n}$",
            zorder=1,
        )
        defect_handles.append(defect_line)
        scale_handles.append(scale_line)
        print(f"{n:3d} {t_n:10.3f} {u_n:12.8f} {sigma_star:12.8f}")

    ax.set_xlim(float(SCALES[0]), float(SCALES[-1]))
    ax.set_ylim(-0.15, 6.25)
    ax.set_yticks((0.0, 1.5, 3.0, 4.5, 6.0))
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$|\mathcal{D}_\sigma(t_n,u(t_n))|$")
    ax.grid(color="0.85", linewidth=0.4, which="major")
    legend = ax.legend(
        handles=defect_handles + scale_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="white",
        framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.6)

    figure.subplots_adjust(
        left=0.13,
        right=0.78,
        bottom=0.18,
        top=0.97,
    )
    figure.savefig(
        os.path.join(figures_dir, "defect_local_error.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
