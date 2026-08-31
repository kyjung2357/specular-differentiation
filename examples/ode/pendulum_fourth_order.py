"""Compare SE2, SE3, and SE4 on normalized pendulum branches."""

from __future__ import annotations

import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import specular

from _common import maximum_error, rk4, uniform_step_count


# The SE4 scale selector starts in E5a, then follows E5b.
X_LEFT = 0.0
X_RIGHT = 0.8
STEP_SIZES = np.array(
    (
        1.0e-1,
        5.714285714285715e-2,
        3.2e-2,
        1.7777777777777778e-2,
        1.0e-2,
        5.633802816901409e-3,
        3.1620553359683794e-3,
        1.7777777777777779e-3,
        1.0e-3,
        5.621925509487e-4,
        3.1620553359683795e-4,
        1.7777777777777778e-4,
        1.0e-4,
    ),
    dtype=np.float64,
)
AMPLITUDES = (1.0, 0.25, 0.1, 0.01)

METHOD_STYLES = (
    (r"SE2", "fixed", "#fcbba1", "s", "-"),
    (r"SE3", "third", "#ef3b2c", "o", "-"),
    (
        r"SE4",
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


def exact_solution(amplitude: float, x: np.ndarray) -> np.ndarray:
    left = np.sin(0.5 * amplitude * (1.0 + x))
    right = np.sin(0.5 * amplitude * (1.0 - x))
    return (2.0 / amplitude) * np.sqrt(np.maximum(left * right, 0.0))


def field(amplitude: float):
    def F(x: float, y: float) -> float:
        return -math.sin(amplitude * x) / (amplitude * y)

    return F


def field_derivatives(amplitude: float):
    def derivatives(point: np.ndarray) -> np.ndarray:
        x, y = map(float, point)
        s = math.sin(amplitude * x)
        c = math.cos(amplitude * x)
        first = -c / y - s * s / (amplitude**2 * y**3)
        second = (
            amplitude * s / y
            - 3.0 * s * c / (amplitude * y**3)
            - 3.0 * s**3 / (amplitude**3 * y**5)
        )
        return np.array([first, second])

    return derivatives


def run_se(amplitude: float, h: float, mode: str) -> float:
    exact = lambda x: exact_solution(amplitude, x)
    y_0 = float(exact(np.array(X_LEFT)))
    n_steps = uniform_step_count(X_LEFT, X_RIGHT, h)

    common_options = {
        "n_steps": n_steps,
        "atol": 0.0,
        "rtol": 2e-16,
        "max_iter": 2000,
    }
    if mode == "fixed":
        result = specular.ellipse_scheme(
            field(amplitude),
            X_LEFT,
            X_RIGHT,
            y_0,
            sigma_n=1.0,
            **common_options,
        )
    elif mode == "third":
        result = specular.ellipse_scheme(
            field(amplitude),
            X_LEFT,
            X_RIGHT,
            y_0,
            third_order=True,
            derivatives_of_F=field_derivatives(amplitude),
            **common_options,
        )
    elif mode == "fourth":
        result = specular.ellipse_scheme(
            field(amplitude),
            X_LEFT,
            X_RIGHT,
            y_0,
            fourth_order=True,
            derivatives_of_F=field_derivatives(amplitude),
            **common_options,
        )
    else:
        raise ValueError(f"unknown SE mode: {mode}")

    return maximum_error(result.t, result.u, exact)


def run_rk4(amplitude: float, h: float) -> float:
    exact = lambda x: exact_solution(amplitude, x)
    y_0 = float(exact(np.array(X_LEFT)))
    n_steps = uniform_step_count(X_LEFT, X_RIGHT, h)
    t, y = rk4(field(amplitude), X_LEFT, X_RIGHT, y_0, n_steps)
    return maximum_error(t, y, exact)


def run_method(amplitude: float, h: float, method: str) -> float:
    if method in {"fixed", "third", "fourth"}:
        return run_se(amplitude, h, method)
    if method == "rk4":
        return run_rk4(amplitude, h)
    raise ValueError(f"unknown method: {method}")


def main() -> None:
    print("Normalized pendulum: SE2, SE3, SE4, and RK4")
    print(f"{'theta':>5} {'method':>36} {'error at h=1e-4':>20}")

    plot_data: list[tuple[float, dict[str, np.ndarray]]] = []
    for amplitude in AMPLITUDES:
        errors_by_method: dict[str, np.ndarray] = {}
        for label, method, *_ in METHOD_STYLES:
            errors = np.array(
                [
                    run_method(amplitude, float(h), method)
                    for h in STEP_SIZES
                ],
                dtype=np.float64,
            )
            errors_by_method[method] = errors
            print(f"{amplitude:5.2f} {label:>36} {errors[-1]:20.6e}")
        plot_data.append((amplitude, errors_by_method))

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(5.125, 3.35),
        sharex=True,
        sharey=True,
    )
    for ax, (amplitude, errors_by_method) in zip(
        axes.flat,
        plot_data,
        strict=True,
    ):
        for label, method, color, marker, linestyle in METHOD_STYLES:
            ax.loglog(
                STEP_SIZES,
                errors_by_method[method],
                color=color,
                marker=marker,
                markersize=2.8,
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
        ax.set_title(rf"$\theta={amplitude:g}$")
        ax.grid(color="0.85", linewidth=0.4, which="major")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$h$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Maximum global error")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.985, 0.5),
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
        left=0.14,
        right=0.82,
        bottom=0.13,
        top=0.95,
        wspace=0.18,
        hspace=0.28,
    )
    figure.savefig(
        os.path.join(figures_dir, "pendulum_fourth_order.pdf"),
        format="pdf",
        dpi=300,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
