import os

import numpy as np


def format_symlog_tick(value, _position=None):
    if value == 0:
        return "0"

    sign = "-" if value < 0 else ""
    exponent = int(round(np.log10(abs(value))))
    return rf"${sign}10^{{{exponent}}}$"


def symlog_major_ticks(y_min, y_max):
    max_negative_power = int(np.ceil(np.log10(abs(y_min)))) if y_min < -1 else 0
    max_positive_power = int(np.ceil(np.log10(y_max))) if y_max > 1 else 0

    negative_powers = even_powers(max_negative_power, min_power=2)
    positive_powers = even_powers(max_positive_power, min_power=0)

    ticks = [-(10.0**p) for p in negative_powers]
    ticks.append(0.0)
    ticks.extend(10.0**p for p in sorted(set(positive_powers)))

    return [tick for tick in ticks if y_min <= tick <= y_max]


def even_powers(max_power, min_power):
    if max_power < min_power:
        return []

    first_power = max_power if max_power % 2 == 0 else max_power - 1
    return list(range(first_power, min_power - 1, -2))


def sparse_powers(max_power, max_count):
    if max_power <= 0:
        return []

    if max_power <= max_count:
        return list(range(max_power, 0, -1))

    powers = np.linspace(max_power, 1, max_count)
    return sorted({int(round(power)) for power in powers}, reverse=True)


def symlog_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 10.0

    min_value = float(np.min(values))
    max_value = float(np.max(values))

    if min_value < 0.0:
        negative_power = int(np.ceil(np.log10(abs(min_value))))
        y_min = -(10.0**negative_power)
    else:
        y_min = -1.0

    if max_value > 0.0:
        positive_power = int(np.ceil(np.log10(max_value)))
        y_max = max(10.0, 10.0**max(0, positive_power))
    else:
        y_max = 10.0

    return y_min, y_max


def plot_comparison(results, base_dir, filename, title, xlim, ylim, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullLocator

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    colors = {
        "BFGS-E": "#08306b",
        "BFGS-S": "#08519c",
        "BFGS-W": "#2171b5",
        "BFGS-A": "#6baed6",
        "S-BFGS-E": "#67000d",
        "S-BFGS-S": "#a50f15",
        "S-BFGS-W": "#de2d26",
        "S-BFGS-A": "#fb6a4a",
    }
    markers = {
        "BFGS-E": "o",
        "BFGS-S": "o",
        "BFGS-W": "o",
        "BFGS-A": "o",
        "S-BFGS-E": "v",
        "S-BFGS-S": "v",
        "S-BFGS-W": "v",
        "S-BFGS-A": "v",
    }

    ext = "pdf" if pdf else "png"
    objective_values = []
    max_iteration = 0
    fig_obj, ax1 = plt.subplots(figsize=(4.4, 2.2))

    for name, data in results.items():
        values = data["values"]
        color = colors.get(name, "black")
        marker = markers.get(name, "o")

        if len(values) > 0:
            objective_values.extend(values)
            max_iteration = max(max_iteration, len(values) - 1)
            ax1.plot(values, label=name, color=color, marker=marker, markersize=2, linewidth=0.8, markevery=5)

    ax1.set_xlabel(r"Iteration $k$", fontsize=8)
    ax1.set_ylabel(r"$f(\mathbf{x}_k)$", fontsize=8)
    ax1.set_yscale("symlog", linthresh=1.0)
    y_min, y_max = ax1.get_ylim()
    ax1.yaxis.set_major_locator(FixedLocator(symlog_major_ticks(y_min, y_max)))
    ax1.yaxis.set_major_formatter(FuncFormatter(format_symlog_tick))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_title("Objective Function Value", fontsize=8)
    ax1.tick_params(axis="both", which="major", labelsize=8)
    ax1.tick_params(axis="y", pad=5)
    ax1.xaxis.set_major_locator(MultipleLocator(iteration_tick_step(max_iteration)))
    ax1.xaxis.set_minor_locator(NullLocator())
    ax1.set_xlim([-2, max_iteration + 2])
    ax1.grid(True, linewidth=0.5)
    ax1.set_box_aspect(0.72)
    ax1.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, fontsize=7, labelspacing=0.7)
    fig_obj.tight_layout(pad=0.2)

    objective_path = os.path.join(figures_dir, f"{filename}_objective.{ext}")
    fig_obj.savefig(objective_path, dpi=1000, bbox_inches="tight")
    print(f"Saved objective figure to: {objective_path}")

    fig_traj, axes = plt.subplots(2, 4, figsize=(8.6, 4.2), sharex=True, sharey=True)
    axes = axes.ravel()
    legend_handles = []
    legend_labels = []

    for ax, (name, data) in zip(axes, results.items()):
        variables = data["variables"]
        color = colors.get(name, "black")
        marker = markers.get(name, "o")

        if len(variables) > 0:
            line, = ax.plot(
                variables[:, 0],
                variables[:, 1],
                color=color,
                marker=marker,
                markersize=2,
                linewidth=0.8,
            )
            legend_handles.append(line)
            legend_labels.append(name)

        ax.set_title(name, fontsize=8, pad=3)
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_box_aspect(1)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    for row in range(2):
        axes[4 * row].set_ylabel(r"$y$", fontsize=8)

    for col in range(4):
        axes[4 + col].set_xlabel(r"$x$", fontsize=8)

    fig_traj.suptitle(r"Top View Trajectories ($xy$-plane)", fontsize=9)
    fig_traj.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.83, 0.5),
        borderaxespad=0.0,
        fontsize=8,
        labelspacing=0.8,
    )
    fig_traj.subplots_adjust(
        left=0.07,
        right=0.80,
        bottom=0.10,
        top=0.90,
        wspace=0.20,
        hspace=0.35,
    )

    trajectory_path = os.path.join(figures_dir, f"{filename}_trajectories.{ext}")
    fig_traj.savefig(trajectory_path, dpi=1000, bbox_inches="tight")
    print(f"Saved trajectory figure to: {trajectory_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_obj)
        plt.close(fig_traj)


def iteration_tick_step(max_iteration):
    if max_iteration <= 10:
        return 2
    if max_iteration <= 50:
        return 10
    if max_iteration <= 100:
        return 20
    if max_iteration <= 250:
        return 50
    if max_iteration <= 500:
        return 100
    if max_iteration <= 1000:
        return 200
    return 500


def plot_reference_iterates(base_dir, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    k_vals = np.arange(25)
    x_vals = np.array([get_reference_iterate(k) for k in k_vals])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax1.plot(x_vals[:, 0], x_vals[:, 1], "bo-", markersize=4, label="Iterates", linewidth=1)
    ax1.set_title("Top View (xy-plane)")
    ax1.set_xlabel(r"$x$", fontsize=10)
    ax1.set_ylabel(r"$y$", fontsize=10)
    ax1.grid(True)
    ax1.set_aspect("equal", adjustable="datalim")

    ax2.plot(k_vals, x_vals[:, 2], "ro-", markersize=4, label="z-coordinate", linewidth=1)
    ax2.set_title("Side View (z-coordinate)")
    ax2.set_xlabel(r"Iteration $k$", fontsize=10)
    ax2.set_ylabel(r"$z$", fontsize=10)
    ax2.grid(True)

    plt.tight_layout()
    ext = "pdf" if pdf else "png"
    out_path = os.path.join(figures_dir, f"mascarenhas_2d_views.{ext}")
    plt.savefig(out_path, dpi=1000, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved reference figure to: {out_path}")


def get_reference_iterate(k):
    x_infty = np.array([3.0 + 2.0 * np.sqrt(2.0), 1.0 + np.sqrt(2.0), 0.0]) / 2.0
    e_z = np.array([0.0, 0.0, 1.0])
    Q = np.array(
        [
            [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0), 0.0],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    X = np.diag([1.0, 1.0, 1.0 / np.sqrt(2.0)])
    return np.linalg.matrix_power(X @ Q, k) @ (x_infty + e_z)
