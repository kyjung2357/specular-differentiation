import os

import numpy as np


def ensure_length(data, length):
    data = np.asarray(data, dtype=float)

    if data.size == 0:
        return data

    if len(data) < length:
        tail = np.repeat(data[-1:], length - len(data), axis=0)
        return np.concatenate([data, tail], axis=0)

    return data[:length]


def print_summary(results):
    print("\n[Summary]")
    for name, data in results.items():
        values = np.asarray(data["values"], dtype=float)
        final_value = values[-1] if len(values) else np.nan
        min_value = np.nanmin(values) if len(values) else np.nan
        runtime = data.get("runtime", np.nan)
        stop_reason = data.get("stop_reason", "")
        print(
            f"{name:12s}: final={final_value:.6e}, "
            f"min={min_value:.6e}, runtime={runtime:.5f} sec, stop={stop_reason}"
        )


def plot_comparison(results, base_dir, filename, title, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    colors = {
        "Dai sequence": "#000000",
        "SPEG": "black",
        "BFGS-E": "#08306b",
        "BFGS-S": "#08519c",
        "BFGS-W": "#2171b5",
        "BFGS-A": "#6baed6",
        "S-BFGS-E": "#67000d",
        "S-BFGS-S": "#a50f15",
        "S-BFGS-W": "#de2d26",
        "S-BFGS-A": "#fb6a4a",
    }
    linestyles = {
        "Dai sequence": "--",
        "SPEG": "-",
        "BFGS-E": "-",
        "BFGS-S": "-",
        "BFGS-W": "-",
        "BFGS-A": "-",
        "S-BFGS-E": "-",
        "S-BFGS-S": "-",
        "S-BFGS-W": "-",
        "S-BFGS-A": "-",
    }
    markers = {
        "Dai sequence": "x",
        "SPEG": "s",
        "BFGS-E": "o",
        "BFGS-S": "o",
        "BFGS-W": "o",
        "BFGS-A": "o",
        "S-BFGS-E": "v",
        "S-BFGS-S": "v",
        "S-BFGS-W": "v",
        "S-BFGS-A": "v",
    }
    plot_order = {
        "Dai sequence": 0,
        "SPEG": 1,
        "BFGS-E": 2,
        "BFGS-S": 3,
        "BFGS-W": 4,
        "BFGS-A": 5,
        "S-BFGS-E": 6,
        "S-BFGS-S": 7,
        "S-BFGS-W": 8,
        "S-BFGS-A": 9,
    }

    plot_items = sorted(results.items(), key=lambda item: plot_order.get(item[0], 100))

    fig_obj = plt.figure(figsize=(6.0, 3.0))
    ax_obj = fig_obj.add_axes([0.15, 0.2, 0.58, 0.68])

    for name, data in plot_items:
        values = np.asarray(data["values"], dtype=float)

        if values.size == 0:
            continue

        ax_obj.plot(
            np.arange(values.size),
            values,
            label=name,
            color=colors.get(name, "black"),
            linestyle=linestyles.get(name, "-"),
            marker=markers.get(name, "o"),
            markersize=2,
            linewidth=1.0,
            markevery=max(1, values.size // 12),
        )

    ax_obj.set_xlabel(r"Iteration $k$", fontsize=10)
    ax_obj.set_ylabel(r"$f(\mathbf{x}_k)$", fontsize=10)
    ax_obj.set_title("Objective Function Value", fontsize=10)
    ax_obj.set_yscale("symlog", linthresh=1e-4, linscale=0.7)
    ax_obj.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax_obj.tick_params(axis="both", which="major", labelsize=9)
    ax_obj.grid(True, linewidth=0.5)
    ax_obj.legend(
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        borderaxespad=0.0,
        fontsize=9,
        labelspacing=0.8,
    )
    fig_obj.suptitle(title, fontsize=11)

    ext = "pdf" if pdf else "png"
    objective_path = os.path.join(figures_dir, f"{filename}_objective.{ext}")
    fig_obj.savefig(objective_path, dpi=1000, bbox_inches="tight")
    print(f"Saved objective figure to: {objective_path}")

    fig_traj = plt.figure(figsize=(4.0, 3.2))
    ax_traj = fig_traj.add_axes([0.16, 0.18, 0.58, 0.68])

    for name, data in plot_items:
        variables = np.asarray(data["variables"], dtype=float)

        if variables.size == 0:
            continue

        ax_traj.plot(
            variables[:, 0],
            variables[:, 1],
            label=name,
            color=colors.get(name, "black"),
            linestyle=linestyles.get(name, "-"),
            marker=markers.get(name, "o"),
            markersize=2,
            linewidth=1.0,
            markevery=max(1, len(variables) // 12),
        )

    ax_traj.set_xlabel(r"$x_1$", fontsize=10)
    ax_traj.set_ylabel(r"$x_2$", fontsize=10)
    ax_traj.set_title(r"Top View Trajectory ($x_1x_2$-plane)", fontsize=10)
    ax_traj.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_traj.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_traj.tick_params(axis="both", which="major", labelsize=9)
    ax_traj.grid(True, linewidth=0.5)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.legend(
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        borderaxespad=0.0,
        fontsize=9,
        labelspacing=0.8,
    )
    fig_traj.suptitle(title, fontsize=11)

    trajectory_path = os.path.join(figures_dir, f"{filename}_trajectory.{ext}")
    fig_traj.savefig(trajectory_path, dpi=1000, bbox_inches="tight")
    print(f"Saved trajectory figure to: {trajectory_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_obj)
        plt.close(fig_traj)
