import os

import numpy as np


def plot_comparison(results, base_dir, filename, title, xlim, ylim, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

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
        "BFGS-D": "blue",
        "S-BFGS-D": "red",
    }
    linestyles = {
        "Dai sequence": "-",
        "SPEG": "-",
        "BFGS-E": ":",
        "BFGS-W": "--",
        "BFGS-A": "-",
        "S-BFGS-E": ":",
        "S-BFGS-W": "--",
        "S-BFGS-A": "-",
        "BFGS-D": "-.",
        "S-BFGS-D": "-.",
    }
    markers = {
        "Dai sequence": "x",
        "SPEG": "s",
        "BFGS-E": "o",
        "BFGS-W": "o",
        "BFGS-A": "o",
        "S-BFGS-E": "v",
        "S-BFGS-W": "v",
        "S-BFGS-A": "v",
        "BFGS-D": "o",
        "S-BFGS-D": "v",
    }

    def format_objective_tick(value, _):
        if abs(value) < 1e-12:
            return "0"
        sign = "-" if value < 0 else ""
        abs_value = abs(value)
        exponent = int(np.round(np.log10(abs_value)))
        if np.isclose(abs_value, 10.0**exponent):
            return rf"${sign}10^{{{exponent}}}$"
        return f"{value:g}"

    max_values_len = max((len(data["values"]) for data in results.values()), default=0)
    objective_values = np.concatenate(
        [np.asarray(data["values"], dtype=float) for data in results.values() if len(data["values"]) > 0]
    )

    plot_order = {
        "Dai sequence": 0,
        "SPEG": 1,
        "BFGS-E": 2,
        "BFGS-D": 3,
        "BFGS-A": 4,
        "S-BFGS-E": 5,
        "S-BFGS-D": 6,
        "S-BFGS-A": 7,
    }
    plot_items = sorted(results.items(), key=lambda item: plot_order.get(item[0], 100))

    fig_obj = plt.figure(figsize=(6, 4))
    ax1 = fig_obj.add_axes([0.18, 0.18, 0.58, 0.68])
    for name, data in plot_items:
        values = data["values"]
        color = colors.get(name, "black")
        linestyle = linestyles.get(name, "-")
        marker = markers.get(name, "o")
        zorder = 2 if name == "Dai sequence" else 3

        if len(values) > 0:
            ax1.plot(
                values,
                label=name,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2,
                linewidth=0.8,
                markevery=max(1, len(values) // 10),
                zorder=zorder,
            )

    ax1.set_xlabel(r"Iteration $k$", fontsize=8)
    ax1.set_ylabel(r"$f(\mathbf{x}_k)$", fontsize=8)
    ax1.set_yscale("symlog", linthresh=1e-2, linscale=0.6)
    y_max = max(1.0, float(np.nanmax(objective_values)))
    y_min = float(np.nanmin(objective_values))
    if y_min >= -1e-2:
        y_min = -1e-2
    ax1.set_ylim([y_min, 1.2 * y_max])
    ax1.yaxis.set_major_locator(FixedLocator([-1e-2, 0.0, 1e-2, 1e-1, 1.0, 10.0]))
    ax1.yaxis.set_major_formatter(FuncFormatter(format_objective_tick))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_title("Objective Function Value", fontsize=8)
    ax1.tick_params(axis="both", which="major", labelsize=8)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax1.set_xlim([-1, max(1, max_values_len)])
    ax1.grid(True, linewidth=0.5, zorder=0)
    ax1.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, fontsize=7, labelspacing=1.0)

    fig_obj.suptitle(title, fontsize=8)

    ext = "pdf" if pdf else "png"
    objective_path = os.path.join(figures_dir, f"{filename}_objective.{ext}")
    fig_obj.savefig(objective_path, dpi=1000, bbox_inches="tight")
    print(f"Saved objective figure to: {objective_path}")

    fig_traj = plt.figure(figsize=(3.4, 2.7))
    ax2 = fig_traj.add_axes([0.16, 0.18, 0.58, 0.68])
    for name, data in plot_items:
        variables = data["variables"]
        color = colors.get(name, "black")
        linestyle = linestyles.get(name, "-")
        marker = markers.get(name, "o")
        zorder = 2 if name == "Dai sequence" else 3

        if len(variables) > 0:
            ax2.plot(
                variables[:, 0],
                variables[:, 1],
                label=name,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2,
                linewidth=0.8,
                markevery=max(1, len(variables) // 10),
                zorder=zorder,
            )

    ax2.set_xlabel(r"$x_1$", fontsize=8)
    ax2.set_ylabel(r"$x_2$", fontsize=8)
    ax2.set_title(r"Top View Trajectory ($x_1x_2$-plane)", fontsize=8)
    ax2.tick_params(axis="both", which="major", labelsize=8)
    ax2.grid(True, linewidth=0.5, zorder=0)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.legend(loc="center left", bbox_to_anchor=(1.35, 0.5), borderaxespad=0.0, fontsize=7, labelspacing=1.0)

    fig_traj.suptitle(title, fontsize=8)

    trajectory_path = os.path.join(figures_dir, f"{filename}_trajectory.{ext}")
    fig_traj.savefig(trajectory_path, dpi=1000, bbox_inches="tight")
    print(f"Saved trajectory figure to: {trajectory_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_obj)
        plt.close(fig_traj)


def plot_curvature_diagnostics(diagnostics, base_dir, filename, title, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    colors = {
        "BFGS-D": "blue",
        "S-BFGS-D": "red",
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
        "BFGS-D": "o",
        "S-BFGS-D": "v",
        "BFGS-W": "o",
        "S-BFGS-W": "v",
    }

    ext = "pdf" if pdf else "png"

    fig_ys = plt.figure(figsize=(4.2, 2.6))
    ax_ys = fig_ys.add_axes([0.18, 0.18, 0.58, 0.68])
    has_ys = False

    for name, info in diagnostics.items():
        ys = np.asarray(info.get("ys", []), dtype=float)
        if ys.size == 0:
            continue

        x_data = np.arange(1, ys.size + 1)
        ax_ys.plot(
            x_data,
            ys,
            label=name,
            color=colors.get(name, "black"),
            marker=markers.get(name, "o"),
            markersize=2,
            linewidth=0.9,
        )
        has_ys = True

    ax_ys.axhline(0.0, color="black", linewidth=0.6)
    ax_ys.set_xlabel(r"Iteration $k$", fontsize=8)
    ax_ys.set_ylabel(r"$y_k^\top s_k$", fontsize=8)
    ax_ys.set_yscale("symlog", linthresh=1e-12)
    ax_ys.set_title("Curvature Quantity", fontsize=8)
    ax_ys.grid(True, linewidth=0.5)
    ax_ys.tick_params(axis="both", which="major", labelsize=8)
    ax_ys.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    if has_ys:
        ax_ys.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, fontsize=7)

    fig_ys.suptitle(title, fontsize=8)
    ys_path = os.path.join(figures_dir, f"{filename}_curvature.{ext}")
    fig_ys.savefig(ys_path, dpi=1000, bbox_inches="tight")
    print(f"Saved curvature figure to: {ys_path}")

    fig_norm = plt.figure(figsize=(4.2, 2.6))
    ax_norm = fig_norm.add_axes([0.18, 0.18, 0.58, 0.68])
    has_norm = False

    for name, info in diagnostics.items():
        curvature = np.asarray(info.get("normalized_curvature", []), dtype=float)
        if curvature.size == 0:
            continue

        x_data = np.arange(1, curvature.size + 1)
        ax_norm.plot(
            x_data,
            curvature,
            label=name,
            color=colors.get(name, "black"),
            marker=markers.get(name, "o"),
            markersize=2,
            linewidth=0.9,
        )
        has_norm = True

    ax_norm.axhline(0.0, color="black", linewidth=0.6)
    ax_norm.set_xlabel(r"Iteration $k$", fontsize=8)
    ax_norm.set_ylabel(r"$\frac{y_k^\top s_k}{\|y_k\|\|s_k\|}$", fontsize=8)
    ax_norm.set_ylim([-1.1, 1.1])
    ax_norm.set_title("Normalized Curvature", fontsize=8)
    ax_norm.grid(True, linewidth=0.5)
    ax_norm.tick_params(axis="both", which="major", labelsize=8)
    ax_norm.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    if has_norm:
        ax_norm.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, fontsize=7)

    fig_norm.suptitle(title, fontsize=8)
    norm_path = os.path.join(figures_dir, f"{filename}_normalized_curvature.{ext}")
    fig_norm.savefig(norm_path, dpi=1000, bbox_inches="tight")
    print(f"Saved normalized curvature figure to: {norm_path}")

    if show:
        plt.show()
    else:
        plt.close(fig_ys)
        plt.close(fig_norm)


def plot_theoretical_sequence(x_seq, f_seq, u1_points, base_dir, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(f_seq, label="Dai theoretical sequence", marker="o", markersize=4, linestyle="-")
    ax1.set_xlabel(r"Iteration $k$", fontsize=12)
    ax1.set_ylabel(r"$f(\mathbf{x}_k)$", fontsize=12)
    ax1.set_yscale("symlog")
    ax1.set_title("Objective Function Value (Theoretical)", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    ax2.plot(x_seq[:, 0], x_seq[:, 1], label="Dai sequence", marker="o", markersize=4, linestyle="-", alpha=0.7)
    ax2.plot(u1_points, np.zeros_like(u1_points), "rx", label=r"Convergence axis ($x_2=0$)", alpha=0.5)
    ax2.set_xlabel(r"$x_1$", fontsize=12)
    ax2.set_ylabel(r"$x_2$", fontsize=12)
    ax2.set_title(r"Top View Trajectory ($x_1x_2$-plane)", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True)
    ax2.set_xlim([-100, 100])
    ax2.set_ylim([-10, 20])

    plt.tight_layout()
    ext = "pdf" if pdf else "png"
    out_path = os.path.join(figures_dir, f"dai_theoretical_sequence.{ext}")
    plt.savefig(out_path, dpi=1000)
    print(f"Saved theoretical figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()
