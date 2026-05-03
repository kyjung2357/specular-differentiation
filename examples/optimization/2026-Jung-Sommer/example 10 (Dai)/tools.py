import os

import numpy as np


def plot_comparison(results, base_dir, filename, title, xlim, ylim, pdf=False, show=False):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    colors = {
        "BFGS-E": "#08306b",
        "BFGS-S": "#08519c",
        "BFGS-W": "#6baed6",
        "BFGS-A": "#3182bd",
        "S-BFGS-E": "#67000d",
        "S-BFGS-S": "#a50f15",
        "S-BFGS-W": "#fb6a4a",
        "S-BFGS-A": "#de2d26",
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.125, 2.1))

    for name, data in results.items():
        values = data["values"]
        variables = data["variables"]
        color = colors.get(name, "black")
        marker = markers.get(name, "o")

        if len(values) > 0:
            ax1.plot(values, label=name, color=color, marker=marker, markersize=2, linewidth=0.8)

        if len(variables) > 0:
            ax2.plot(variables[:, 0], variables[:, 1], label=name, color=color, marker=marker, markersize=2, linewidth=0.8)

    ax1.set_xlabel(r"Iteration $k$", fontsize=8)
    ax1.set_ylabel(r"$f(\mathbf{x}_k)$", fontsize=8)
    ax1.set_yscale("symlog")
    ax1.set_title("Objective Function Value", fontsize=8)
    ax1.tick_params(axis="both", which="major", labelsize=8)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    ax1.set_xlim([-2, 52])
    ax1.grid(True, linewidth=0.5)

    ax2.set_xlabel(r"$x_1$", fontsize=8)
    ax2.set_ylabel(r"$x_2$", fontsize=8)
    ax2.set_title(r"Top View Trajectory ($x_1x_2$-plane)", fontsize=8)
    ax2.tick_params(axis="both", which="major", labelsize=8)
    ax2.grid(True, linewidth=0.5)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.legend(loc="center left", bbox_to_anchor=(1.35, 0.5), borderaxespad=0.0, fontsize=7, labelspacing=1.0)

    plt.suptitle(title, fontsize=8)
    plt.tight_layout(pad=0.2, w_pad=4.0)

    ext = "pdf" if pdf else "png"
    out_path = os.path.join(figures_dir, f"{filename}.{ext}")
    plt.savefig(out_path, dpi=1000, bbox_inches="tight")
    print(f"Saved comparison figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close()


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
