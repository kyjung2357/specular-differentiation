import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "Times New Roman"


def ensure_length(data, length):
    data = list(data)
    if len(data) == 0:
        return []
    if len(data) < length:
        return data + [data[-1]] * (length - len(data))
    return data[:length]


def format_sci_latex(x):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    s = "{:.4e}".format(x)
    base, exponent = s.split("e")
    return fr"${base} \times 10^{{{int(exponent)}}}$"


def report_results(
    all_results,
    running_times,
    failure_counts,
    file_number,
    m,
    n,
    lambda1,
    lambda2,
    iteration,
    base_dir,
    pdf=False,
    show=False,
):
    colors = {
        "S-BFGS-A": "#de2d26",
        "S-BFGS-W": "#fb6a4a",
        "S-BFGS-S": "#a50f15",
    }
    linestyles = {
        "S-BFGS-A": "-",
        "S-BFGS-W": "--",
        "S-BFGS-S": ":",
    }

    summary_stats = {}
    plt.figure(figsize=(6, 3))

    for name, results_list in all_results.items():
        if results_list:
            df = pd.DataFrame(results_list).T
            df.columns = [f"trial_{j + 1}" for j in range(len(results_list))]

            min_vals = df.min(axis=0)
            mean_curve = df.mean(axis=1)
            median_curve = df.median(axis=1)

            x_data = df.index + 1
            color = colors.get(name, "black")

            plt.plot(
                x_data,
                mean_curve,
                label=name,
                color=color,
                linewidth=1.5,
                linestyle=linestyles.get(name, "-"),
            )
            plt.plot(
                x_data,
                median_curve,
                color=color,
                linestyle="--",
                alpha=0.45,
                linewidth=1,
            )

            mean_best = min_vals.mean()
            median_best = min_vals.median()
            std_best = min_vals.std()
        else:
            plt.plot(
                [],
                [],
                label=name,
                color=colors.get(name, "black"),
                linewidth=1.5,
                linestyle=linestyles.get(name, "-"),
            )
            mean_best = np.nan
            median_best = np.nan
            std_best = np.nan

        summary_stats[name] = {
            "Success": len(results_list),
            "Failure": failure_counts.get(name, 0),
            "Mean": mean_best,
            "Median": median_best,
            "Standard deviation": std_best,
            "Average runtime": np.mean(running_times[name]) if running_times.get(name) else np.nan,
        }

    print("\n[Line Search Summary]")
    summary_df = pd.DataFrame(summary_stats).T
    print(summary_df)

    display_df = summary_df.copy()
    for col in ["Mean", "Median", "Standard deviation", "Average runtime"]:
        display_df[col] = display_df[col].apply(format_sci_latex)

    os.makedirs(os.path.join(base_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)

    suffix = f"{file_number}-{m}-{n}-{lambda1}-{lambda2}"
    path_txt = os.path.join(base_dir, f"tables/table{suffix}.txt")
    path_fig = os.path.join(base_dir, f"figures/figure{suffix}.{'pdf' if pdf else 'png'}")

    with open(path_txt, "w", encoding="utf-8") as table_file:
        table_file.write(display_df.to_latex(escape=False))

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(\mathbf{x}_k)$", fontsize=10)
    plt.grid(True)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(1, iteration)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=10)
    plt.tight_layout()
    plt.savefig(path_fig, dpi=1000, bbox_inches="tight")
    print(f"Saved table to: {path_txt}")
    print(f"Saved figure to: {path_fig}")

    if show:
        plt.show()
