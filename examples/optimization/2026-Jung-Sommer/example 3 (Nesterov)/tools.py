import os

import numpy as np
import pandas as pd


def ensure_length(data, length):
    data = list(data)
    if len(data) == 0:
        return [0.0] * length
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


def format_param(value):
    return f"{value:g}".replace(".", "p")


def format_percent_latex(x):
    if pd.isna(x):
        return "--"
    return f"{x:.1f}"


def report_results(
    all_results,
    running_times,
    file_number,
    n,
    p,
    q,
    function_label,
    trials,
    iteration,
    base_dir,
    pdf=False,
    show=False,
):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    colors = {
        "SPEG": "black",
        "SPEG-s": "red",
        "SPEG-g": "brown",
        "S-SPEG": "blue",
        "H-SPEG": "purple",
        "S-BFGS": "darkorange",
        "BFGS-E": "#08306b",
        "BFGS-S": "#08519c",
        "BFGS-W": "#2171b5",
        "BFGS-A": "#6baed6",
        "S-BFGS-E": "#67000d",
        "S-BFGS-S": "#a50f15",
        "S-BFGS-W": "#de2d26",
        "S-BFGS-A": "#fb6a4a",
        "GD": "orange",
        "Adam": "green"
    }

    summary_stats = {}
    plot_floor = 1e-16

    plt.figure(figsize=(6, 3))

    for name, results_list in all_results.items():
        failed_trials = trials - len(results_list)
        failure_probability = 100.0 * failed_trials / trials if trials > 0 else np.nan

        if not results_list:
            summary_stats[name] = {
                "Mean": np.nan,
                "Median": np.nan,
                "Standard deviation": np.nan,
                "Failure probability (%)": failure_probability,
            }
            continue

        df = pd.DataFrame(results_list).T
        df.columns = [f"trial_{j + 1}" for j in range(len(results_list))]

        min_vals = df.min(axis=0)
        mean_curve = df.mean(axis=1)
        median_curve = df.median(axis=1)

        summary_stats[name] = {
            "Mean": min_vals.mean(),
            "Median": min_vals.median(),
            "Standard deviation": min_vals.std(),
            "Failure probability (%)": failure_probability,
        }

        x_data = df.index + 1
        color = colors.get(name, "black")

        mean_plot = np.maximum(mean_curve, plot_floor)
        median_plot = np.maximum(median_curve, plot_floor)

        plt.plot(x_data, mean_plot, label=name, color=color, linewidth=1.5)
        # plt.plot(x_data, median_plot, color=color, linestyle="--", alpha=0.5, linewidth=1)

    print("\n[Running Time Summary]")
    for name, times in running_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"{name:7s} : {avg_time:.5f} sec")

    print("\n[Failure Summary]")
    for name, results_list in all_results.items():
        failed_trials = trials - len(results_list)
        failure_probability = 100.0 * failed_trials / trials if trials > 0 else np.nan
        print(f"{name:9s}: {failed_trials}/{trials} failed ({failure_probability:.2f}%)")

    print("\n[Final Performance Summary]")
    summary_df = pd.DataFrame(summary_stats).T
    display_df = summary_df.copy()
    console_df = summary_df.copy()

    for col in display_df.columns:
        if col == "Failure probability (%)":
            display_df[col] = display_df[col].apply(format_percent_latex)
            console_df[col] = console_df[col].apply(
                lambda x: "--" if pd.isna(x) else f"{x:.1f}"
            )
        else:
            display_df[col] = display_df[col].apply(format_sci_latex)

    display_df = display_df.rename(columns={"Failure probability (%)": r"Failure probability (\%)"})

    pd.options.display.float_format = "{:.4e}".format

    os.makedirs(os.path.join(base_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)

    suffix = f"{file_number}-N{format_param(p)}{format_param(q)}-n{n}-{function_label}"
    path_txt = os.path.join(base_dir, f"tables/table{suffix}.txt")
    path_fig = os.path.join(base_dir, f"figures/figure{suffix}.{'pdf' if pdf else 'png'}")

    with open(path_txt, "w", encoding="utf-8") as table_file:
        table_file.write(display_df.to_latex(escape=False))

    print(console_df)

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(\mathbf{x}_k)$", fontsize=10)
    plt.title(fr"$N_{{{p:g},{q:g}}}$, $n={n}$", fontsize=10)
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
