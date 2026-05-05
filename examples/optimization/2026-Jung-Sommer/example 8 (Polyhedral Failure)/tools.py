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


def format_percent_latex(x):
    if pd.isna(x):
        return "--"
    return f"{x:.1f}"


def format_symlog_tick(value, _position=None):
    if value == 0:
        return "0"

    sign = "-" if value < 0 else ""
    exponent = int(round(np.log10(abs(value))))
    return rf"${sign}10^{{{exponent}}}$"


def symlog_major_ticks(y_min, y_max):
    max_abs = max(abs(y_min), abs(y_max), 1.0)
    max_power = int(np.ceil(np.log10(max_abs)))
    powers = list(range(0, max_power + 1, 2))

    ticks = [-(10.0**p) for p in reversed(powers)]
    ticks.append(0.0)
    ticks.extend(10.0**p for p in powers)

    return [tick for tick in ticks if y_min <= tick <= y_max]


def report_results(
    all_results,
    running_times,
    file_number,
    m,
    n,
    lambda1,
    lambda2,
    trials,
    iteration,
    base_dir,
    pdf=False,
    show=False,
    yscale="log",
    title=None,
):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    matplotlib.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "Times New Roman"

    colors = {
        "SPEG": "black",
        "SPEG-s": "red",
        "SPEG-g": "brown",
        "S-SPEG": "blue",
        "H-SPEG": "purple",
        "BFGS": "#08519c",
        "BFGS-E": "#08306b",
        "BFGS-S": "#08519c",
        "BFGS-W": "#2171b5",
        "BFGS-A": "#6baed6",
        "S-BFGS": "#fb6a4a",
        "S-BFGS-E": "#67000d",
        "S-BFGS-S": "#a50f15",
        "S-BFGS-W": "#de2d26",
        "S-BFGS-A": "#fb6a4a",
        "GD": "orange",
        "Adam": "green",
    }

    summary_stats = {}
    has_curve = False
    plot_floor = 1e-16

    plt.figure(figsize=(6, 3))

    for name, results_list in all_results.items():
        failed_trials = trials - len(results_list)
        failure_probability = 100.0 * failed_trials / trials if trials > 0 else pd.NA

        if not results_list:
            summary_stats[name] = {
                "Mean": pd.NA,
                "Median": pd.NA,
                "Standard deviation": pd.NA,
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

        if yscale == "log":
            mean_plot = np.maximum(mean_curve, plot_floor)
            median_plot = np.maximum(median_curve, plot_floor)
        else:
            mean_plot = mean_curve
            median_plot = median_curve

        plt.plot(x_data, mean_plot, label=name, color=color, linewidth=1.5)
        # plt.plot(x_data, median_plot, color=color, linestyle="--", alpha=0.5, linewidth=1)
        has_curve = True

    print("\n[Running Time Summary]")
    for name, times in running_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"{name:7s} : {avg_time:.5f} sec")

    print("\n[Failure Summary]")
    for name, results_list in all_results.items():
        failed_trials = trials - len(results_list)
        failure_probability = 100.0 * failed_trials / trials if trials > 0 else pd.NA
        failure_text = "--" if pd.isna(failure_probability) else f"{failure_probability:.1f}"
        print(f"{name:9s}: {failed_trials}/{trials} failed ({failure_text}%)")

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

    suffix = f"{file_number}-{m}-{n}-{lambda1}-{lambda2}"
    path_txt = os.path.join(base_dir, f"tables/table{suffix}.txt")
    path_fig = os.path.join(base_dir, f"figures/figure{suffix}.{'pdf' if pdf else 'png'}")

    with open(path_txt, "w", encoding="utf-8") as table_file:
        table_file.write(display_df.to_latex(escape=False))

    print(console_df)

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(\mathbf{x}_k)$", fontsize=10)
    if title:
        plt.title(title, fontsize=10)
    plt.grid(True)
    plt.xscale("log")
    if yscale == "symlog":
        plt.yscale("symlog", linthresh=1.0)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        y_max = min(y_max, 100.0)
        ax.set_ylim(y_min-5*10**16, y_max)
        ax.yaxis.set_major_locator(FixedLocator(symlog_major_ticks(y_min, y_max)))
        ax.yaxis.set_major_formatter(FuncFormatter(format_symlog_tick))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="y", labelsize=9, pad=5)
    else:
        plt.yscale(yscale)
    plt.xlim(1, iteration)

    if has_curve:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=10)

    plt.tight_layout()
    plt.savefig(path_fig, dpi=1000, bbox_inches="tight")
    print(f"Saved table to: {path_txt}")
    print(f"Saved figure to: {path_fig}")

    if show:
        plt.show()
