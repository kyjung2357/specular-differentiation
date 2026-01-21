import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

def ensure_length(data, length):
    data = list(data)
    if len(data) == 0: return [0.0] * length 
    if len(data) < length: return data + [data[-1]] * (length - len(data))
    else: return data[:length]
    
def format_sci_latex(x):
    if isinstance(x, str): return x
    if x == 0: return "0"
    s = "{:.4e}".format(x) 
    base, exponent = s.split('e')
    return fr"${base} \times 10^{{{int(exponent)}}}$"

def save_table_to_txt(df, filename, precision=4, formatting="exponential"):
    if formatting == "exponential":
        formatter = "{:.4e}".format
    else:
        formatter = "{:.4f}".format
        
    latex_str = df.to_latex(float_format=formatter)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"[Saved] Table saved to: {filename}")

def report_results(all_results, running_times, file_number, m, n, lambda1, lambda2, iteration, base_dir, pdf=False, show=False):
    colors = {
        'SPEG': 'red', 'SPEG-s': 'red', 'SPEG-g': 'brown', 'S-SPEG': 'blue', 'H-SPEG': 'purple', 'GD': 'orange', 'Adam': 'green', 'BFGS': 'black'
    }

    summary_stats = {}
    
    plt.figure(figsize=(6, 3))

    for name, results_list in all_results.items():
        if not results_list: continue

        df = pd.DataFrame(results_list).T
        df.columns = [f'trial_{j+1}' for j in range(len(results_list))]
        
        min_vals = df.min(axis=0)
        mean_curve = df.mean(axis=1)
        median_curve = df.median(axis=1)
        std_curve = df.std(axis=1)

        summary_stats[name] = {
            'Mean': min_vals.mean(),
            'Median': min_vals.median(),
            'Standard deviation': min_vals.std()
        }

        x_data = df.index + 1
        color = colors.get(name, 'black')
        
        plt.plot(x_data, mean_curve, label=name, color=color, linewidth=1.5)
        plt.plot(x_data, median_curve, color=color, linestyle='--', alpha=0.5, linewidth=1)
        plt.fill_between(x_data, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

    print("\n[Running Time Summary]")
    for name, times in running_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"{name:5s} : {avg_time:.5f} sec")

    print("\n[Final Performance Summary]")
    summary_df = pd.DataFrame(summary_stats).T
    display_df = summary_df.copy()

    for col in display_df.columns:
        display_df[col] = display_df[col].apply(format_sci_latex)
        
    pd.options.display.float_format = '{:.4e}'.format
    
    os.makedirs(os.path.join(base_dir, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'figures'), exist_ok=True)
    
    path_txt = os.path.join(base_dir, f'tables/table{file_number}-{m}-{n}-{lambda1}-{lambda2}.txt')
    path_fig = os.path.join(base_dir, f'figures/figure{file_number}-{m}-{n}-{lambda1}-{lambda2}.{"pdf" if pdf else "png"}')

    with open(path_txt, "w", encoding="utf-8") as table_file:
        table_file.write(display_df.to_latex(escape=False))

    print(summary_df)

    plt.xlabel(r"Iterations $k$", fontsize=10)
    plt.ylabel(r"Objective function value $f(\mathbf{x}_k)$", fontsize=10)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1, iteration)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)
    plt.tight_layout() 
    
    plt.savefig(path_fig, dpi=1000, bbox_inches='tight')
    print(f"Saved figure to: {path_fig}")

    if show:
        plt.show()