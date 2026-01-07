import specular
import numpy as np
import pandas as pd
from tqdm import tqdm  
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import matplotlib
import sys
import os
import torch
from concurrent.futures import ProcessPoolExecutor 

matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from classical_methods import Adam, BFGS, Gradient_descent_method

machine_epsilon = np.finfo(float).eps

# -----------------------------------------------------------
# [1] Auxiliary functions
# -----------------------------------------------------------
def ensure_length(data, length):
    data = list(data)
    if len(data) == 0: return [0.0] * length 
    if len(data) < length: return data + [data[-1]] * (length - len(data))
    else: return data[:length]
    
def format_sci_latex(x):
    if isinstance(x, str): return x
    if x == 0: return "0"
    s = "{:.2e}".format(x) 
    base, exponent = s.split('e')
    return fr"${base} \times 10^{{{int(exponent)}}}$"

# -----------------------------------------------------------
# [2] A single function
# -----------------------------------------------------------
def run_single_experiment(args):
    seed, f, f_torch, max_iter = args
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1) 

    history = {}
    x_0_val = np.random.randn(1)
    
    # ==== Specular Methods ====
    step_size1 = specular.StepSize(name='geometric_series', parameters=[1, 0.5])
    
    # ISGM
    _, res = specular.gradient_method(f=f, x_0=x_0_val, step_size=step_size1, form='implicit', max_iter=max_iter, print_bar=False).history()
    history["ISGM"] = ensure_length(res, max_iter)

    # SPEG (geometric series)
    _, res = specular.gradient_method(f=f, x_0=x_0_val, step_size=step_size1, max_iter=max_iter, print_bar=False).history()
    history["SPEG geo"] = ensure_length(res, max_iter)

    # SPEG (square summable not summable)
    step_size2 = specular.StepSize(name='square_summable_not_summable', parameters=[2, 0])
    _, res = specular.gradient_method(f=f, x_0=x_0_val, step_size=step_size2, max_iter=max_iter, print_bar=False).history()
    history["SPEG sq"] = ensure_length(res, max_iter)

    # ==== Classical Methods ====
    
    # Gradient Descent (geometric series)
    _, res = Gradient_descent_method(f_torch=f_torch, x_0=x_0_val, step_size=step_size1, max_iter=max_iter)
    history["GD geo"] = ensure_length(res, max_iter)

    # Gradient Descent (square summable not summable)
    _, res = Gradient_descent_method(f_torch=f_torch, x_0=x_0_val, step_size=step_size2, max_iter=max_iter)
    history["GD sq"] = ensure_length(res, max_iter)

    # Adam
    _, res = Adam(f_torch=f_torch, x_0=x_0_val, step_size=0.01, max_iter=max_iter)
    history["Adam"] = ensure_length(res, max_iter)

    # BFGS
    _, res = BFGS(f_np=f, x_0=x_0_val, max_iter=max_iter)
    history["BFGS"] = ensure_length(res, max_iter)
    
    return history

# -----------------------------------------------------------
# [3] Main function
# -----------------------------------------------------------
def repeat_experiment(f, f_torch, num_runs, max_iter, latex_code=False, save_name=False):
    histories = {"ISGM": [], "SPEG geo": [], "SPEG sq": [], "GD geo": [], "GD sq": [], "Adam": [], "BFGS": []}

    seeds = range(num_runs)
    tasks = [(seed, f, f_torch, max_iter) for seed in seeds]

    print(f"Starting parallel execution with {os.cpu_count()} cores...")

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_single_experiment, tasks), total=num_runs, desc="Running Experiments"))

    for single_history in results:
        for method, data in single_history.items():
            histories[method].append(data)

    # ==== Visualization ====
    plt.figure(figsize=(7, 3))
    x_axis = range(1, max_iter + 1)

    colors = {'ISGM': 'blue', 'SPEG geo': 'red', 'SPEG sq': 'orange', 'GD geo': 'darkgreen', 'GD sq': 'limegreen', 'Adam': 'brown', 'BFGS': 'black'}
    linestyles = {'ISGM': '-', 'SPEG geo': '-', 'SPEG sq': '-', 'GD geo': '-', 'GD sq': '-', 'Adam': '-', 'BFGS': '-'}

    for name, data in histories.items():
        arr = np.array(data)
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        
        c = colors.get(name, 'black')
        ls = linestyles.get(name, '-')
        
        plt.plot(x_axis, med, label=f'{name}', color=c, linestyle=ls)
        plt.fill_between(x_axis, q25, q75, color=c, alpha=0.15)

    plt.xlabel('Iteration $k$', fontsize='10')
    plt.ylabel(r'Objective function value $f(x_k)$', fontsize='10') 
    plt.yscale('log')
    # plt.xscale('log') 
    plt.xlim(1, max_iter)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize='10', frameon=True)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_name:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        figures_dir = os.path.join(current_dir, 'figures')

        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        fig_name = f"figure_{save_name}.pdf"
        fig_path = os.path.join(figures_dir, fig_name)

        plt.savefig(fig_path, dpi=1000, bbox_inches='tight') # type: ignore
        print(f"[Saved] Plot saved to: {fig_path}")

    # plt.show()

    # ==== Table ====
    table_data = []
    for name, runs in histories.items():
        best_errors = [max(np.min(run_history), machine_epsilon) for run_history in runs]
        table_data.append({
            "Method": name,
            "Mean": np.mean(best_errors),
            "Median": np.median(best_errors),
            "Standard deviation": np.std(best_errors)
        })

    df_summary = pd.DataFrame(table_data)
    df_display = df_summary.copy()
    cols_to_format = ["Mean", "Median", "Standard deviation"]
    
    for col in cols_to_format:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_sci_latex)

    print(f"\n=== Performance Summary ({num_runs} runs) ===")
    display(df_display)

    if latex_code:
        latex_str = df_display.to_latex(index=False, escape=False)

        if save_name:
            table_dir = os.path.join(current_dir, 'tables') # type: ignore

            if not os.path.exists(table_dir):
                os.makedirs(table_dir)

            table_name = f"table_{save_name}.txt"
            table_path = os.path.join(table_dir, table_name)

            with open(table_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(latex_str)
            
            print(f"[Saved] Table saved to: {table_path}")    
