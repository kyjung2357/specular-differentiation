import specular
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

import sys
import os
sys.path.append(os.path.abspath(".."))
from classical_methods import Adam_optimizer, BFGS_optimizer, Gradient_descent_method

def ensure_length(data, length):
    data = list(data)
    if len(data) == 0:
        return [0.0] * length 
    
    if len(data) < length:
        return data + [data[-1]] * (length - len(data))
    else:
        return data[:length]
    
def format_sci_latex(x):
    if isinstance(x, str): return x

    if x == 0: return "0"
    
    s = "{:.2e}".format(x) 
    base, exponent = s.split('e')
    exponent = int(exponent) 
    
    return fr"${base} \times 10^{{{exponent}}}$"

def repeat_experiment(f, f_torch, num_runs, max_iter, latex_code=False, save_path=False):
    histories = {"ISGM": [], "SPEG geo": [], "SPEG sq": [], "GD": [], "Adam": [], "BFGS": []}

    for run in tqdm(range(num_runs), desc="Running Experiments"):
        x_0_val = np.random.randn(1)
        
        # --- Specular gradient methods ---
        step_size1 = specular.StepSize(name='geometric_series', parameters=[1, 0.5])
        
        # ISGM
        _, res_ISGM = specular.gradient_method(f=f, x_0=x_0_val, step_size=step_size1, form='implicit', max_iter=max_iter, print_bar=False).history()
        histories["ISGM"].append(ensure_length(res_ISGM, max_iter))

        # SPEG (Geometric)
        _, res_SPEG_geo = specular.gradient_method(f=f, x_0=x_0_val, step_size=step_size1, max_iter=max_iter, print_bar=False).history()
        histories["SPEG geo"].append(ensure_length(res_SPEG_geo, max_iter))

        # SPEG (Square Summable)
        step_size2 = specular.StepSize(name='square_summable_not_summable', parameters=[2, 0])
        _, res_SPEG_sq = specular.gradient_method(f=f, x_0=x_0_val, step_size=step_size2, max_iter=max_iter, print_bar=False).history()
        histories["SPEG sq"].append(ensure_length(res_SPEG_sq, max_iter))

        # --- Classical Methods ---

        # Gradient Descent
        _, res_gd = Gradient_descent_method(f_torch=f_torch, x_0=x_0_val, step_size=0.001, max_iter=max_iter)
        histories["GD"].append(ensure_length(res_gd, max_iter))

        #  Adam
        _, res_adam = Adam_optimizer(f_torch=f_torch, x_0=x_0_val, step_size=0.01, max_iter=max_iter)
        histories["Adam"].append(ensure_length(res_adam, max_iter))

        #  BFGS 
        _, res_bfgs = BFGS_optimizer(f_np=f, x_0=x_0_val, max_iter=max_iter)
        histories["BFGS"].append(ensure_length(res_bfgs, max_iter))

    # ==== Visualization ====
    plt.figure(figsize=(7, 3))
    x_axis = range(1, max_iter + 1)

    colors = {'ISGM': 'red', 'SPEG geo': 'blue', 'SPEG sq': 'black', 'GD': 'green', 'Adam': 'brown', 'BFGS': 'purple'}
    linestyles = {'ISGM': '-', 'SPEG geo': '-', 'SPEG sq': '-', 'GD': '-.', 'Adam': '-.', 'BFGS': '-.'}

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
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize='10', frameon=True)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path is not False:
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        path = 'figures/' + save_path # type: ignore
        plt.savefig(path, dpi=1000, bbox_inches='tight')

    plt.show()

    table_data = []
    for name, runs in histories.items():
        best_errors = [np.min(run_history) for run_history in runs]
        table_data.append({
            "Method": name,
            "Mean Min Error": np.mean(best_errors),
            "Median Min Error": np.median(best_errors),
            "Std Dev": np.std(best_errors)
        })

    df_summary = pd.DataFrame(table_data)
    df_display = df_summary.copy()
    cols_to_format = ["Mean Min Error", "Median Min Error", "Std Dev"]
    
    for col in cols_to_format:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_sci_latex)

    print(f"\n=== Performance Summary ({num_runs} runs) ===")
    display(df_display)

    # LaTeX 코드 출력
    if latex_code:
        print("\n=== LaTeX Code ===")
        print(df_display.to_latex(index=False, escape=False))