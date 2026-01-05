import specular
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import List, Tuple, Optional

def compute_ratios(
        error_list: List[Tuple[int, float]]
) -> List[Tuple[int, float, Optional[float]]]:
    """
    Computes the convergence order (p) from a list of errors.
    Works for ANY sequence of n (not just doubling).
    """
    ratio_list = []
    for i in range(len(error_list)):
        n, e = error_list[i]

        if i == 0:
            ratio_list.append((n, e, None))  
        else:
            n_prev, e_prev = error_list[i - 1]
            
            if e <= 0 or e_prev <= 0:
                ratio = None
            else:
                ratio = np.log(e_prev / e) / np.log(n / n_prev)
            
            ratio_list.append((n, e, ratio))

    return ratio_list

def save_table_to_txt(
    df: pd.DataFrame,
    filename: str,
    error_precision: int = 2,
    ratio_precision: int = 2
) -> None:
    """
    Saves a DataFrame of convergence results to a text file in LaTeX table format.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with three columns: 'n', 'error', and 'ratio'.
    filename : str
        The name of the text file to save the results to.
    error_precision : int, optional
        The number of decimal places for the 'error' column. 
        Default: ``2``.
    ratio_precision : int, optional
        The number of decimal places for the 'ratio' column. 
        Default: ``2``.

    Returns
    -------
    None
        This function does not return any value; it writes directly to a file.
    """
    if not os.path.exists('tables'):
        os.makedirs('tables')

    filename = "tables/" + filename

    with open(filename, "w") as f:
        for n, error, ratio in df.itertuples(index=False, name=None):
            error_str = f"{error:.{error_precision}e}"
            ratio_str = f"{ratio:.{ratio_precision}f}" if pd.notna(ratio) else "--"
            f.write(f"{n:<8}& {error_str} & {ratio_str} \\\\\n")

def error_analysis(example, norm, F, t_0, T, exact_sol):
    norm_label_map = {
    'max': r'\infty',
    'l1': '1',
    'l2': '2'
    }
    
    u_0_val = exact_sol(t_0) 

    error_list = {
        "EE": [],
        "IE": [],
        "CN": [],
        "S5": [],
        "S6": []
    }

    for k in range(3, 18):
        n = 2**k
        h = 1/n
        
        print(f"Running for k={k}, n={n}...")

        res_EE = specular.ode.classical_scheme(F=F, u_0=u_0_val, t_0=t_0, T=T, h=h, scheme="explicit Euler")
        res_IE = specular.ode.classical_scheme(F=F, u_0=u_0_val, t_0=t_0, T=T, h=h, scheme="implicit Euler")
        res_CN = specular.ode.classical_scheme(F=F, u_0=u_0_val, t_0=t_0, T=T, h=h, scheme="Crank-Nicolson")
        res_S5 = specular.Euler_scheme(of_Type=5, F=F, t_0=t_0, u_0=u_0_val, T=T, h=h, max_iter=1000)
        res_S6 = specular.Euler_scheme(of_Type=6, F=F, t_0=t_0, u_0=u_0_val, T=T, h=h, max_iter=1000)

        error_list["EE"].append((n, res_EE.total_error(exact_sol=exact_sol, norm=norm)))
        error_list["IE"].append((n, res_IE.total_error(exact_sol=exact_sol, norm=norm)))
        error_list["CN"].append((n, res_CN.total_error(exact_sol=exact_sol, norm=norm)))
        error_list["S5"].append((n, res_S5.total_error(exact_sol=exact_sol, norm=norm)))
        error_list["S6"].append((n, res_S6.total_error(exact_sol=exact_sol, norm=norm)))

        clear_output(wait=True)

    df_EE = pd.DataFrame(compute_ratios(error_list["EE"]), columns=["n", "Error", "Ratio"])
    df_IE = pd.DataFrame(compute_ratios(error_list["IE"]), columns=["n", "Error", "Ratio"])
    df_CN = pd.DataFrame(compute_ratios(error_list["CN"]), columns=["n", "Error", "Ratio"])
    df_S5 = pd.DataFrame(compute_ratios(error_list["S5"]), columns=["n", "Error", "Ratio"])
    df_S6 = pd.DataFrame(compute_ratios(error_list["S6"]), columns=["n", "Error", "Ratio"])

    save_table_to_txt(df_EE, f"{example}-{norm}-Table-EE.txt", error_precision=2, ratio_precision=3)
    save_table_to_txt(df_IE, f"{example}-{norm}-Table-IE.txt", error_precision=2, ratio_precision=3)
    save_table_to_txt(df_CN, f"{example}-{norm}-Table-CN.txt", error_precision=2, ratio_precision=3)
    save_table_to_txt(df_S5, f"{example}-{norm}-Table-S5.txt", error_precision=2, ratio_precision=3)
    save_table_to_txt(df_S6, f"{example}-{norm}-Table-S6.txt", error_precision=2, ratio_precision=3)

    plt.figure(figsize=(5.5, 2.5))

    plt.plot(df_EE["n"], df_EE["Error"], color='red', marker='x', markerfacecolor='none', markeredgecolor='red', label="Explicit Euler")
    plt.plot(df_IE["n"], df_IE["Error"], color='blue', marker='x', markerfacecolor='none', markeredgecolor='blue', label='Implicit Euler')
    plt.plot(df_CN["n"], df_CN["Error"], color='purple', marker='x', markerfacecolor='none', markeredgecolor='purple', label='Crank-Nicolson')
    plt.plot(df_S5["n"], df_S5["Error"], color='green', marker='v', markerfacecolor='none', markeredgecolor='green', label="Specular Euler (Type 5)")
    plt.plot(df_S6["n"], df_S6["Error"], color='orange', marker='v', markerfacecolor='none', markeredgecolor='orange', label="Specular Euler (Type 6)")

    plt.xscale('log', base=2)
    plt.yscale('log')

    plt.xlabel(r"Number of time steps $N$", fontsize=10)
    ylabel_str = fr"Error $\mathcal{{E}}(N, {norm_label_map[norm]})$"
    plt.ylabel(ylabel_str, fontsize=10)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.savefig(f'figures/{example}-{norm}-Figure.png', dpi=300, bbox_inches='tight')
    plt.show()