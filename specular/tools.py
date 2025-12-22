import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable

class ODEResult:
    def __init__(self, t: np.ndarray, u: np.ndarray, scheme: str):
        self.t = t
        self.u = u
        self.scheme = scheme

    def visualization(self, 
                      exact_sol: Optional[Callable[[float], float]] = None, 
                      figsize: tuple = (5.5, 2.5), 
                      save_path: Optional[str] = None):
        
        plt.figure(figsize=figsize)
        
        # Exact solution 그리기
        if exact_sol is not None:
            exact_values = np.array([exact_sol(ti) for ti in self.t])
            plt.plot(self.t, exact_values, color='black', label='Exact solution')

        # 수치해 그리기
        plt.plot(self.t, self.u, label=self.scheme)

        # 꾸미기
        plt.xlabel(r"Time $t$", fontsize=10)
        plt.ylabel(r"Solution $u(t)$", fontsize=10)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


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
        The number of decimal places for the 'error' column. Default is 12.
    ratio_precision : int, optional
        The number of decimal places for the 'ratio' column. Default is 4.

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

class OptimizationResult:
    def __init__(self):
        pass