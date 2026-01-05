import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple

class ODEResult:
    def __init__(
        self, 
        time_grid: np.ndarray, 
        numerical_sol: np.ndarray, 
        scheme: str
    ):
        self.time_grid = time_grid
        self.numerical_sol = numerical_sol
        self.scheme = scheme
    
    def values(
        self
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the time grid and the numerical solution as a tuple.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (time_grid, numerical_sol)
        """
        return self.time_grid, self.numerical_sol
    
    def visualization(
        self, 
        figure_size: tuple = (5.5, 2.5), 
        exact_sol: Optional[Callable[[float], float]] = None,                       
        save_path: Optional[str] = None
    ):
        plt.figure(figsize=figure_size)
        
        if exact_sol is not None:
            exact_values = np.array([exact_sol(t) for t in self.time_grid])
            plt.plot(self.time_grid, exact_values, color='black', label='Exact solution')

        number_of_circles = max(1, len(self.time_grid) // 30)

        plt.plot(self.time_grid, self.numerical_sol, linestyle='--', marker='o', color='red', markersize=5, markevery=number_of_circles, markerfacecolor='none', markeredgewidth=1.0, label=self.scheme)

        plt.xlabel(r"Time", fontsize=10)
        plt.ylabel(r"Solution", fontsize=10)
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)

        if save_path:
            if not os.path.exists('figures'):
                os.makedirs('figures')
            
            save_path = save_path.replace(" ", "-")
            full_path = os.path.join("figures", save_path)

            if not save_path.endswith(".png"):
                save_path += ".png"
                
            plt.savefig(full_path, dpi=1000, bbox_inches='tight')

            print(f"Figure saved to `{full_path}`")
        
        plt.show()

        return self

    def table(self,
        exact_sol: Optional[Callable[[float], float]] = None,   
        save_path: Optional[str] = None
    ):
        
        result = pd.DataFrame(self.numerical_sol, index=self.time_grid, columns=["Numerical solution"])
        result.index.name = "Time"

        if exact_sol:
            result["Exact solution"] = [exact_sol(t) for t in self.time_grid]
            result["Error"] = abs(result["Numerical solution"] - result["Exact solution"])

        if save_path:
            if not os.path.exists('tables'):
                os.makedirs('tables')

            save_path = save_path.replace(" ", "-")
            full_path = os.path.join("tables", save_path)
            
            if full_path.endswith(".txt"):
                with open(full_path, "w") as f:
                    f.write(result.to_string())
            else:
                if not full_path.endswith(".csv"):
                    full_path += ".csv"
                
                result.to_csv(full_path) 
            
            print(f"Table saved to {full_path}")

        return result

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
