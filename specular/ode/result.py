import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple

class ODEResult:
    def __init__(
        self, 
        h: float,
        time_grid: np.ndarray, 
        numerical_sol: np.ndarray, 
        scheme: str
    ):
        self.h = h
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

        capitalized_name = self.scheme[0].upper() + self.scheme[1:]
        
        plt.plot(self.time_grid, self.numerical_sol, linestyle='--', marker='o', color='red', markersize=5, markevery=number_of_circles, markerfacecolor='none', markeredgewidth=1.0, label=capitalized_name)

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

            print(f"Figure saved: {full_path}")
        
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
            
            print(f"Table saved: {full_path}")

        return result

    def total_error(self,
        exact_sol: Callable[[float], float] | list | np.ndarray,
        norm: str = 'max'
    ) -> float:
        """
        Calculates the error between the numerical solution and the exact solution.

        Parameters
        ----------
        exact_sol : callable, list, np.ndarray
            A function that returns the exact solution at a given time ``t``, or a list/array containing the exact values corresponding to ``time_grid``.
        norm : str, optional
            The type of norm to use ``'max'``, ``'l2'``, or ``'l1'``.
            Default: ``'max'``

        Returns
        -------
        float
            The computed error value.

        Raises
        ------
        TypeError
            If ``exact_sol`` is neither a callable nor a list/array.
        ValueError
            If ``exact_sol`` (list) shape does not match ``numerical_sol``.
        """
        if callable(exact_sol):
            exact_values = np.array([exact_sol(t) for t in self.time_grid])
        elif isinstance(exact_sol, (list, np.ndarray)):
            exact_values = np.asarray(exact_sol, dtype=float) 
        else:
            raise TypeError("exact_sol must be a callable or a list/array.")
        
        if exact_values.shape != self.numerical_sol.shape:
             raise ValueError(f"Shape mismatch: exact_sol {exact_values.shape} vs numerical_sol {self.numerical_sol.shape}")
        
        error_vector = np.abs(exact_values - self.numerical_sol) 

        if norm == 'max':
            return float(np.max(error_vector))
        
        elif norm == 'l2':
            return float(np.sqrt(np.sum(error_vector**2) * self.h))
        
        elif norm == 'l1':
            return float(np.sum(error_vector) * self.h)
        
        else:
            raise ValueError(f"Unknown norm type. Got '{norm}'. Supported types: 'max', 'l2', 'l1'.")
        
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
