import os
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

class OptimizationResult:
    def __init__(
            self, 
            method: str,
            solution: np.ndarray, 
            func_val: int | float | np.number, 
            iteration: int, 
            runtime: float,
            all_history: dict
    ):
        self.method = method
        self.solution = solution
        self.func_val = func_val
        self.iteration = iteration
        self.runtime = runtime
        self.all_history = all_history

    def __repr__(self):
        return (
            f"[{self.method}]\n"
            f"    solution: {self.solution}\n"
            f"  func value: {self.func_val}\n"
            f"   iteration: {self.iteration}"
        )
    
    def last_record(
        self
    ) -> Tuple[float, float, float]:
        """
        Returns the final solution x, the value of f at x, and the runtime as a tuple.
        
        Returns:
            (x, f(x), runtime)
        """
        return self.solution, self.func_val, self.runtime # type: ignore

    def history(
        self
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns the time grid and the numerical solution as a tuple.
        
        Returns:
            (x_history, f_history, runtime)
        """
        return self.all_history["variables"], self.all_history["values"], self.runtime