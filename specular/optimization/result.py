import numpy as np

from .._typing import Scalar, Vector


class OptimizationResult:
    def __init__(
        self,
        method: str,
        solution: Scalar | Vector,
        func_val: Scalar,
        iteration: int,
        runtime: float,
        all_history: dict,
        stop_reason: str | None = None
    ):
        self.method = method
        self.solution = solution
        self.func_val = func_val
        self.iteration = iteration
        self.runtime = runtime
        self.all_history = all_history
        self.stop_reason = stop_reason

    def __repr__(self):

        text = (
            f"[{self.method}]\n"
            f"    solution: {self.solution}\n"
            f"  func value: {self.func_val}\n"
            f"   iteration: {self.iteration}"
        )

        if self.stop_reason is not None:
            text += f"\n stop reason: {self.stop_reason}"

        return text

    def last_record(
        self
    ) -> tuple[Scalar | Vector, Scalar, float]:
        """
        Returns the final solution x, the value of f at x, and the runtime as a tuple.

        Returns:
            (x, f(x), runtime)
        """
        return self.solution, self.func_val, self.runtime

    def history(
        self
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns the recorded iterates, objective values, and runtime.

        Returns:
            (x_history, f_history, runtime)
        """
        return self.all_history["variables"], self.all_history["values"], self.runtime