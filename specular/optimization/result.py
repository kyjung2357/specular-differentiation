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
        fill_iteration: bool = False,
        max_iter: int | None = None,
        stop_reason: str | None = None
    ):  
        if fill_iteration and max_iter is None:
            raise ValueError("max_iter must be provided when fill_iteration=True.")
        
        self.method = method
        self.solution = solution
        self.func_val = func_val
        self.iteration = iteration
        self.runtime = runtime
        self.all_history = all_history
        self.fill_iteration = fill_iteration
        self.max_iter = max_iter
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

    def last_record(self) -> tuple[Scalar | Vector, Scalar, float]:
        """
        Returns the final solution x, the value of f at x, and the runtime as a tuple.

        Returns:
            (x, f(x), runtime)
        """
        return self.solution, self.func_val, self.runtime

    def history(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns the recorded iterates, objective values, and runtime.

        Returns:
            (x_history, f_history, runtime)
        """
        variables = self.all_history["variables"]
        values = self.all_history["values"]

        if self.fill_iteration and self.max_iter is not None:
            variables = self._fill_history(variables, self.max_iter)
            values = self._fill_history(values, self.max_iter)

        return variables, values, self.runtime

    @staticmethod
    def _fill_history(history, target_length: int):
        history = np.asarray(history)

        if len(history) == 0 or len(history) >= target_length:
            return history

        tail = np.repeat(history[-1:], target_length - len(history), axis=0)
        return np.concatenate([history, tail], axis=0)