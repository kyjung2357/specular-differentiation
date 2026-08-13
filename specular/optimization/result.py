from .._typing import Scalar, Vector

from typing import TypedDict
import numpy as np

class OptimizationHistory(TypedDict):
    variables: list | np.ndarray
    values: list | np.ndarray

class OptimizationResult:
    def __init__(
        self,
        method: str,
        solution: Scalar | Vector,
        func_val: Scalar,
        iteration: int,
        runtime: float,
        history: OptimizationHistory,
        stop_reason: str | None = None
    ):
        self.method = method
        self.solution = solution
        self.func_val = func_val
        self.iteration = iteration
        self.runtime = runtime
        self.all_history = history
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

    def get_history(
            self,
            fill_to: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns the recorded iterates, objective values, and runtime.

        Returns:
            (x_history, f_history, runtime)
        """
        variables = np.asarray(self.all_history["variables"])
        values = np.asarray(self.all_history["values"])

        if fill_to is not None:
            variables = self._fill(variables, fill_to)
            values = self._fill(values, fill_to)

        return variables, values, self.runtime

    def history(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the recorded iterates, objective values, and runtime."""
        return self.get_history()


    @staticmethod
    def _fill(opt_history, target_length: int):
        opt_history = np.asarray(opt_history)

        if len(opt_history) == 0 or len(opt_history) >= target_length:
            return opt_history

        tail = np.repeat(opt_history[-1:], target_length - len(opt_history), axis=0)
        return np.concatenate([opt_history, tail], axis=0)
