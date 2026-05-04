from __future__ import annotations

from typing import cast
from typing import Callable, Literal, TypeAlias
from specular._typing import Vector, ScalarToScalarFunc, VectorToScalarFunc, ScalarToVectorFunc, VectorToVectorFunc

import numpy as np

LineSearchName: TypeAlias = Literal["exact", "armijo", "wolfe", "strong_wolfe"]

class LineSearchError(RuntimeError):
    """Raised when a line search fails to satisfy its condition."""

class LineSearch:
    """
    Line search rules for choosing the step length along a search direction.
    """

    __options__ = [
        "exact",
        "armijo",
        "wolfe",
        "strong_wolfe",
    ]

    def __init__(
        self,
        name: str | None = None,
        alpha_0: float = 1.0,
        c_1: float = 1e-4,
        c_2: float = 0.9,
        rho: float = 0.5,
        max_iter: int = 20,
        max_alpha: float = 1e8,
        raise_on_fail: bool = False,
        f: ScalarToScalarFunc | VectorToScalarFunc | None = None,
        gradient_f: ScalarToVectorFunc | VectorToVectorFunc | None = None,
    ):
        """
        Parameters:
            name (str):
                Required line search rule. Options: ``'armijo'``, ``'wolfe'``, and ``'strong_wolfe'``.
            alpha_0 (float, optional):
                Initial trial step length.
            c_1 (float, optional):
                Armijo sufficient decrease parameter.
            c_2 (float, optional):
                Wolfe curvature parameter.
            rho (float, optional):
                Backtracking factor.
            max_iter (int, optional):
                Maximum number of line search iterations.
            max_alpha (float, optional):
                Maximum trial step length used when expanding the interval.
            raise_on_fail (bool, optional):
                If ``True``, raise an error when the line search fails.
                If ``False``, return the final trial step length.
        """
        if name is None:
            raise ValueError(f"Line search name is required. Options: {self.__options__}")

        aliases = {
            "strong wolfe": "strong_wolfe",
            "strong-wolfe": "strong_wolfe",
        }
        name = aliases.get(name.lower(), name.lower())

        if name not in self.__options__:
            raise ValueError(f"Invalid line search '{name}'. Options: {self.__options__}")

        if alpha_0 <= 0:
            raise ValueError(f"alpha_0 must be positive. Got {alpha_0}")

        if not (0.0 < c_1 < 1.0):
            raise ValueError(f"c_1 must satisfy 0 < c_1 < 1. Got {c_1}")

        if not (0.0 < c_2 < 1.0):
            raise ValueError(f"c_2 must satisfy 0 < c_2 < 1. Got {c_2}")

        if name in ("wolfe", "strong_wolfe") and not (c_1 < c_2):
            raise ValueError(
                f"Wolfe line search requires 0 < c_1 < c_2 < 1. "
                f"Got c_1={c_1}, c_2={c_2}"
            )

        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must satisfy 0 < rho < 1. Got {rho}")

        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive. Got {max_iter}")

        if max_alpha <= 0:
            raise ValueError(f"max_alpha must be positive. Got {max_alpha}")

        self.name = name
        self.alpha_0 = float(alpha_0)
        self.c_1 = float(c_1)
        self.c_2 = float(c_2)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.max_alpha = float(max_alpha)
        self.raise_on_fail = raise_on_fail
        self.f = f
        self.gradient_f = gradient_f

    @staticmethod
    def _is_scalar_input(x) -> bool:
        return np.isscalar(x) or np.asarray(x).ndim == 0

    @staticmethod
    def _as_vector(value) -> np.ndarray:
        return np.asarray(value, dtype=float).reshape(-1)

    @staticmethod
    def _restore_input_type(
        x: np.ndarray,
        scalar_input: bool
    ) -> float | np.ndarray:
        if scalar_input:
            return float(np.asarray(x, dtype=float).reshape(-1)[0])

        return np.asarray(x, dtype=float)
    
    def __call__(
        self,
        x,
        direction,
        gradient_current,
        f: ScalarToScalarFunc | VectorToScalarFunc | None = None,
        gradient_f: ScalarToVectorFunc | VectorToVectorFunc | None = None
    ) -> float:
        objective = f if f is not None else self.f

        if objective is None:
            raise ValueError("LineSearch requires an objective function.")

        gradient_provider = gradient_f if gradient_f is not None else self.gradient_f

        scalar_input = self._is_scalar_input(x)
        x_vec = self._as_vector(x)
        direction_vec = self._as_vector(direction)
        gradient_current_vec = self._as_vector(gradient_current)

        def objective_vec(z: Vector) -> float:
            z_vec = self._as_vector(z)

            if scalar_input:
                objective_scalar = cast(ScalarToScalarFunc, objective)
                return float(objective_scalar(float(z_vec[0])))

            objective_vector = cast(VectorToScalarFunc, objective)
            return float(objective_vector(z_vec))

        def gradient_vec(z: Vector) -> np.ndarray:
            if gradient_provider is None:
                raise ValueError(f"{self.name} line search requires gradient_f.")

            z_vec = self._as_vector(z)

            if scalar_input:
                gradient_scalar = cast(ScalarToVectorFunc, gradient_provider)
                return self._as_vector(gradient_scalar(float(z_vec[0])))

            gradient_vector = cast(VectorToVectorFunc, gradient_provider)
            return self._as_vector(gradient_vector(z_vec))

        f_current = objective_vec(x_vec)
        initial_slope = self._directional_derivative(gradient_current_vec, direction_vec)

        if initial_slope >= 0.0:
            raise ValueError(
                f"{self.name} line search requires a descent direction. "
                f"Got initial directional derivative {initial_slope}"
            )

        if self.name == "exact":
            return self._exact(objective_vec, x_vec, direction_vec)

        if self.name == "armijo":
            return self._armijo(
                objective_vec,
                x_vec,
                direction_vec,
                f_current,
                initial_slope
            )

        if self.name == "wolfe":
            return self._wolfe(
                objective_vec,
                gradient_vec,
                x_vec,
                direction_vec,
                f_current,
                initial_slope,
                strong=False
            )

        if self.name == "strong_wolfe":
            return self._wolfe(
                objective_vec,
                gradient_vec,
                x_vec,
                direction_vec,
                f_current,
                initial_slope,
                strong=True
            )

        raise ValueError(f"Unknown line search rule: {self.name}")

    def satisfies_armijo(
        self,
        f_trial: float,
        f_current: float,
        alpha: float,
        initial_slope: float,
    ) -> bool:
        """
        Checks the Armijo sufficient decrease condition.
        """
        return f_trial <= f_current + self.c_1 * alpha * initial_slope

    def satisfies_wolfe(
        self,
        f_trial: float,
        f_current: float,
        alpha: float,
        initial_slope: float,
        trial_slope: float,
    ) -> bool:
        """
        Checks the weak Wolfe conditions.
        """
        return self.satisfies_armijo(f_trial, f_current, alpha, initial_slope) and (
            trial_slope >= self.c_2 * initial_slope
        )

    def satisfies_strong_wolfe(
        self,
        f_trial: float,
        f_current: float,
        alpha: float,
        initial_slope: float,
        trial_slope: float,
    ) -> bool:
        """
        Checks the strong Wolfe conditions.
        """
        return self.satisfies_armijo(f_trial, f_current, alpha, initial_slope) and (
            abs(trial_slope) <= self.c_2 * abs(initial_slope)
        )

    def _exact(
        self,
        f: VectorToScalarFunc,
        x: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        """
        Numerical exact line search over ``[0, max_alpha]``.

        Candidate step sizes are sampled around ``alpha_0`` by shrinking and expanding with ``rho``. 
        Local candidates are then refined by golden-section search.
        """
        def phi(alpha: float) -> float:
            value = float(f(x + alpha * direction))
            if np.isfinite(value):
                return value
            return np.inf

        def add_alpha(alpha_set: set[float], alpha: float) -> None:
            if 0.0 <= alpha <= self.max_alpha and np.isfinite(alpha):
                alpha_set.add(float(alpha))

        alphas: set[float] = {0.0}

        alpha = min(self.alpha_0, self.max_alpha)
        for _ in range(self.max_iter + 1):
            add_alpha(alphas, alpha)
            alpha *= self.rho

        alpha = min(self.alpha_0, self.max_alpha)
        for _ in range(self.max_iter):
            next_alpha = min(alpha / self.rho, self.max_alpha)

            if next_alpha <= alpha:
                break

            add_alpha(alphas, next_alpha)

            if next_alpha >= self.max_alpha:
                break

            alpha = next_alpha

        positive_alphas = sorted(alpha for alpha in alphas if alpha > 0.0)
        for left, right in zip(positive_alphas, positive_alphas[1:]):
            if right <= left:
                continue

            add_alpha(alphas, np.sqrt(left * right))
            add_alpha(alphas, 0.5 * (left + right))

        samples = sorted((alpha, phi(alpha)) for alpha in alphas)
        best_alpha, best_value = min(samples, key=lambda item: item[1])

        for i in range(1, len(samples) - 1):
            left_alpha, left_value = samples[i - 1]
            mid_alpha, mid_value = samples[i]
            right_alpha, right_value = samples[i + 1]

            is_local_candidate = (
                mid_value <= left_value
                and mid_value <= right_value
                and (mid_value < left_value or mid_value < right_value)
            )

            if not is_local_candidate:
                continue

            candidate_alpha = self._golden_section(phi, left_alpha, right_alpha)
            candidate_value = phi(candidate_alpha)

            if candidate_value < best_value:
                best_alpha = candidate_alpha
                best_value = candidate_value

        return best_alpha


    def _golden_section(
        self,
        phi: Callable[[float], float],
        lower: float,
        upper: float,
    ) -> float:
        a = float(lower)
        b = float(upper)

        if b <= a:
            return a

        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
        inv_phi_sq = (3.0 - np.sqrt(5.0)) / 2.0

        h = b - a
        c = a + inv_phi_sq * h
        d = a + inv_phi * h
        f_c = phi(c)
        f_d = phi(d)

        for _ in range(self.max_iter):
            if abs(b - a) <= np.sqrt(np.finfo(float).eps) * max(1.0, abs(a), abs(b)):
                break

            if f_c <= f_d:
                b = d
                d = c
                f_d = f_c
                h = b - a
                c = a + inv_phi_sq * h
                f_c = phi(c)
            else:
                a = c
                c = d
                f_c = f_d
                h = b - a
                d = a + inv_phi * h
                f_d = phi(d)

        return 0.5 * (a + b)

    def _armijo(
        self,
        f: VectorToScalarFunc,
        x: np.ndarray,
        direction: np.ndarray,
        f_current: float,
        initial_slope: float,
    ) -> float:
        """
        Backtracking Armijo line search.

        The trial step starts at ``alpha_0`` and is multiplied by ``rho`` until ``f(x + alpha * direction)`` satisfies the Armijo sufficient decrease condition, or the maximum number of line-search iterations is reached.
        """
        alpha = self.alpha_0

        for _ in range(self.max_iter):
            f_trial = float(f(x + alpha * direction))

            if self.satisfies_armijo(f_trial, f_current, alpha, initial_slope):
                return alpha

            alpha *= self.rho

        return self._failed(alpha)

    def _wolfe(
        self,
        f: VectorToScalarFunc,
        gradient_f: VectorToVectorFunc,
        x: np.ndarray,
        direction: np.ndarray,
        f_current: float,
        initial_slope: float,
        strong: bool
    ) -> float:
        alpha = self.alpha_0
        alpha_low = 0.0
        alpha_high: float | None = None

        for _ in range(self.max_iter):
            x_trial = x + alpha * direction
            f_trial = float(f(x_trial))

            if not self.satisfies_armijo(f_trial, f_current, alpha, initial_slope):
                alpha_high = alpha
                alpha = self._next_smaller_alpha(alpha_low, alpha_high, alpha)
                continue

            gradient_trial = np.asarray(gradient_f(x_trial), dtype=float)
            trial_slope = self._directional_derivative(gradient_trial, direction)

            if strong:
                if self.satisfies_strong_wolfe(
                    f_trial, f_current, alpha, initial_slope, trial_slope
                ):
                    return alpha
            elif self.satisfies_wolfe(
                f_trial, f_current, alpha, initial_slope, trial_slope
            ):
                return alpha

            if trial_slope < 0.0:
                alpha_low = alpha
                alpha = self._next_larger_alpha(alpha_low, alpha_high)
            else:
                alpha_high = alpha
                alpha = self._next_smaller_alpha(alpha_low, alpha_high, alpha)

        return self._failed(alpha)

    def _next_smaller_alpha(
        self,
        alpha_low: float,
        alpha_high: float,
        alpha: float,
    ) -> float:
        if alpha_low > 0.0:
            return 0.5 * (alpha_low + alpha_high)

        return alpha * self.rho

    def _next_larger_alpha(
        self,
        alpha_low: float,
        alpha_high: float | None,
    ) -> float:
        if alpha_high is None:
            return min(alpha_low / self.rho, self.max_alpha)

        return 0.5 * (alpha_low + alpha_high)

    def _failed(self, alpha: float) -> float:
        if self.raise_on_fail:
            raise LineSearchError(
                f"Line search '{self.name}' failed to satisfy its condition."
            )

        return alpha

    @staticmethod
    def _directional_derivative(gradient: np.ndarray, direction: np.ndarray) -> float:
        if gradient.shape != direction.shape:
            raise ValueError(
                "Shape mismatch: "
                f"gradient has shape {gradient.shape}, direction has shape {direction.shape}"
            )

        return float(np.dot(gradient.ravel(), direction.ravel()))