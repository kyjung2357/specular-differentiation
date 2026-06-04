from __future__ import annotations

import numpy as np
from typing import Callable, Any

from ...calculation import derivative, gradient


class LineSearchError(RuntimeError):
    """Raised when a line search fails to satisfy its condition."""


class LineSearch:
    """
    Line search methods for choosing the step length along a search direction.

    The ``specular_`` prefix determines which gradient function is injected
    from the caller; the algorithm itself is identical to the unprefixed version.
    """

    _SUPPORTED_OPTIONS = {
        'exact': ['t_0'],
        'Armijo': ['t_0', 'c_1'],
        'specular_Armijo': ['t_0', 'c_1'],
        'Wolfe': ['t_0', 'c_1', 'c_2'],
        'specular_Wolfe': ['t_0', 'c_1', 'c_2'],
        'strong_Wolfe': ['t_0', 'c_1', 'c_3'],
        'specular_strong_Wolfe': ['t_0', 'c_1', 'c_3'],
    }

    def __init__(
        self,
        name: str,
        *,
        f: Callable | None = None,
        h: float = 1e-6,
        zero_tol: float = 1e-8,
        t_0: float = 1.0,
        c_1: float = 1e-4,
        c_2: float = 0.9,
        c_3: float = 0.9,
        rho: float = 0.5,
        max_iter: int = 20,
        max_alpha: float = 1e8,
        skip_on_fail: bool = False,
    ):
        """
        Parameters:
            name: Line search rule name (see ``_SUPPORTED_OPTIONS``).
            f: Objective function. Required for Wolfe / strong_Wolfe to compute gradients.
            h: Mesh size for finite difference gradient approximation.
            zero_tol: Tolerance for specular gradients.
            t_0: Initial trial step length.
            c_1: Armijo sufficient decrease parameter.
            c_2: Wolfe curvature parameter.
            c_3: Strong Wolfe curvature parameter.
            rho: Backtracking contraction factor.
            max_iter: Maximum number of line-search iterations.
            max_alpha: Maximum trial step length (used in exact & Wolfe expansion).
            skip_on_fail: If ``True``, return the last trial step instead of raising.
        """
        if name not in self._SUPPORTED_OPTIONS:
            raise ValueError(
                f"Unknown line search '{name}'. "
                f"Options: {list(self._SUPPORTED_OPTIONS.keys())}"
            )

        # Strip prefix to get base algorithm name.
        self._base_name = name.removeprefix("specular_")
        self.name = name

        # ---- Validation ----
        if t_0 <= 0:
            raise ValueError(f"t_0 must be positive. Got {t_0}")
        if not (0.0 < c_1 < 1.0):
            raise ValueError(f"c_1 must be in (0, 1). Got {c_1}")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1). Got {rho}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive. Got {max_iter}")
        if max_alpha <= 0:
            raise ValueError(f"max_alpha must be positive. Got {max_alpha}")

        if self._base_name == "Wolfe":
            if not (0.0 < c_2 < 1.0):
                raise ValueError(f"c_2 must be in (0, 1). Got {c_2}")
            if not (c_1 < c_2):
                raise ValueError(f"Wolfe requires c_1 < c_2. Got c_1={c_1}, c_2={c_2}")

        if self._base_name == "strong_Wolfe":
            if not (0.0 < c_3 < 1.0):
                raise ValueError(f"c_3 must be in (0, 1). Got {c_3}")
            if not (c_1 < c_3):
                raise ValueError(f"strong_Wolfe requires c_1 < c_3. Got c_1={c_1}, c_3={c_3}")

        if self._base_name in ("Wolfe", "strong_Wolfe"):
            if f is None:
                raise ValueError(f"'{name}' requires objective function 'f' to compute gradients.")
            
            if name.startswith("specular_"):
                def _specular_grad(x_: float | np.ndarray):
                    if isinstance(x_, float):
                        return derivative(f, x_, h=h, zero_tol=zero_tol)
                    return gradient(f, x_, h=h, zero_tol=zero_tol)
                self.grad_fn = _specular_grad
            else:
                def _classical_grad(x_: float | np.ndarray):
                    if isinstance(x_, float):
                        return float((f(x_ + h) - f(x_ - h)) / (2.0 * h))
                    x_arr = np.asarray(x_, dtype=float)
                    g = np.empty_like(x_arr)
                    for i in range(x_arr.size):
                        e = np.zeros_like(x_arr)
                        e[i] = h
                        g[i] = (f(x_arr + e) - f(x_arr - e)) / (2.0 * h)
                    return g
                self.grad_fn = _classical_grad
        else:
            self.grad_fn = None
            
        self.t_0 = float(t_0)
        self.c_1 = float(c_1)
        self.c_2 = float(c_2)
        self.c_3 = float(c_3)
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.max_alpha = float(max_alpha)
        self.skip_on_fail = skip_on_fail

    # ==== Unified interface ====

    def __call__(self, k: int, *, x, d_k, grad, f) -> float:
        """
        Compute the step size along direction ``d_k`` from point ``x``.

        Parameters:
            k: Iteration index (unused; accepted for interface compatibility with StepSchedule).
            x: Current point (scalar or array).
            d_k: Search direction (same type as ``x``).
            grad: Gradient at ``x`` (same type as ``x``).
            f: Objective function.

        Returns:
            Step size ``t > 0``.
        """
        scalar_input = np.isscalar(x) or np.asarray(x).ndim == 0

        x_vec = np.asarray(x, dtype=float).ravel()
        d_vec = np.asarray(d_k, dtype=float).ravel()
        g_vec = np.asarray(grad, dtype=float).ravel()

        initial_slope = float(np.dot(g_vec, d_vec))

        if initial_slope >= 0.0:
            raise ValueError(
                f"Line search requires a descent direction. "
                f"Got directional derivative = {initial_slope}"
            )

        # Wrap f to always accept a vector internally.
        def f_vec(z: np.ndarray) -> float:
            if scalar_input:
                return float(f(float(z.ravel()[0])))
            return float(f(z))

        f_current = f_vec(x_vec)

        if self._base_name == "exact":
            return self._exact(f_vec, x_vec, d_vec)

        if self._base_name == "Armijo":
            return self._armijo(f_vec, x_vec, d_vec, f_current, initial_slope)

        # Wolfe / strong_Wolfe need a vectorised gradient wrapper.
        def grad_vec(z: np.ndarray) -> np.ndarray:
            if scalar_input:
                return np.asarray(self.grad_fn(float(z.ravel()[0])), dtype=float).ravel()  # type: ignore[misc]
            return np.asarray(self.grad_fn(z), dtype=float).ravel()  # type: ignore[misc]

        if self._base_name == "Wolfe":
            return self._wolfe(
                f_vec, grad_vec, x_vec, d_vec, f_current, initial_slope, strong=False,
            )

        if self._base_name == "strong_Wolfe":
            return self._wolfe(
                f_vec, grad_vec, x_vec, d_vec, f_current, initial_slope, strong=True,
            )

        raise ValueError(f"Unknown base algorithm: {self._base_name}")

    # ==== Condition checks ====

    def _satisfies_armijo(self, f_trial: float, f_current: float, alpha: float, initial_slope: float) -> bool:
        return f_trial <= f_current + self.c_1 * alpha * initial_slope

    def _satisfies_wolfe(self, f_trial: float, f_current: float, alpha: float, initial_slope: float, trial_slope: float) -> bool:
        return (
            self._satisfies_armijo(f_trial, f_current, alpha, initial_slope)
            and trial_slope >= self.c_2 * initial_slope
        )

    def _satisfies_strong_wolfe(self, f_trial: float, f_current: float, alpha: float, initial_slope: float, trial_slope: float) -> bool:
        return (
            self._satisfies_armijo(f_trial, f_current, alpha, initial_slope)
            and abs(trial_slope) <= self.c_3 * abs(initial_slope)
        )

    # ==== Algorithms ====

    def _armijo(self, f, x, direction, f_current, initial_slope) -> float:
        """
        Backtracking Armijo line search.

        Starts at ``t_0`` and contracts by ``rho`` until the sufficient decrease
        condition is satisfied or ``max_iter`` is reached.
        """
        t = self.t_0

        for _ in range(self.max_iter):
            f_trial = float(f(x + t * direction))

            if self._satisfies_armijo(f_trial, f_current, t, initial_slope):
                return t

            t *= self.rho

        return self._failed(t)

    def _wolfe(self, f, gradient_f, x, direction, f_current, initial_slope, *, strong: bool) -> float:
        """
        Zoom-based Wolfe / strong Wolfe line search.
        """
        t = self.t_0
        t_low = 0.0
        t_high: float | None = None

        for _ in range(self.max_iter):
            x_trial = x + t * direction
            f_trial = float(f(x_trial))

            if not self._satisfies_armijo(f_trial, f_current, t, initial_slope):
                t_high = t
                t = self._next_smaller(t_low, t_high, t)
                continue

            grad_trial = np.asarray(gradient_f(x_trial), dtype=float).ravel()
            trial_slope = float(np.dot(grad_trial, direction))

            if strong:
                if self._satisfies_strong_wolfe(f_trial, f_current, t, initial_slope, trial_slope):
                    return t
            elif self._satisfies_wolfe(f_trial, f_current, t, initial_slope, trial_slope):
                return t

            if trial_slope < 0.0:
                t_low = t
                t = self._next_larger(t_low, t_high)
            else:
                t_high = t
                t = self._next_smaller(t_low, t_high, t)

        return self._failed(t)

    def _exact(self, f, x, direction) -> float:
        """
        Numerical exact line search over ``[0, max_alpha]``.

        Candidate step sizes are sampled around ``t_0`` by shrinking and
        expanding with ``rho``.  Local minima are refined by golden-section search.
        """
        def phi(alpha: float) -> float:
            value = float(f(x + alpha * direction))
            return value if np.isfinite(value) else np.inf

        def _add(alpha_set: set[float], alpha: float) -> None:
            if 0.0 <= alpha <= self.max_alpha and np.isfinite(alpha):
                alpha_set.add(float(alpha))

        candidates: set[float] = {0.0}

        # Shrink from t_0.
        alpha = min(self.t_0, self.max_alpha)
        for _ in range(self.max_iter + 1):
            _add(candidates, alpha)
            alpha *= self.rho

        # Expand from t_0.
        alpha = min(self.t_0, self.max_alpha)
        for _ in range(self.max_iter):
            next_alpha = min(alpha / self.rho, self.max_alpha)
            if next_alpha <= alpha:
                break
            _add(candidates, next_alpha)
            if next_alpha >= self.max_alpha:
                break
            alpha = next_alpha

        # Geometric and arithmetic midpoints.
        positive = sorted(a for a in candidates if a > 0.0)
        for left, right in zip(positive, positive[1:]):
            if right > left:
                _add(candidates, float(np.sqrt(left * right)))
                _add(candidates, 0.5 * (left + right))

        # Evaluate all candidates.
        samples = sorted((a, phi(a)) for a in candidates)
        best_alpha, best_value = min(samples, key=lambda s: s[1])

        # Refine local minima with golden-section search.
        for i in range(1, len(samples) - 1):
            left_val = samples[i - 1][1]
            mid_alpha, mid_val = samples[i]
            right_val = samples[i + 1][1]

            is_local = (
                mid_val <= left_val
                and mid_val <= right_val
                and (mid_val < left_val or mid_val < right_val)
            )
            if not is_local:
                continue

            refined = self._golden_section(phi, samples[i - 1][0], samples[i + 1][0])
            refined_val = phi(refined)

            if refined_val < best_value:
                best_alpha, best_value = refined, refined_val

        return best_alpha

    def _golden_section(self, phi: Callable[[float], float], lower: float, upper: float) -> float:
        a, b = float(lower), float(upper)
        if b <= a:
            return a

        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
        inv_phi_sq = (3.0 - np.sqrt(5.0)) / 2.0

        h = b - a
        c = a + inv_phi_sq * h
        d = a + inv_phi * h
        f_c, f_d = phi(c), phi(d)

        for _ in range(self.max_iter):
            if abs(b - a) <= np.sqrt(np.finfo(float).eps) * max(1.0, abs(a), abs(b)):
                break

            if f_c <= f_d:
                b, d, f_d = d, c, f_c
                h = b - a
                c = a + inv_phi_sq * h
                f_c = phi(c)
            else:
                a, c, f_c = c, d, f_d
                h = b - a
                d = a + inv_phi * h
                f_d = phi(d)

        return 0.5 * (a + b)

    # ==== Helpers ====

    def _next_smaller(self, alpha_low: float, alpha_high: float, alpha: float) -> float:
        if alpha_low > 0.0:
            return 0.5 * (alpha_low + alpha_high)
        return alpha * self.rho

    def _next_larger(self, alpha_low: float, alpha_high: float | None) -> float:
        if alpha_high is None:
            return min(alpha_low / self.rho, self.max_alpha)
        return 0.5 * (alpha_low + alpha_high)

    def _failed(self, alpha: float) -> float:
        if not self.skip_on_fail:
            raise LineSearchError(
                f"Line search '{self.name}' failed to satisfy its condition."
            )
        return alpha