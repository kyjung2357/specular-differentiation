from __future__ import annotations

import numpy as np
from scipy.optimize import line_search
from scipy.optimize import approx_fprime
from typing import Callable, Any

from ..._typing import Scalar, Vector, ScalarToScalarFunc, VectorToScalarFunc
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
        'exact': ['t_k'],
        'Armijo': ['t_k', 'c_1'],
        'specular_Armijo': ['t_k', 'c_1'],
        'Wolfe': ['t_k', 'c_1', 'c_2'],
        'specular_Wolfe': ['t_k', 'c_1', 'c_2'],
        'strong_Wolfe': ['t_k', 'c_1', 'c_3'],
        'specular_strong_Wolfe': ['t_k', 'c_1', 'c_3'],
    }

    def __init__(
        self,
        name: str,
        *,
        f: ScalarToScalarFunc | VectorToScalarFunc,
        h: Scalar = 1e-6,
        t_k: Scalar = 1.0,
        t_max: Scalar = 1e8,
        c_1: Scalar = 1e-4,
        c_2: Scalar = 0.9,
        c_3: Scalar = 0.9,
        rho: Scalar = 0.5,
        zero_tol: Scalar = 1e-8,
        max_iter: int = 20,
        skip_on_fail: bool = False,
    ):
        """
        Parameters:
            name: Line search rule name (see ``_SUPPORTED_OPTIONS``).
            f: Objective function. Required for Wolfe / strong_Wolfe to compute gradients.
            h: Mesh size for finite difference gradient approximation.
            zero_tol: Tolerance for specular gradients.
            t_k: Initial trial step length.
            c_1: Armijo sufficient decrease parameter.
            c_2: Wolfe curvature parameter.
            c_3: Strong Wolfe curvature parameter.
            rho: Backtracking contraction factor.
            max_iter: Maximum number of line-search iterations.
            t_max: Maximum trial step length (used in exact & Wolfe expansion).
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
        if t_k <= 0:
            raise ValueError(f"t_k must be positive. Got {t_k}")
        if t_max <= 0:
            raise ValueError(f"t_max must be positive. Got {t_max}")
        if not (0.0 < c_1 < 1.0):
            raise ValueError(f"c_1 must be in (0, 1). Got {c_1}")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1). Got {rho}")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive. Got {max_iter}")
        
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
        
        self.f = f
        self.t_k = float(t_k)
        self.t_max = float(t_max)
        self.h = float(h)
        self.c_1 = float(c_1)
        self.c_2 = float(c_2)
        self.c_3 = float(c_3)
        self.rho = float(rho)
        self.zero_tol = float(zero_tol)
        self.max_iter = int(max_iter)
        self.skip_on_fail = skip_on_fail

    # ==== Unified interface ====

    def __call__(self, k: int, *, x, d_k, grad) -> float:
        """
        Compute the step size along direction ``d_k`` from point ``x``.

        Parameters:
            k: Iteration index (unused; accepted for interface compatibility with StepSchedule).
            x: Current point (scalar or array).
            d_k: Search direction (same type as ``x``).
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
                return float(self.f(float(z.ravel()[0])))  # type: ignore
            return float(self.f(z))  # type: ignore

        f_current = f_vec(x_vec)

        if self.name == 'exact':
            return self._exact(f_vec, x_vec, d_vec)
        elif self._base_name == 'Armijo':
            return self._armijo(f_vec, x_vec, d_vec, f_current, initial_slope)
        elif self.name == 'Wolfe':
            return self._wolfe(f_vec, x_vec, d_vec, f_current, initial_slope)
        elif self.name == 'specular_Wolfe':
            return self._specular_wolfe(f_vec, x_vec, d_vec, f_current, initial_slope)
        elif self.name == 'strong_Wolfe':
            return self._strong_wolfe(f_vec, x_vec, d_vec, f_current)
        elif self.name == 'specular_strong_Wolfe':
            return self._specular_strong_wolfe(f_vec, x_vec, d_vec, f_current, initial_slope)
        else:
            raise ValueError(
                f"Unknown line search '{self.name}'. "
                f"Options: {list(self._SUPPORTED_OPTIONS.keys())}"
            )

    # ==== Condition checks ====

    def _satisfies_armijo(self, f_trial: float, f_current: float, t: float, initial_slope: float) -> bool:
        return f_trial <= f_current + self.c_1 * t * initial_slope

    def _satisfies_wolfe(self, f_trial: float, f_current: float, t: float, initial_slope: float, trial_slope: float) -> bool:
        return (
            self._satisfies_armijo(f_trial, f_current, t, initial_slope)
            and trial_slope >= self.c_2 * initial_slope
        )

    def _satisfies_strong_wolfe(self, f_trial: float, f_current: float, t: float, initial_slope: float, trial_slope: float) -> bool:
        return (
            self._satisfies_armijo(f_trial, f_current, t, initial_slope)
            and abs(trial_slope) <= self.c_3 * abs(initial_slope)
        )

    # ==== Algorithms ====

    def _armijo(self, f, x, d_k, f_current, initial_slope) -> float:
        """
        Backtracking Armijo line search.

        Starts at ``t_k`` and contracts by ``rho`` until the sufficient decrease
        condition is satisfied or ``max_iter`` is reached.
        """
        t = self.t_k

        for _ in range(self.max_iter):
            f_trial = float(f(x + t * d_k))

            if self._satisfies_armijo(f_trial, f_current, t, initial_slope):
                return t

            t *= self.rho

        return self._updated_t_k(t)

    def _wolfe(self, f, x, direction, f_current, initial_slope) -> float:
        """
        Zoom-based Wolfe / strong Wolfe line search.
        """
        t = self.t_k
        t_min = 0.0
        t_max = self.t_max

        for _ in range(self.max_iter):
            x_trial = x + t * direction
            f_trial = f(x_trial)

            if not self._satisfies_armijo(f_trial, f_current, t, initial_slope):
                t_max = t
                t = self._next_smaller(0.0, t_max, t)
                continue
                
            grad_trial = approx_fprime(x_trial, f, epsilon=self.h)
            trial_slope = float(grad_trial @ direction)
            
            if self._satisfies_wolfe(f_trial, f_current, t, initial_slope, trial_slope):
                return t

            if trial_slope < 0.0:
                t_min = t
                t = self._next_larger(t_min, t_max)
            else:
                t_max = t
                t = self._next_smaller(t_min, t_max, t)

        return self._updated_t_k(t)
    
    def _strong_wolfe(self, f, x, d_k, f_current) -> float:
        """
        Scipy-based strong Wolfe line search.
        """
        gradient_f = lambda w: approx_fprime(w, f, epsilon=self.h)

        t, fc, gc, new_fval, old_fval, new_slope = line_search(
            f, 
            gradient_f, 
            xk=x, 
            pk=d_k, 
            gfk=None,
            old_fval=f_current, 
            c1=self.c_1, 
            c2=self.c_2,
            amax=self.t_max,
            maxiter=self.max_iter
        )
        
        if t is None:
            return self._updated_t_k(self.t_max)
            
        return float(t)
    
    def _specular_wolfe(self, f, x, d_k, f_current, initial_slope) -> float:
        """
        Zoom-based Wolfe / strong Wolfe line search.
        """
        t = self.t_k
        t_min = 0.0
        t_max = self.t_max

        for _ in range(self.max_iter):
            x_trial = x + t * d_k
            f_trial = float(f(x_trial))

            if not self._satisfies_armijo(f_trial, f_current, t, initial_slope):
                t_max = t
                t = self._next_smaller(0.0, t_max, t)
                continue
                
            if x_trial.size == 1:
                val = float(x_trial.ravel()[0])
                specular_grad_trial = np.array(derivative(f, val, self.h))
            else:
                specular_grad_trial = gradient(f, x_trial, self.h, self.zero_tol)
            
            trial_slope = float(specular_grad_trial @ d_k)
            
            if self._satisfies_wolfe(f_trial, f_current, t, initial_slope, trial_slope):
                return t

            if trial_slope < 0.0:
                t_min = t
                t = self._next_larger(t_min, t_max)
            else:
                t_max = t
                t = self._next_smaller(t_min, t_max, t)

        return self._updated_t_k(t)

    def _specular_strong_wolfe(self, f, x, d_k, f_current, initial_slope) -> float:
        """
        Zoom-based Wolfe / strong Wolfe line search.
        """
        t = self.t_k
        t_min = 0.0
        t_max: float | None = None

        for _ in range(self.max_iter):
            x_trial = x + t * d_k
            f_trial = float(f(x_trial))

            if not self._satisfies_armijo(f_trial, f_current, t, initial_slope):
                t_max = t
                t = self._next_smaller(t_min, t_max, t)
                continue
            
            if x_trial.size == 1:
                val = float(x_trial.ravel()[0])
                specular_grad_trial = np.array(derivative(f, val, self.h, self.zero_tol))
            else:
                specular_grad_trial = gradient(f, x_trial, self.h)
            
            trial_slope = float(specular_grad_trial @ d_k)

            if self._satisfies_strong_wolfe(f_trial, f_current, t, initial_slope, trial_slope):
                return t

            if trial_slope < 0.0:
                t_min = t
                t = self._next_larger(t_min, t_max)
            else:
                t_max = t
                t = self._next_smaller(t_min, t_max, t)

        return self._updated_t_k(t)
    
    def _exact(self, f, x, direction) -> float:
        """
        Numerical exact line search over ``[0, t_max]``.

        Candidate step sizes are sampled around ``t_k`` by shrinking and expanding with ``rho``.
        Local minima are refined by golden-section search.
        """
        def phi(t: float) -> float:
            value = float(f(x + t * direction))
            return value if np.isfinite(value) else np.inf

        def _add(t_set: set[float], t: float) -> None:
            if 0.0 <= t <= self.t_max and np.isfinite(t):
                t_set.add(float(t))

        candidates: set[float] = {0.0}

        # Shrink from t_k.
        t = min(self.t_k, self.t_max)
        for _ in range(self.max_iter + 1):
            _add(candidates, t)
            t *= self.rho

        # Expand from t_k.
        t = min(self.t_k, self.t_max)
        for _ in range(self.max_iter):
            next_t = min(t / self.rho, self.t_max)
            if next_t <= t:
                break
            _add(candidates, next_t)
            if next_t >= self.t_max:
                break
            t = next_t

        # Geometric and arithmetic midpoints.
        positive = sorted(a for a in candidates if a > 0.0)
        for left, right in zip(positive, positive[1:]):
            if right > left:
                _add(candidates, float(np.sqrt(left * right)))
                _add(candidates, 0.5 * (left + right))

        # Evaluate all candidates.
        samples = sorted((a, phi(a)) for a in candidates)
        best_t, best_value = min(samples, key=lambda s: s[1])

        # Refine local minima with golden-section search.
        for i in range(1, len(samples) - 1):
            left_val = samples[i - 1][1]
            mid_t, mid_val = samples[i]
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
                best_t, best_value = refined, refined_val

        return best_t

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

    def _next_smaller(self, t_min: float, t_max: float, t: float) -> float:
        if t_min > 0.0:
            return 0.5 * (t_min + t_max)
        return t * self.rho

    def _next_larger(self, t_min: float, t_max: float | None) -> float:
        if t_max is None:
            return min(t_min / self.rho, self.t_max)
        return 0.5 * (t_min + t_max)

    def _updated_t_k(self, t: float) -> float:
        if not self.skip_on_fail:
            raise LineSearchError(
                f"Line search '{self.name}' failed to satisfy its condition."
            )
        return t