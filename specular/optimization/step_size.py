"""Step schedules and bounded line searches for optimization.

Factories create independent callables. Iterations start at one; a returned
step is always finite and strictly positive. Line searches use raw gradients,
never normalized search directions, to evaluate their slope conditions.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from ..calculation import derivative as _specular_derivative
from ..calculation import gradient as _specular_gradient

__all__ = ["make_step_size", "LineSearchError"]


class LineSearchError(RuntimeError):
    """A line search could not accept a finite, positive step."""


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar")
    return float(array)


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    result = _scalar(value, name)
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        comparison = "nonnegative" if zero else "positive"
        raise ValueError(f"{name} must be finite and {comparison}")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _unknown(options: dict[str, Any], allowed: set[str]) -> None:
    unexpected = options.keys() - allowed
    if unexpected:
        raise TypeError(f"Unknown step-size option(s): {', '.join(sorted(unexpected))}")


def _point(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real scalar or nonempty vector")
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _argument(array: np.ndarray) -> float | np.ndarray:
    return float(array) if array.ndim == 0 else array.copy()


def _centered_gradient(f: Callable, x: Any, h: float | None) -> Any:
    point = np.asarray(x, dtype=float)
    flat = point.reshape(-1)
    result = np.empty_like(flat)
    for index, value in enumerate(flat):
        step = h if h is not None else np.cbrt(np.finfo(float).eps) * max(1.0, abs(value))
        plus, minus = flat.copy(), flat.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            plus[index], minus[index] = value + step, value - step
        if not np.isfinite(plus[index]) or not np.isfinite(minus[index]):
            raise LineSearchError("Finite-difference sample is not finite")
        if plus[index] == value or minus[index] == value:
            raise LineSearchError("Finite-difference step does not change the point")
        fp = _scalar(f(_argument(plus.reshape(point.shape))), "objective value")
        fm = _scalar(f(_argument(minus.reshape(point.shape))), "objective value")
        numerator = fp - fm
        denominator = float(plus[index]) - float(minus[index])
        if math.isinf(denominator):
            result[index] = (fp / 2 - fm / 2) / (float(plus[index]) / 2 - float(minus[index]) / 2)
        elif math.isinf(numerator):
            result[index] = fp / denominator - fm / denominator
        else:
            result[index] = numerator / denominator
    return _argument(result.reshape(point.shape))


def make_step_size(
    rule: str | float | Callable = "constant",
    *,
    f: Callable | None = None,
    gradient: Callable | None = None,
    **options: Any,
) -> Callable:
    """Build ``step(n, x=None, d=None, *, gradient_value=None, f_value=None)``.

    Schedules accept a positive number, a callable of ``n``, or the names
    ``constant``, ``not_summable``, ``square_summable_not_summable``,
    ``geometric_series``, and ``user_defined``. Their formulas are respectively
    ``a``, ``a/sqrt(n)``, ``a/(b+n)``, and ``a*r**n``; ``n >= 1``.

    Line searches are ``armijo``, ``wolfe``, ``strong_wolfe``, and ``exact``
    (case-insensitive). They require ``f``; Armijo/Wolfe additionally require
    a descent direction. A supplied
    ``gradient`` callback always takes precedence. Without one, ordinary
    centered differences are used, or specular differences for names prefixed
    with ``specular_``. Cached values must match that raw gradient and objective.

    Common line-search options are ``t_0=1``, ``max_alpha=1e8``,
    ``max_iter=100``, and ``h=None``. Armijo/Wolfe use ``c_1=1e-4`` and
    ``rho=0.5``; Wolfe also uses ``c_2=0.9`` (``c_3`` is a legacy alias for
    strong Wolfe). ``exact`` uses golden-section minimization on
    ``[0, max_alpha]`` with ``tol=1e-8`` and at most ``max_iter`` objective
    evaluations, including the initial value when uncached. It is derivative
    free; ``specular_exact`` is an identical alias. The name describes
    numerical bounded minimization, not a global or symbolic guarantee.

    Failed searches raise :class:`LineSearchError`; no unchecked fallback step
    is returned. Specular slopes do not imply classical smooth Wolfe guarantees.
    """
    name = rule.strip().lower() if isinstance(rule, str) else None
    schedules = {
        "constant", "not_summable", "square_summable_not_summable",
        "geometric_series", "user_defined",
    }
    if name is None or name in schedules:
        if callable(rule):
            _unknown(options, set())
            schedule = rule
        elif name is None:
            _unknown(options, set())
            value = _positive(rule, "step size")
            schedule = lambda n: value
        elif name == "user_defined":
            _unknown(options, {"user_defined_rule"})
            schedule = options.get("user_defined_rule")
            if not callable(schedule):
                raise TypeError("user_defined_rule must be callable")
        else:
            allowed = {"a"}
            if name == "square_summable_not_summable":
                allowed.add("b")
            if name == "geometric_series":
                allowed.add("r")
            _unknown(options, allowed)
            a = _positive(options.get("a", 1.0), "a")
            if name == "constant":
                schedule = lambda n: a
            elif name == "not_summable":
                schedule = lambda n: a / math.sqrt(n)
            elif name == "square_summable_not_summable":
                b = _positive(options.get("b", 1.0), "b", zero=True)
                schedule = lambda n: a / (b + n)
            else:
                r = _positive(options.get("r", 0.5), "r")
                if r >= 1:
                    raise ValueError("r must be strictly between zero and one")
                schedule = lambda n: a * r**n

        def scheduled(n, x=None, d=None, *, gradient_value=None, f_value=None):
            return _positive(schedule(_integer(n, "n")), "scheduled step")

        return scheduled

    specular = name.startswith("specular_")
    base = name.removeprefix("specular_")
    if base not in {"armijo", "wolfe", "strong_wolfe", "exact"}:
        raise ValueError(f"Unknown step-size rule: {rule!r}")
    if not callable(f):
        raise TypeError("Line searches require a callable f")
    if gradient is not None and not callable(gradient):
        raise TypeError("gradient must be callable")

    allowed = {"t_0", "max_alpha", "max_iter", "h"}
    if base == "exact":
        allowed.add("tol")
    else:
        allowed.update({"c_1", "rho"})
        if base != "armijo":
            allowed.add("c_2")
        if base == "strong_wolfe":
            allowed.add("c_3")
    _unknown(options, allowed)
    initial = _positive(options.get("t_0", 1.0), "t_0")
    bound = _positive(options.get("max_alpha", 1e8), "max_alpha")
    budget = _integer(options.get("max_iter", 100), "max_iter")
    h = options.get("h")
    if h is not None:
        h = _positive(h, "h")
    c1 = _positive(options.get("c_1", 1e-4), "c_1")
    rho = _positive(options.get("rho", 0.5), "rho")
    if c1 >= 1 or rho >= 1:
        raise ValueError("c_1 and rho must be strictly between zero and one")
    if "c_2" in options and "c_3" in options:
        raise TypeError("Use only one of c_2 and its legacy alias c_3")
    c2 = _positive(options.get("c_2", options.get("c_3", 0.9)), "c_2")
    if base in {"wolfe", "strong_wolfe"} and not c1 < c2 < 1:
        raise ValueError("Wolfe parameters must satisfy 0 < c_1 < c_2 < 1")
    tolerance = _positive(options.get("tol", 1e-8), "tol")

    if gradient is None:
        if specular:
            def raw_gradient(point):
                calc = _specular_derivative if np.asarray(point).ndim == 0 else _specular_gradient
                return calc(f, point, h=h)
        else:
            def raw_gradient(point):
                return _centered_gradient(f, point, h)
    else:
        raw_gradient = gradient

    def search(n, x=None, d=None, *, gradient_value=None, f_value=None):
        _integer(n, "n")
        point, direction = _point(x, "x"), _point(d, "d")
        if point.shape != direction.shape:
            raise ValueError("x and d must have the same shape")
        if not np.any(direction):
            raise LineSearchError("Line search requires a nonzero descent direction")

        def slope(at, cached=None):
            try:
                value = raw_gradient(_argument(at)) if cached is None else cached
            except ArithmeticError:
                return math.nan
            array = np.asarray(value)
            if array.shape != point.shape or array.dtype.kind not in "iuf":
                raise ValueError("Raw gradient must be real and have the same shape as x")
            if not np.all(np.isfinite(array)):
                return math.nan
            with np.errstate(over="ignore", invalid="ignore"):
                return float(np.sum(array.astype(np.longdouble) * direction.astype(np.longdouble)))

        try:
            phi0 = _scalar(f(_argument(point)) if f_value is None else f_value, "objective value")
        except ArithmeticError as exc:
            raise LineSearchError("Initial objective value must be finite") from exc
        if not math.isfinite(phi0):
            raise LineSearchError("Initial objective value must be finite")
        if base != "exact":
            slope0 = slope(point, gradient_value)
            if not math.isfinite(slope0):
                raise LineSearchError("Initial slope must be finite")
            if slope0 >= 0:
                raise LineSearchError("Line search requires a descent direction (negative raw slope)")

        def trial(alpha):
            try:
                at = np.array([
                    math.fma(alpha, float(di), float(xi))
                    for xi, di in zip(point.reshape(-1), direction.reshape(-1))
                ]).reshape(point.shape)
            except OverflowError:
                return np.full_like(point, math.inf), math.inf
            if not np.all(np.isfinite(at)) or np.array_equal(at, point):
                return at, math.inf
            try:
                value = _scalar(f(_argument(at)), "objective value")
            except ArithmeticError:
                return at, math.inf
            return at, value if math.isfinite(value) else math.inf

        def armijo(alpha, phi):
            threshold = np.longdouble(phi0) + np.longdouble(c1) * alpha * slope0
            return math.isfinite(phi) and phi <= threshold

        def curvature(value):
            if not math.isfinite(value):
                return False
            if base == "strong_wolfe":
                return abs(value) <= -c2 * slope0
            return value >= c2 * slope0

        if base == "exact":
            evaluations = 0 if f_value is not None else 1
            best_alpha, best_value = 0.0, phi0

            def evaluate(alpha):
                nonlocal evaluations, best_alpha, best_value
                if evaluations >= budget:
                    raise LineSearchError("Bounded minimization evaluation budget exhausted")
                evaluations += 1
                _, value = trial(alpha)
                if math.isfinite(value) and (value < best_value or (value == best_value and best_alpha == 0)):
                    best_alpha, best_value = alpha, value
                return value

            left, right = 0.0, bound
            evaluate(right)
            if initial < bound:
                evaluate(initial)
            ratio = (math.sqrt(5.0) - 1.0) / 2.0
            c, e = right - ratio * (right - left), left + ratio * (right - left)
            fc, fe = evaluate(c), evaluate(e)
            while right - left > tolerance * max(1.0, abs(left + (right - left) / 2.0)):
                previous = (left, right)
                if fc <= fe:
                    right, e, fe = e, c, fc
                    c = right - ratio * (right - left)
                    fc = evaluate(c)
                else:
                    left, c, fc = c, e, fe
                    e = left + ratio * (right - left)
                    fe = evaluate(e)
                if (left, right) == previous:
                    raise LineSearchError("Bounded minimization stagnated at floating-point precision")
            if best_alpha <= 0:
                raise LineSearchError("Bounded minimization found no positive improving step")
            return best_alpha

        alpha = min(initial, bound)
        if base == "armijo":
            for _ in range(budget):
                at, phi = trial(alpha)
                if np.array_equal(at, point):
                    raise LineSearchError("Armijo step stagnated at floating-point precision")
                if armijo(alpha, phi):
                    return alpha
                next_alpha = alpha * rho
                if next_alpha <= 0 or next_alpha == alpha:
                    break
                alpha = next_alpha
            raise LineSearchError("Armijo search could not accept a step")

        previous_alpha, previous_phi = 0.0, phi0
        low = high = low_phi = None
        for iteration in range(budget):
            if low is not None:
                alpha = low + (high - low) / 2.0
                if alpha == low or alpha == high or alpha <= 0:
                    raise LineSearchError("Wolfe bracket stagnated at floating-point precision")
            at, phi = trial(alpha)
            if np.array_equal(at, point):
                next_alpha = min(alpha / rho, bound)
                if low is not None or next_alpha <= alpha:
                    raise LineSearchError("Wolfe step stagnated at floating-point precision")
                alpha = next_alpha
                continue
            if not armijo(alpha, phi) or (low is not None and phi >= low_phi):
                if low is None:
                    low, low_phi = previous_alpha, previous_phi
                high = alpha
                continue
            if low is None and iteration > 0 and phi >= previous_phi:
                low, low_phi, high = previous_alpha, previous_phi, alpha
                continue
            value = slope(at)
            if curvature(value):
                return alpha
            if not math.isfinite(value):
                if low is None:
                    low, low_phi = previous_alpha, previous_phi
                high = alpha
                continue
            if low is not None:
                if (value >= 0) == (high >= low):
                    high = low
                low, low_phi = alpha, phi
            elif value >= 0:
                low, low_phi, high = alpha, phi, previous_alpha
            else:
                previous_alpha, previous_phi = alpha, phi
                next_alpha = min(alpha / rho, bound)
                if next_alpha <= alpha:
                    raise LineSearchError("Wolfe search reached max_alpha without acceptable curvature")
                alpha = next_alpha
        raise LineSearchError("Wolfe search exhausted its iteration budget")

    return search
