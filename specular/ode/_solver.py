"""Lightweight scalar ODE solvers based on the specular ellipse scheme."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


type RealScalar = int | float | np.integer[Any] | np.floating[Any]
type FloatArray = npt.NDArray[np.float64]
type ScalarField = Callable[[float, float], RealScalar]
type StepScale = Callable[[int, float, float, float], RealScalar]
type CoupledScale = Callable[
    [int, float, float, float, float, float], RealScalar
]


@dataclass(frozen=True, slots=True)
class ODEResult:
    """Values produced by a specular ellipse scheme.

    ``t`` and ``y`` contain the initial value and every accepted step, while
    ``sigma`` contains the scale used on each step.
    """

    t: FloatArray
    y: FloatArray
    sigma: FloatArray


def _finite_real(value: object, *, name: str) -> float:
    """Convert a real numeric scalar to a finite Python float."""

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise TypeError(f"{name} must be a real scalar") from exc

    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar")

    result = float(array)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    """Return a strictly positive integer, excluding booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_scale(value: object, *, step: int) -> float:
    """Validate a scale selected for one time step."""

    scale = _finite_real(value, name=f"sigma at step {step}")
    if scale <= 0.0:
        raise ValueError(f"sigma at step {step} must be positive")
    return scale


def _scaled_mean(alpha: float, beta: float, sigma: float) -> float:
    r"""Evaluate :math:`\mathcal C_\sigma(\alpha,\beta)` stably.

    Pairwise ``hypot`` calls retain the relative scale of ``sigma`` beside
    each slope.  Separate same- and opposite-sign formulas then avoid both
    overflowing weighted sums and cancellation near the antidiagonal.
    """

    if alpha == beta:
        return alpha
    if alpha == -beta:
        return 0.0

    radius_alpha = math.hypot(sigma, alpha)
    radius_beta = math.hypot(sigma, beta)
    same_sign = math.copysign(1.0, alpha) == math.copysign(1.0, beta)

    if same_sign:
        if radius_alpha >= radius_beta:
            ratio = radius_beta / radius_alpha
            contribution = (alpha / radius_alpha) * radius_beta
            numerator = contribution + beta
            if not math.isfinite(numerator):
                result = (
                    0.5 * contribution + 0.5 * beta
                ) / (0.5 + 0.5 * ratio)
            else:
                result = numerator / (1.0 + ratio)
        else:
            ratio = radius_alpha / radius_beta
            contribution = (beta / radius_beta) * radius_alpha
            numerator = alpha + contribution
            if not math.isfinite(numerator):
                result = (
                    0.5 * alpha + 0.5 * contribution
                ) / (0.5 + 0.5 * ratio)
            else:
                result = numerator / (1.0 + ratio)
    else:
        slope_sum = alpha + beta
        unit_alpha = alpha / radius_alpha
        unit_beta = beta / radius_beta
        inverse_alpha = sigma / radius_alpha
        inverse_beta = sigma / radius_beta
        denominator = (
            1.0
            + inverse_alpha * inverse_beta
            - unit_alpha * unit_beta
        )

        sum_mantissa, sum_exponent = math.frexp(slope_sum)
        sigma_mantissa, sigma_exponent = math.frexp(sigma)
        alpha_mantissa, alpha_exponent = math.frexp(radius_alpha)
        beta_mantissa, beta_exponent = math.frexp(radius_beta)
        mantissa = (
            sum_mantissa
            * sigma_mantissa
            * sigma_mantissa
            / (alpha_mantissa * beta_mantissa * denominator)
        )
        result = math.ldexp(
            mantissa,
            sum_exponent
            + 2 * sigma_exponent
            - alpha_exponent
            - beta_exponent,
        )

    return min(max(result, min(alpha, beta)), max(alpha, beta))


def _field_value(
    fun: ScalarField,
    t: float,
    y: float,
    *,
    step: int,
) -> float:
    """Evaluate and validate the scalar vector field."""

    return _finite_real(
        fun(t, y),
        name=f"fun({t!r}, {y!r}) at step {step}",
    )


def _fixed_point_step(
    fun: ScalarField,
    *,
    step: int,
    t: float,
    t_next: float,
    y: float,
    h: float,
    sigma: float | StepScale | CoupledScale,
    coupled: bool,
    atol: float,
    rtol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Advance one step with an Euler predictor and fixed-point iteration."""

    beta = _field_value(fun, t, y, step=step)
    predictor = y + h * beta
    if not math.isfinite(predictor):
        raise RuntimeError(
            f"Euler predictor produced a non-finite value at step {step}"
        )

    constant_scale: float | None = None
    if not callable(sigma):
        constant_scale = _positive_scale(sigma, step=step)
    elif not coupled:
        constant_scale = _positive_scale(
            sigma(step, t, y, h),
            step=step,
        )

    current = predictor
    for _ in range(max_iter):
        if constant_scale is None:
            scale = _positive_scale(
                sigma(step, t, y, t_next, current, h),
                step=step,
            )
        else:
            scale = constant_scale

        alpha = _field_value(fun, t_next, current, step=step)
        updated = y + h * _scaled_mean(alpha, beta, scale)
        if not math.isfinite(updated):
            raise RuntimeError(
                "fixed-point iteration produced a non-finite value "
                f"at step {step}"
            )

        if math.isclose(updated, current, rel_tol=rtol, abs_tol=atol):
            return updated, scale
        current = updated

    raise RuntimeError(
        "fixed-point iteration failed to converge "
        f"at step {step} after {max_iter} iterations"
    )


def _inputs(
    fun: object,
    t_span: Sequence[RealScalar],
    y0: RealScalar,
    *,
    n_steps: object,
    sigma: object,
    atol: object,
    rtol: object,
    max_iter: object,
) -> tuple[ScalarField, float, float, float, int, float, float, int]:
    """Validate shared solver inputs."""

    if not callable(fun):
        raise TypeError("fun must be callable")

    try:
        if len(t_span) != 2:
            raise ValueError
        t0_value, final_value = t_span
    except (TypeError, ValueError) as exc:
        raise ValueError("t_span must contain exactly two values") from exc

    t0 = _finite_real(t0_value, name="t_span[0]")
    final_time = _finite_real(final_value, name="t_span[1]")
    if final_time <= t0:
        raise ValueError("t_span[1] must be greater than t_span[0]")

    initial_value = _finite_real(y0, name="y0")
    step_count = _positive_integer(n_steps, name="n_steps")
    iteration_limit = _positive_integer(max_iter, name="max_iter")

    absolute_tolerance = _finite_real(atol, name="atol")
    relative_tolerance = _finite_real(rtol, name="rtol")
    if absolute_tolerance < 0.0:
        raise ValueError("atol must be nonnegative")
    if relative_tolerance < 0.0:
        raise ValueError("rtol must be nonnegative")
    if absolute_tolerance == 0.0 and relative_tolerance == 0.0:
        raise ValueError("atol and rtol cannot both be zero")

    if not callable(sigma):
        _positive_scale(sigma, step=0)

    return (
        fun,
        t0,
        final_time,
        initial_value,
        step_count,
        absolute_tolerance,
        relative_tolerance,
        iteration_limit,
    )


def _solve(
    fun: ScalarField,
    t_span: Sequence[RealScalar],
    y0: RealScalar,
    *,
    n_steps: int,
    sigma: RealScalar | StepScale | CoupledScale,
    atol: RealScalar,
    rtol: RealScalar,
    max_iter: int,
    coupled: bool,
) -> ODEResult:
    """Solve with one of the three schemes after shared validation."""

    (
        field,
        t0,
        final_time,
        initial_value,
        step_count,
        absolute_tolerance,
        relative_tolerance,
        iteration_limit,
    ) = _inputs(
        fun,
        t_span,
        y0,
        n_steps=n_steps,
        sigma=sigma,
        atol=atol,
        rtol=rtol,
        max_iter=max_iter,
    )

    span = final_time - t0
    h = span / step_count
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError("the requested uniform step size is not finite")

    t_values = np.linspace(t0, final_time, step_count + 1, dtype=np.float64)
    if np.any(t_values[1:] <= t_values[:-1]):
        raise ValueError(
            "the requested uniform time grid is not representable in float64"
        )

    y_values = np.empty(step_count + 1, dtype=np.float64)
    scale_values = np.empty(step_count, dtype=np.float64)
    y_values[0] = initial_value

    for step in range(step_count):
        next_value, scale = _fixed_point_step(
            field,
            step=step,
            t=float(t_values[step]),
            t_next=float(t_values[step + 1]),
            y=float(y_values[step]),
            h=h,
            sigma=sigma,
            coupled=coupled,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
            max_iter=iteration_limit,
        )
        y_values[step + 1] = next_value
        scale_values[step] = scale

    return ODEResult(t=t_values, y=y_values, sigma=scale_values)


def ellipse_scheme(
    fun: ScalarField,
    t_span: Sequence[RealScalar],
    y0: RealScalar,
    *,
    n_steps: int,
    sigma: RealScalar | StepScale,
    atol: RealScalar = 1e-12,
    rtol: RealScalar = 1e-10,
    max_iter: int = 100,
) -> ODEResult:
    r"""Solve a scalar ODE with the specular ellipse scheme.

    A callable ``sigma`` is evaluated as ``sigma(n, t_n, y_n, h)`` once at
    the start of each step and is frozen during that step's implicit solve.
    """

    return _solve(
        fun,
        t_span,
        y0,
        n_steps=n_steps,
        sigma=sigma,
        atol=atol,
        rtol=rtol,
        max_iter=max_iter,
        coupled=False,
    )


def ellipse_scheme_3rd_order(
    fun: ScalarField,
    t_span: Sequence[RealScalar],
    y0: RealScalar,
    *,
    n_steps: int,
    sigma: RealScalar | StepScale,
    atol: RealScalar = 1e-12,
    rtol: RealScalar = 1e-10,
    max_iter: int = 100,
) -> ODEResult:
    r"""Solve with the third-order left-state ellipse scheme.

    A callable ``sigma`` is evaluated as ``sigma(n, t_n, y_n, h)`` and is
    frozen during the implicit solve.  Third-order convergence requires the
    supplied scale rule to satisfy the defect-cancellation assumptions of the
    method; the function does not infer or verify those assumptions.
    """

    return _solve(
        fun,
        t_span,
        y0,
        n_steps=n_steps,
        sigma=sigma,
        atol=atol,
        rtol=rtol,
        max_iter=max_iter,
        coupled=False,
    )


def ellipse_scheme_4th_order(
    fun: ScalarField,
    t_span: Sequence[RealScalar],
    y0: RealScalar,
    *,
    n_steps: int,
    sigma: RealScalar | CoupledScale,
    atol: RealScalar = 1e-12,
    rtol: RealScalar = 1e-10,
    max_iter: int = 100,
) -> ODEResult:
    r"""Solve with the fourth-order coupled ellipse scheme.

    A callable ``sigma`` is reevaluated during the implicit iteration as
    ``sigma(n, t_n, y_n, t_next, y_trial, h)``.  Fourth-order convergence
    requires that rule to select a positive two-endpoint cancelling branch.
    A scalar scale is accepted, but does not by itself provide that guarantee.
    """

    return _solve(
        fun,
        t_span,
        y0,
        n_steps=n_steps,
        sigma=sigma,
        atol=atol,
        rtol=rtol,
        max_iter=max_iter,
        coupled=True,
    )


__all__ = [
    "ODEResult",
    "ellipse_scheme",
    "ellipse_scheme_3rd_order",
    "ellipse_scheme_4th_order",
]
