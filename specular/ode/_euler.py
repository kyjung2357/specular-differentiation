"""The unscaled specular Euler schemes of Types 1, 2, and 5."""

from __future__ import annotations

import math

import numpy as np

from ..calculation import scaled_mean
from ._common import (
    RealScalar,
    ScalarField,
    _FieldEvaluationCounter,
    _field_value,
    _finite_real,
    _positive_integer,
    _time_grid,
)
from ._ellipse import ellipse_scheme
from ._numerics import (
    _dyadic,
    _dyadic_negate,
    _dyadic_product,
    _dyadic_ratio_float,
    _dyadic_sum,
)
from ._result import FloatArray, ODEResult


def _two_step_inputs(
    F: ScalarField,
    t_0: RealScalar,
    T: RealScalar,
    u_0: RealScalar,
    u_1: RealScalar,
    n_steps: int,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Validate a two-step public call before evaluating ``F``."""

    if not callable(F):
        raise TypeError("F must be callable")
    initial_time = _finite_real(t_0, name="t_0")
    final_time = _finite_real(T, name="T")
    if final_time <= initial_time:
        raise ValueError("T must be greater than t_0")
    initial_value = _finite_real(u_0, name="u_0")
    first_value = _finite_real(u_1, name="u_1")
    step_count = _positive_integer(n_steps, name="n_steps")

    t_values, step_sizes = _time_grid(
        initial_time,
        final_time,
        step_count,
    )
    return t_values, step_sizes, initial_value, first_value


def _advance(
    u_n: float,
    h_n: float,
    slope: float,
    *,
    step: int,
) -> float:
    """Evaluate ``u_n + h_n * slope`` without intermediate overflow."""

    value = _dyadic_ratio_float(
        _dyadic_sum(
            _dyadic(u_n),
            _dyadic_product(_dyadic(h_n), _dyadic(slope)),
        ),
        _dyadic(1.0),
    )
    if not math.isfinite(value):
        raise RuntimeError(f"the state is non-finite at step {step}")
    return value


def _backward_slope(
    u_n: float,
    u_previous: float,
    h_previous: float,
    *,
    step: int,
) -> float:
    """Evaluate the represented backward difference quotient."""

    value = _dyadic_ratio_float(
        _dyadic_sum(
            _dyadic(u_n),
            _dyadic_negate(_dyadic(u_previous)),
        ),
        _dyadic(h_previous),
    )
    if not math.isfinite(value):
        raise RuntimeError(
            f"the backward difference is non-finite at step {step}"
        )
    return value


def euler_scheme_1(
    F: ScalarField,
    t_0: RealScalar,
    T: RealScalar,
    u_0: RealScalar,
    u_1: RealScalar,
    *,
    n_steps: int,
) -> ODEResult:
    r"""Apply the explicit two-step specular Euler scheme of Type 1.

    ``u_1`` is an externally supplied value at the first represented node.
    For :math:`n\geq1`, the method uses

    .. math::

        u_{n+1}=u_n+h_n\mathcal C\left(
            F(t_n,u_n),F(t_{n-1},u_{n-1})
        \right).
    """

    t_values, step_sizes, initial_value, first_value = _two_step_inputs(
        F,
        t_0,
        T,
        u_0,
        u_1,
        n_steps,
    )
    counted_field = _FieldEvaluationCounter(F)
    step_count = len(step_sizes)
    u_values = np.empty(step_count + 1, dtype=np.float64)
    u_values[0] = initial_value
    u_values[1] = first_value

    if step_count > 1:
        F_previous = _field_value(
            counted_field,
            float(t_values[0]),
            initial_value,
            step=0,
        )
        for step in range(1, step_count):
            u_n = float(u_values[step])
            F_current = _field_value(
                counted_field,
                float(t_values[step]),
                u_n,
                step=step,
            )
            slope = float(scaled_mean(F_current, F_previous))
            u_values[step + 1] = _advance(
                u_n,
                float(step_sizes[step]),
                slope,
                step=step,
            )
            F_previous = F_current

    return ODEResult(
        t=t_values,
        u=u_values,
        sigma=np.ones(step_count, dtype=np.float64),
        number_of_field_evaluations=counted_field.number_of_field_evaluations,
    )


def euler_scheme_2(
    F: ScalarField,
    t_0: RealScalar,
    T: RealScalar,
    u_0: RealScalar,
    u_1: RealScalar,
    *,
    n_steps: int,
) -> ODEResult:
    r"""Apply the explicit two-step specular Euler scheme of Type 2.

    ``u_1`` is an externally supplied value at the first represented node.
    For :math:`n\geq1`, the method uses

    .. math::

        u_{n+1}=u_n+h_n\mathcal C\left(
            F(t_n,u_n),\frac{u_n-u_{n-1}}{h_{n-1}}
        \right).
    """

    t_values, step_sizes, initial_value, first_value = _two_step_inputs(
        F,
        t_0,
        T,
        u_0,
        u_1,
        n_steps,
    )
    counted_field = _FieldEvaluationCounter(F)
    step_count = len(step_sizes)
    u_values = np.empty(step_count + 1, dtype=np.float64)
    u_values[0] = initial_value
    u_values[1] = first_value

    for step in range(1, step_count):
        u_n = float(u_values[step])
        backward = _backward_slope(
            u_n,
            float(u_values[step - 1]),
            float(step_sizes[step - 1]),
            step=step,
        )
        F_current = _field_value(
            counted_field,
            float(t_values[step]),
            u_n,
            step=step,
        )
        slope = float(scaled_mean(F_current, backward))
        u_values[step + 1] = _advance(
            u_n,
            float(step_sizes[step]),
            slope,
            step=step,
        )

    return ODEResult(
        t=t_values,
        u=u_values,
        sigma=np.ones(step_count, dtype=np.float64),
        number_of_field_evaluations=counted_field.number_of_field_evaluations,
    )


def euler_scheme_5(
    F: ScalarField,
    t_0: RealScalar,
    T: RealScalar,
    u_0: RealScalar,
    *,
    n_steps: int,
    atol: RealScalar = 1e-12,
    rtol: RealScalar = 1e-10,
    max_iter: int = 100,
) -> ODEResult:
    r"""Apply the implicit one-step specular Euler scheme of Type 5.

    This is exactly the specular ellipse scheme with the fixed scale
    :math:`\sigma_n=1`.
    """

    return ellipse_scheme(
        F,
        t_0,
        T,
        u_0,
        n_steps=n_steps,
        sigma_n=1.0,
        atol=atol,
        rtol=rtol,
        max_iter=max_iter,
    )


__all__ = [
    "euler_scheme_1",
    "euler_scheme_2",
    "euler_scheme_5",
]
