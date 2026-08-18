"""The lightweight scalar ODE specular ellipse scheme."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ..backends.numpy._types import VectorToVectorFunc
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
from ._numerics import (
    _Dyadic,
    _dyadic,
    _dyadic_negate,
    _dyadic_product,
    _dyadic_ratio_float,
    _dyadic_sum,
    _magnitude,
    _magnitude_add,
    _magnitude_divide,
    _magnitude_float,
    _magnitude_multiply,
    _magnitude_sqrt,
    _relative_dyadic_sum,
)
from ._result import ODEResult


type StepScale = Callable[[int, float, float, float], RealScalar]


class _ScaleSelectionError(RuntimeError):
    """Internal marker for an unavailable automatic scale branch."""


def _positive_scale(value: object, *, step: int) -> float:
    """Validate a scale selected for one time step."""

    scale = _finite_real(value, name=f"sigma_n at step {step}")
    if scale <= 0.0:
        raise ValueError(f"sigma_n at step {step} must be positive")
    return scale


def _finite_point(t: float, u: float, *, step: int) -> None:
    """Reject a non-finite intermediate point before evaluating ``F``."""

    if not math.isfinite(t) or not math.isfinite(u):
        raise RuntimeError(
            f"a derivative sample is non-finite at step {step}"
        )


def _flow_endpoint(
    F: ScalarField,
    t: float,
    u: float,
    delta: float,
    F_0: float,
    *,
    step: int,
) -> float:
    """Approximate the local ODE flow with one classical RK4 step."""

    half_delta = 0.5 * delta
    t_half = t + half_delta
    t_end = t + delta

    u_2 = u + half_delta * F_0
    _finite_point(t_half, u_2, step=step)
    k_2 = _field_value(F, t_half, u_2, step=step)

    u_3 = u + half_delta * k_2
    _finite_point(t_half, u_3, step=step)
    k_3 = _field_value(F, t_half, u_3, step=step)

    u_4 = u + delta * k_3
    _finite_point(t_end, u_4, step=step)
    k_4 = _field_value(F, t_end, u_4, step=step)

    weighted_slope = math.fsum(
        (F_0 / 6.0, k_2 / 3.0, k_3 / 3.0, k_4 / 6.0)
    )
    endpoint = u + delta * weighted_slope
    _finite_point(t_end, endpoint, step=step)
    return endpoint


def _default_derivative_step(
    t: float,
    u: float,
    F_0: float,
) -> float:
    """Choose a float64 step for two flow-centered derivatives of ``F``."""

    eps = np.finfo(np.float64).eps
    # Richardson extrapolation leaves fourth-order truncation while the
    # centered second derivative amplifies roundoff by delta**-2.  Their
    # float64 balance is eps**(1/6).  The target is translation invariant;
    # ulp-based floors below enlarge it only when the represented point would
    # otherwise fail to move.
    delta = eps ** (1.0 / 6.0)

    # Both delta and delta / 2 must move the represented point.  The state
    # term matters for autonomous fields at a large-magnitude value.
    floor = 4.0 * math.ulp(t)
    if F_0 != 0.0:
        state_floor = 4.0 * math.ulp(u) / abs(F_0)
        if math.isfinite(state_floor):
            floor = max(floor, state_floor)
    return max(delta, floor, np.finfo(np.float64).tiny)


def _numeric_derivatives_of_F(
    F: ScalarField,
    t: float,
    u: float,
    F_0: float,
    derivative_step: float | None,
    *,
    step: int,
) -> tuple[float, float]:
    """Estimate ``L_F F`` and ``L_F^2 F`` along a local RK4 flow."""

    delta = (
        _default_derivative_step(t, u, F_0)
        if derivative_step is None
        else derivative_step
    )
    half_delta = 0.5 * delta
    if (
        not math.isfinite(delta)
        or delta <= 0.0
        or t + half_delta == t
        or t - half_delta == t
    ):
        raise ValueError(
            "derivative_step does not produce distinct float64 samples"
        )

    def centered(radius: float) -> tuple[float, float]:
        u_plus = _flow_endpoint(
            F, t, u, radius, F_0, step=step
        )
        u_minus = _flow_endpoint(
            F, t, u, -radius, F_0, step=step
        )
        F_plus = _field_value(F, t + radius, u_plus, step=step)
        F_minus = _field_value(F, t - radius, u_minus, step=step)
        Q_numerator = _dyadic_sum(
            _dyadic(F_plus),
            _dyadic_negate(_dyadic(F_minus)),
        )
        R_numerator = _dyadic_sum(
            _dyadic(F_plus),
            _dyadic_negate(_dyadic(F_0)),
            _dyadic(F_minus),
            _dyadic_negate(_dyadic(F_0)),
        )
        Q = _dyadic_ratio_float(
            Q_numerator,
            _dyadic_product(_dyadic(2.0), _dyadic(radius)),
        )
        R = _dyadic_ratio_float(
            R_numerator,
            _dyadic_product(_dyadic(radius), _dyadic(radius)),
        )
        return Q, R

    Q_full, R_full = centered(delta)
    Q_half, R_half = centered(half_delta)
    if not all(
        math.isfinite(value)
        for value in (Q_full, R_full, Q_half, R_half)
    ):
        raise RuntimeError(
            f"numerical derivatives of F are non-finite at step {step}"
        )
    Q = _dyadic_ratio_float(
        _dyadic_sum(
            _dyadic_product(_dyadic(4.0), _dyadic(Q_half)),
            _dyadic_negate(_dyadic(Q_full)),
        ),
        _dyadic(3.0),
    )
    R = _dyadic_ratio_float(
        _dyadic_sum(
            _dyadic_product(_dyadic(4.0), _dyadic(R_half)),
            _dyadic_negate(_dyadic(R_full)),
        ),
        _dyadic(3.0),
    )
    if not math.isfinite(Q) or not math.isfinite(R):
        raise RuntimeError(
            f"numerical derivatives of F are non-finite at step {step}"
        )
    return Q, R


def _field_quantities(
    F: ScalarField,
    t: float,
    u: float,
    derivatives_of_F: VectorToVectorFunc | None,
    derivative_step: float | None,
    *,
    step: int,
) -> tuple[float, float, float]:
    """Return ``F``, ``L_F F``, and ``L_F^2 F`` at one point."""

    F_0 = _field_value(F, t, u, step=step)
    if derivatives_of_F is None:
        Q, R = _numeric_derivatives_of_F(
            F,
            t,
            u,
            F_0,
            derivative_step,
            step=step,
        )
        return F_0, Q, R

    point = np.array([t, u], dtype=np.float64)
    values = np.asarray(derivatives_of_F(point.copy()))
    if values.shape != (2,) or values.dtype.kind not in "iuf":
        raise TypeError(
            "derivatives_of_F must return a real vector with shape (2,)"
        )
    derivatives = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(derivatives)):
        raise ValueError("derivatives_of_F must return finite values")
    return F_0, float(derivatives[0]), float(derivatives[1])


def _third_order_scale(
    quantities: tuple[float, float, float],
    previous_sigma: float | None,
    *,
    step: int,
) -> float:
    """Select the unique positive left-endpoint cancelling scale."""

    F_0, Q, R = quantities
    if F_0 == 0.0 or Q == 0.0:
        if R == 0.0:
            return 1.0 if previous_sigma is None else previous_sigma
        raise _ScaleSelectionError(
            "no unique positive defect-cancelling scale exists "
            f"at step {step}"
        )
    if R == 0.0 or ((F_0 < 0.0) != (R < 0.0)):
        raise _ScaleSelectionError(
            "no unique positive defect-cancelling scale exists "
            f"at step {step}"
        )

    three_Q_squared = _dyadic_product(
        _dyadic(3.0),
        _dyadic(Q),
        _dyadic(Q),
    )
    F_times_R = _dyadic_product(_dyadic(F_0), _dyadic(R))
    margin = _dyadic_sum(
        three_Q_squared,
        _dyadic_negate(F_times_R),
    )
    if margin[0] <= 0:
        raise _ScaleSelectionError(
            "no unique positive defect-cancelling scale exists "
            f"at step {step}"
        )

    ratio = _magnitude_divide(
        _magnitude(margin),
        _magnitude(F_times_R),
    )
    sigma_magnitude = _magnitude_multiply(
        _magnitude(_dyadic(F_0)),
        _magnitude_sqrt(ratio),
    )
    sigma = _magnitude_float(sigma_magnitude)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise _ScaleSelectionError(
            f"third-order scale is outside float64 range at step {step}"
        )

    sigma_squared = _dyadic_product(_dyadic(sigma), _dyadic(sigma))
    relative_residual = _relative_dyadic_sum(
        _dyadic_product(_dyadic(R), sigma_squared),
        _dyadic_product(_dyadic(R), _dyadic(F_0), _dyadic(F_0)),
        _dyadic_negate(
            _dyadic_product(
                _dyadic(3.0),
                _dyadic(F_0),
                _dyadic(Q),
                _dyadic(Q),
            )
        ),
    )
    if relative_residual > 256.0 * np.finfo(np.float64).eps:
        raise _ScaleSelectionError(
            f"third-order scale residual is not resolved at step {step}"
        )
    return sigma


def _fourth_order_coefficients(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[_Dyadic, _Dyadic, _Dyadic]:
    """Build the two-endpoint scale polynomial exactly."""

    F_left, Q_left, R_left = left
    F_right, Q_right, R_right = right
    a = _dyadic(R_left)
    b = _dyadic_product(
        _dyadic(3.0),
        _dyadic(F_left),
        _dyadic(Q_left),
        _dyadic(Q_left),
    )
    c = _dyadic_product(_dyadic(F_left), _dyadic(F_left))
    d = _dyadic(R_right)
    e = _dyadic_product(
        _dyadic(3.0),
        _dyadic(F_right),
        _dyadic(Q_right),
        _dyadic(Q_right),
    )
    f = _dyadic_product(_dyadic(F_right), _dyadic(F_right))

    A = _dyadic_sum(a, d)
    B = _dyadic_sum(
        _dyadic_product(A, _dyadic_sum(c, f)),
        _dyadic_negate(b),
        _dyadic_negate(e),
    )
    C = _dyadic_sum(
        _dyadic_product(A, c, f),
        _dyadic_negate(_dyadic_product(b, f)),
        _dyadic_negate(_dyadic_product(e, c)),
    )
    return A, B, C


def _fourth_order_scale(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    previous_sigma: float | None,
    *,
    step: int,
) -> float:
    """Select the unique positive E5(i) two-endpoint cancelling scale."""

    A, B, C = _fourth_order_coefficients(left, right)
    if A[0] == B[0] == C[0] == 0:
        return 1.0 if previous_sigma is None else previous_sigma

    opposite_signs = (
        A[0] != 0
        and C[0] != 0
        and ((A[0] < 0) != (C[0] < 0))
    )
    if not opposite_signs:
        two_positive_roots = (
            A[0] != 0
            and B[0] != 0
            and C[0] != 0
            and ((A[0] < 0) == (C[0] < 0))
            and ((A[0] < 0) != (B[0] < 0))
        )
        if two_positive_roots:
            discriminant = _dyadic_sum(
                _dyadic_product(B, B),
                _dyadic_negate(
                    _dyadic_product(_dyadic(4.0), A, C)
                ),
            )
            if discriminant[0] > 0:
                raise _ScaleSelectionError(
                    "ambiguous: two positive defect-balancing scales exist "
                    f"at step {step}"
                )
        raise _ScaleSelectionError(
            "no unique positive defect-balancing scale exists "
            f"at step {step}"
        )

    A_magnitude = _magnitude(A)
    q_magnitude = _magnitude_divide(
        _magnitude(C),
        A_magnitude,
    )
    if B[0] == 0:
        z_magnitude = _magnitude_sqrt(q_magnitude)
    else:
        p_magnitude = _magnitude_divide(
            _magnitude(B),
            A_magnitude,
        )
        discriminant_root = _magnitude_sqrt(
            _magnitude_add(
                _magnitude_multiply(p_magnitude, p_magnitude),
                (q_magnitude[0], q_magnitude[1] + 2),
            )
        )
        denominator = _magnitude_add(p_magnitude, discriminant_root)
        p_is_positive = (A[0] < 0) == (B[0] < 0)
        if p_is_positive:
            z_magnitude = _magnitude_divide(
                (q_magnitude[0], q_magnitude[1] + 1),
                denominator,
            )
        else:
            z_magnitude = denominator[0], denominator[1] - 1

    sigma = _magnitude_float(_magnitude_sqrt(z_magnitude))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise _ScaleSelectionError(
            f"fourth-order scale is outside float64 range at step {step}"
        )

    z = _dyadic_product(_dyadic(sigma), _dyadic(sigma))
    relative_residual = _relative_dyadic_sum(
        _dyadic_product(A, z, z),
        _dyadic_product(B, z),
        C,
    )
    if relative_residual > 1024.0 * np.finfo(np.float64).eps:
        raise _ScaleSelectionError(
            f"fourth-order scale residual is not resolved at step {step}"
        )
    return sigma


def _fixed_scale_step(
    F: ScalarField,
    *,
    step: int,
    t_next: float,
    u_n: float,
    h_n: float,
    sigma: float,
    F_left: float,
    atol: float,
    rtol: float,
    max_iter: int,
) -> float:
    """Solve one implicit SE update with a fixed scale."""

    v = u_n + h_n * F_left
    if not math.isfinite(v):
        raise RuntimeError(
            f"Euler predictor produced a non-finite value at step {step}"
        )
    for _ in range(max_iter):
        F_right = _field_value(F, t_next, v, step=step)
        mean = float(scaled_mean(F_right, F_left, sigma))
        updated = u_n + h_n * mean
        if not math.isfinite(updated):
            raise RuntimeError(
                "fixed-point iteration produced a non-finite value "
                f"at step {step}"
            )
        if math.isclose(updated, v, rel_tol=rtol, abs_tol=atol):
            return updated
        v = updated
    raise RuntimeError(
        "fixed-point iteration failed to converge "
        f"at step {step} after {max_iter} iterations"
    )


def _fourth_order_step(
    F: ScalarField,
    *,
    step: int,
    t_next: float,
    u_n: float,
    h_n: float,
    left: tuple[float, float, float],
    derivatives_of_F: VectorToVectorFunc | None,
    derivative_step: float | None,
    previous_sigma: float | None,
    atol: float,
    rtol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Solve the coupled endpoint and E5(i) scale equations."""

    F_left = left[0]
    predictor = u_n + h_n * F_left
    if not math.isfinite(predictor):
        raise RuntimeError(
            f"Euler predictor produced a non-finite value at step {step}"
        )

    def tolerance(v: float) -> float:
        return atol + rtol * max(abs(u_n), abs(v))

    def residual(v: float) -> tuple[float, float]:
        right = _field_quantities(
            F,
            t_next,
            v,
            derivatives_of_F,
            derivative_step,
            step=step,
        )
        sigma = _fourth_order_scale(
            left, right, previous_sigma, step=step
        )
        mean = float(scaled_mean(right[0], F_left, sigma))
        updated = u_n + h_n * mean
        if not math.isfinite(updated):
            raise RuntimeError(
                "coupled iteration produced a non-finite value "
                f"at step {step}"
            )
        value = v - updated
        if not math.isfinite(value):
            raise RuntimeError(
                f"coupled residual is non-finite at step {step}"
            )
        return value, sigma

    samples: dict[float, tuple[float, float]] = {}
    last_scale_error: _ScaleSelectionError | None = None

    def evaluate(v: float) -> tuple[float, float] | None:
        nonlocal last_scale_error
        if not math.isfinite(v):
            return None
        if v in samples:
            return samples[v]
        try:
            value = residual(v)
        except _ScaleSelectionError as exc:
            last_scale_error = exc
            return None
        samples[v] = value
        return value

    displacement = predictor - u_n
    seed_candidates = (
        predictor,
        u_n + 0.5 * displacement,
        u_n + 0.25 * displacement,
        u_n + 0.75 * displacement,
        u_n,
    )
    v: float | None = None
    for candidate in seed_candidates:
        if evaluate(candidate) is not None:
            v = candidate
            break

    # Fixed-point iteration remains the inexpensive local path.  Trying the
    # interior segment first avoids rejecting a valid coupled root merely
    # because the Euler endpoint lies on a degenerate/non-E5 boundary.
    if v is not None:
        for _ in range(max_iter):
            evaluated = evaluate(v)
            if evaluated is None:
                break
            value, sigma = evaluated
            if abs(value) <= tolerance(v):
                return v, sigma
            updated = v - value
            if evaluate(updated) is None:
                break
            v = updated

    if not samples and last_scale_error is not None:
        raise last_scale_error

    def find_bracket(
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None:
        ordered = sorted(
            (point, value, sigma)
            for point, (value, sigma) in samples.items()
        )
        for point, value, sigma in ordered:
            if abs(value) <= tolerance(point):
                return (point, value, sigma), (point, value, sigma)
        for left_sample, right_sample in zip(
            ordered, ordered[1:], strict=False
        ):
            if math.copysign(1.0, left_sample[1]) != math.copysign(
                1.0, right_sample[1]
            ):
                return left_sample, right_sample
        return None

    bracket = find_bracket()
    center = 0.5 * (u_n + predictor)
    width = max(
        abs(displacement),
        abs(h_n),
        math.sqrt(np.finfo(np.float64).eps) * max(1.0, abs(u_n)),
    )
    for _ in range(min(6, max_iter)):
        if bracket is not None:
            break
        for fraction in np.linspace(-1.0, 1.0, 9):
            evaluate(center + width * float(fraction))
        bracket = find_bracket()
        width *= 2.0

    if bracket is None:
        if not samples and last_scale_error is not None:
            raise last_scale_error
        raise RuntimeError(
            "coupled iteration failed to bracket a local root "
            f"at step {step}"
        )

    lower, upper = bracket
    if lower[0] == upper[0]:
        return lower[0], lower[2]

    for _ in range(max_iter):
        midpoint = lower[0] + 0.5 * (upper[0] - lower[0])
        evaluated = evaluate(midpoint)
        if evaluated is None:
            # Under the uniform E5(i) hypothesis the branch remains valid
            # throughout a sufficiently small bracket.  If numerical branch
            # classification breaks inside it, do not invent a continuation.
            if last_scale_error is not None:
                raise last_scale_error
            break
        value, sigma = evaluated
        if abs(value) <= tolerance(midpoint):
            return midpoint, sigma
        if abs(upper[0] - lower[0]) <= tolerance(midpoint):
            raise RuntimeError(
                "coupled root residual is not resolved "
                f"at step {step}"
            )
        sample = (midpoint, value, sigma)
        if math.copysign(1.0, value) == math.copysign(1.0, lower[1]):
            lower = sample
        else:
            upper = sample

    raise RuntimeError(
        "coupled root iteration failed to converge "
        f"at step {step} after {max_iter} iterations"
    )


def _validated_inputs(
    F: object,
    t_0: object,
    T: object,
    u_0: object,
    *,
    n_steps: object,
    sigma_n: object,
    third_order: object,
    fourth_order: object,
    derivatives_of_F: object,
    derivative_step: object,
    atol: object,
    rtol: object,
    max_iter: object,
) -> tuple[
    float,
    float,
    float,
    int,
    float | StepScale | None,
    bool,
    bool,
    VectorToVectorFunc | None,
    float | None,
    float,
    float,
    int,
]:
    """Validate the public API before the first evaluation of ``F``."""

    if not isinstance(third_order, (bool, np.bool_)):
        raise TypeError("third_order must be a boolean")
    if not isinstance(fourth_order, (bool, np.bool_)):
        raise TypeError("fourth_order must be a boolean")
    use_third = bool(third_order)
    use_fourth = bool(fourth_order)
    if use_third and use_fourth:
        raise ValueError("third_order and fourth_order cannot both be True")

    automatic = use_third or use_fourth
    if automatic and sigma_n is not None:
        raise ValueError("sigma_n must be None in an automatic order mode")
    if not automatic and sigma_n is None:
        raise ValueError("sigma_n is required in the base mode")
    if not automatic and derivatives_of_F is not None:
        raise ValueError(
            "derivatives_of_F requires third_order=True or fourth_order=True"
        )
    if not automatic and derivative_step is not None:
        raise ValueError(
            "derivative_step requires third_order=True or fourth_order=True"
        )
    if derivatives_of_F is not None and not callable(derivatives_of_F):
        raise TypeError("derivatives_of_F must be callable")
    if derivatives_of_F is not None and derivative_step is not None:
        raise ValueError(
            "derivative_step cannot be used together with derivatives_of_F"
        )

    if not callable(F):
        raise TypeError("F must be callable")
    initial_time = _finite_real(t_0, name="t_0")
    final_time = _finite_real(T, name="T")
    if final_time <= initial_time:
        raise ValueError("T must be greater than t_0")
    initial_value = _finite_real(u_0, name="u_0")
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

    selected_derivative_step: float | None = None
    if derivative_step is not None:
        selected_derivative_step = _finite_real(
            derivative_step, name="derivative_step"
        )
        if selected_derivative_step <= 0.0:
            raise ValueError("derivative_step must be positive")

    selected_sigma: float | StepScale | None
    if sigma_n is None or callable(sigma_n):
        selected_sigma = sigma_n
    else:
        selected_sigma = _positive_scale(sigma_n, step=0)

    return (
        initial_time,
        final_time,
        initial_value,
        step_count,
        selected_sigma,
        use_third,
        use_fourth,
        derivatives_of_F,
        selected_derivative_step,
        absolute_tolerance,
        relative_tolerance,
        iteration_limit,
    )


def ellipse_scheme(
    F: ScalarField,
    t_0: RealScalar,
    T: RealScalar,
    u_0: RealScalar,
    *,
    n_steps: int,
    sigma_n: RealScalar | StepScale | None = None,
    third_order: bool = False,
    fourth_order: bool = False,
    derivatives_of_F: VectorToVectorFunc | None = None,
    derivative_step: RealScalar | None = None,
    atol: RealScalar = 1e-12,
    rtol: RealScalar = 1e-10,
    max_iter: int = 100,
) -> ODEResult:
    r"""Apply the scalar specular ellipse scheme on ``[t_0, T]``.

    In the base mode, ``sigma_n`` is a positive scalar or a callable
    ``sigma_n(n, t_n, u_n, h_n)`` and is frozen during its implicit step.
    ``third_order=True`` numerically enforces left-endpoint defect
    cancellation. ``fourth_order=True`` couples the endpoint update to the
    unique positive E5(i) two-endpoint cancelling scale.

    When ``derivatives_of_F`` is omitted, ``L_F F`` and ``L_F^2 F`` are
    estimated from ``F`` along a local RK4 flow. Otherwise the callback must
    map ``[t, u]`` to ``[L_F F, L_F^2 F]``.
    """

    (
        initial_time,
        final_time,
        initial_value,
        step_count,
        selected_sigma,
        use_third,
        use_fourth,
        selected_derivatives,
        selected_derivative_step,
        absolute_tolerance,
        relative_tolerance,
        iteration_limit,
    ) = _validated_inputs(
        F,
        t_0,
        T,
        u_0,
        n_steps=n_steps,
        sigma_n=sigma_n,
        third_order=third_order,
        fourth_order=fourth_order,
        derivatives_of_F=derivatives_of_F,
        derivative_step=derivative_step,
        atol=atol,
        rtol=rtol,
        max_iter=max_iter,
    )
    counted_field = _FieldEvaluationCounter(F)

    t_values, step_sizes = _time_grid(
        initial_time,
        final_time,
        step_count,
    )
    if (
        selected_derivative_step is not None
        and selected_derivatives is None
    ):
        derivative_times = t_values if use_fourth else t_values[:-1]
        for sample_time_value in derivative_times:
            sample_time = float(sample_time_value)
            for radius in (
                0.5 * selected_derivative_step,
                selected_derivative_step,
            ):
                forward_time = sample_time + radius
                backward_time = sample_time - radius
                if (
                    not math.isfinite(forward_time)
                    or not math.isfinite(backward_time)
                    or forward_time == sample_time
                    or backward_time == sample_time
                ):
                    raise ValueError(
                        "derivative_step does not produce distinct float64 "
                        "samples on [t_0, T]"
                    )

    u_values = np.empty(step_count + 1, dtype=np.float64)
    sigma_values = np.empty(step_count, dtype=np.float64)
    u_values[0] = initial_value
    previous_sigma: float | None = None

    for step in range(step_count):
        t_n = float(t_values[step])
        t_next = float(t_values[step + 1])
        u_n = float(u_values[step])
        h_n = float(step_sizes[step])

        if not use_third and not use_fourth:
            if callable(selected_sigma):
                sigma = _positive_scale(
                    selected_sigma(step, t_n, u_n, h_n), step=step
                )
            else:
                assert selected_sigma is not None
                sigma = selected_sigma
            F_left = _field_value(counted_field, t_n, u_n, step=step)
            u_next = _fixed_scale_step(
                counted_field,
                step=step,
                t_next=t_next,
                u_n=u_n,
                h_n=h_n,
                sigma=sigma,
                F_left=F_left,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                max_iter=iteration_limit,
            )
        else:
            left = _field_quantities(
                counted_field,
                t_n,
                u_n,
                selected_derivatives,
                selected_derivative_step,
                step=step,
            )
            if use_third:
                sigma = _third_order_scale(
                    left, previous_sigma, step=step
                )
                u_next = _fixed_scale_step(
                    counted_field,
                    step=step,
                    t_next=t_next,
                    u_n=u_n,
                    h_n=h_n,
                    sigma=sigma,
                    F_left=left[0],
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    max_iter=iteration_limit,
                )
            else:
                u_next, sigma = _fourth_order_step(
                    counted_field,
                    step=step,
                    t_next=t_next,
                    u_n=u_n,
                    h_n=h_n,
                    left=left,
                    derivatives_of_F=selected_derivatives,
                    derivative_step=selected_derivative_step,
                    previous_sigma=previous_sigma,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                    max_iter=iteration_limit,
                )

        u_values[step + 1] = u_next
        sigma_values[step] = sigma
        previous_sigma = sigma

    return ODEResult(
        t=t_values,
        u=u_values,
        sigma=sigma_values,
        number_of_field_evaluations=counted_field.number_of_field_evaluations,
    )


__all__ = ["ellipse_scheme"]
