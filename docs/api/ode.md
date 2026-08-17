# Scalar ODE schemes

The ODE API solves scalar initial-value problems in the notation of the
specular ellipse scheme:

\[
u'(t)=F(t,u(t)), \qquad u(t_0)=u_0, \qquad t\in[t_0,T].
\]

The public entry points are `specular.euler_scheme_1`,
`specular.euler_scheme_2`, `specular.euler_scheme_5`, and
`specular.ellipse_scheme`. Importing `specular` does not eagerly load the ODE
implementation. Vector-valued states are not currently supported.

The same functions are also available from the public
`specular.ode.solver` module.

The angular means used by these schemes are evaluated by the currently
selected calculation backend. The scalar result is converted back to a
Python float for the ODE iteration; NumPy therefore remains the default and
the reproducible float64 path.

The result contains numerical data only. Plotting, table generation, event
handling, and dense output are intentionally outside this API.

## Unscaled specular Euler schemes

The functions `euler_scheme_1`, `euler_scheme_2`, and `euler_scheme_5`
implement the unscaled specular Euler schemes SE1, SE2, and SE5. Throughout
this section,

\[
h_n:=t_{n+1}-t_n,
\qquad
\mathcal C:=\mathcal C_1,
\]

where `t` is the represented mesh returned by the function and
\(\mathcal C_1\) is the unscaled angular mean.

### SE1

SE1 is the explicit two-step recurrence

\[
u_{n+1}=u_n+h_n\mathcal C\!\left(
F(t_n,u_n),F(t_{n-1},u_{n-1})
\right),
\qquad n=1,\ldots,N-1.
\]

The value `u_1` at `t[1]` is supplied externally. The function neither
constructs nor modifies that starting value:

```python
import specular


def F(t, u):
    return -u


t_0 = 0.0
T = 1.0
n_steps = 100
h = (T - t_0) / n_steps
u_0 = 1.0
u_1 = u_0 + h * F(t_0, u_0)  # starter chosen by the caller

se1 = specular.euler_scheme_1(
    F,
    t_0,
    T,
    u_0,
    u_1,
    n_steps=n_steps,
)
```

### SE2

SE2 is the explicit two-step recurrence

\[
u_{n+1}=u_n+h_n\mathcal C\!\left(
F(t_n,u_n),\frac{u_n-u_{n-1}}{h_{n-1}}
\right),
\qquad n=1,\ldots,N-1.
\]

It has the same external `u_1` contract as SE1:

```python
se2 = specular.euler_scheme_2(
    F,
    t_0,
    T,
    u_0,
    u_1,
    n_steps=n_steps,
)
```

SE1 and SE2 are generically only first-order consistent. No higher-order
theorem is implied by these functions or by the caller's choice of `u_1`.

### SE5

SE5 is the implicit one-step recurrence

\[
u_{n+1}=u_n+h_n\mathcal C\!\left(
F(t_{n+1},u_{n+1}),F(t_n,u_n)
\right),
\qquad n=0,\ldots,N-1.
\]

It is exactly the ellipse scheme with the fixed scale \(\sigma_n=1\):

```python
se5 = specular.euler_scheme_5(
    F,
    t_0,
    T,
    u_0,
    n_steps=n_steps,
)

same_method = specular.ellipse_scheme(
    F,
    t_0,
    T,
    u_0,
    n_steps=n_steps,
    sigma_n=1.0,
)
```

`euler_scheme_5` accepts the same `atol`, `rtol`, and `max_iter` controls used
by the base ellipse scheme's implicit solve.

## Prescribed scale

In the base scheme, pass `sigma_n` as either a positive real scalar or a
callable. A scalar uses the same scale at every step.

```python
import specular


def F(t, u):
    return -u


result = specular.ellipse_scheme(
    F,
    0.0,
    1.0,
    1.0,
    n_steps=100,
    sigma_n=1.0,
)

print(result.t[-1], result.u[-1])
```

A callable scale has the contract

```python
sigma_n(n, t_n, u_n, h_n) -> positive float
```

It is evaluated once at the accepted left endpoint and frozen during that
step's implicit solve. The index `n` permits prescribed sequences, and `h_n`
is the represented interval `t[n + 1] - t[n]` actually used by the solver.
Consequently, the scale may depend explicitly on the mesh:

```python
def sigma_n(n, t_n, u_n, h_n):
    return h_n**0.5
```

## Numerical cancellation modes

Set exactly one of `third_order` or `fourth_order` to request numerical scale
selection. Leave `sigma_n=None` in either mode:

```python
third_order_result = specular.ellipse_scheme(
    F,
    0.0,
    1.0,
    1.0,
    n_steps=100,
    third_order=True,
)

fourth_order_result = specular.ellipse_scheme(
    F,
    0.0,
    1.0,
    1.0,
    n_steps=100,
    fourth_order=True,
)
```

The third-order mode numerically selects a positive `sigma_n` satisfying the
left-endpoint defect-cancellation condition. The fourth-order mode couples
the trial right endpoint and `sigma_n` and numerically enforces the
two-endpoint defect balance on the unique positive E5(i) branch. If no such
scale is found, or if the branch is ambiguous, the solver raises an error
instead of silently falling back to the base scheme.
If the defect vanishes for every positive scale, the previous accepted scale
is continued; the first such step uses `1.0`.

Both modes require derivatives of `F` along solution curves of the ODE:

\[
L_F F,\qquad L_F^2 F,
\qquad L_F=\partial_t+F\partial_u.
\]

By default these are estimated numerically from `F`. `derivative_step` may be
used to set the finite-difference step. Centered local-flow samples can evaluate
`F` just outside `[t_0, T]`, so this mode requires `F` to be defined on a
neighborhood of the time interval. Near a hard domain boundary, provide
`derivatives_of_F` instead. For an exact, symbolic, or automatic-differentiation
implementation, that callback uses the existing `VectorToVectorFunc` shape:

```python
import numpy as np


def derivatives_of_F(point):
    t, u = point
    L_F_F = ...
    L_F_2_F = ...
    return np.array([L_F_F, L_F_2_F])
```

The input and output both have shape `(2,)`. The returned components are
`L_F F` and `L_F^2 F`, in that order.
`derivative_step` cannot be supplied together with `derivatives_of_F`, because
the exact callback bypasses numerical differentiation.

`third_order=True` and `fourth_order=True` are mutually exclusive. Supplying
`sigma_n` together with either flag is also an error: a prescribed scale and
an automatically selected scale are different modes.

!!! warning "Conditional order"

    These flags numerically enforce the corresponding cancellation condition;
    they do not provide an unconditional convergence-order guarantee. Third-
    order convergence still depends on the smoothness and boundedness of the
    selected branch and on sufficiently accurate field derivatives. The
    fourth-order result is also conditional: it requires the unique positive
    E5(i) branch together with the stated uniform smoothness and boundedness
    hypotheses. It is not an unconditional guarantee for arbitrary `F`.
    Finite-difference error can also produce an accuracy plateau as the mesh
    is refined. Rapid variation, or variation that is small relative to a
    large additive offset in `F`, may require an explicit `derivative_step` or
    a `derivatives_of_F` callback.

## Result

`ODEResult.t` and `ODEResult.u` contain the initial value and every accepted
step. `ODEResult.sigma` contains the scale associated with each represented
interval and therefore has one fewer entry. For SE1, SE2, and SE5 it is an
array of ones, representing the convention \(\mathcal C=\mathcal C_1\). In
the two-step methods this also includes the externally supplied first
interval; it does not describe how `u_1` was produced.

`ODEResult.number_of_field_evaluations` records the total number of calls to
the supplied field `F(t, u)` made by the solver. Calls made internally by a
user-provided `derivatives_of_F` callback are outside this count.

## API reference

::: specular.ode.ODEResult
    options:
      show_root_heading: true

::: specular.ode.euler_scheme_1
    options:
      show_root_heading: true

::: specular.ode.euler_scheme_2
    options:
      show_root_heading: true

::: specular.ode.euler_scheme_5
    options:
      show_root_heading: true

::: specular.ode.ellipse_scheme
    options:
      show_root_heading: true
