# ODE

The ODE API solves scalar initial-value problems in the notation of the specular ellipse method:

\[
u'(t)=F(t,u(t)), \qquad u(t_0)=u_0, \qquad t\in[t_0,T].
\]

The public entry points are `specular.euler_scheme_1`, `specular.euler_scheme_2`, `specular.euler_scheme_5`, and `specular.ellipse_scheme`. Importing `specular` does not eagerly load the ODE implementation.
Vector-valued states are not currently supported.

The same functions are also available from the public `specular.ode.solver` module.

The angular means used by these methods are evaluated by the currently selected calculation backend.
The scalar result is converted back to a Python float for the ODE iteration; NumPy therefore remains the default and the reproducible float64 path.

The result contains numerical data only.
Plotting, table generation, event handling, and dense output are intentionally outside this API.

The [scalar ODE examples](../examples/ode/index.md) reproduce the six numerical experiments discussed in the manuscript.

## Unscaled specular Euler methods

The functions `euler_scheme_1`, `euler_scheme_2`, and `euler_scheme_5` implement the unscaled specular Euler methods of Types 1, 2, and 5. Throughout this section, the mesh is uniform and

\[
h:=\frac{T-t_0}{N},
\qquad
t_n:=t_0+nh,
\qquad
\mathcal C:=\mathcal C_1,
\]

where \(N\) is `n_steps` and \(\mathcal C_1\) is the unscaled angular mean.

!!! note "Type numbers and convergence-order labels"

    Type 2 here means the two-step method implemented by `euler_scheme_2`. On the numerical-examples page, SE2, SE3, and SE4 instead denote second-, third-, and fourth-order configurations of `ellipse_scheme`.
    In particular, ellipse SE2 is Type 5, not Type 2.

### Type 1

Type 1 is the explicit two-step recurrence

\[
u_{n+1}=u_n+h\mathcal C\!\left(
F(t_n,u_n),F(t_{n-1},u_{n-1})
\right),
\qquad n=1,\ldots,N-1.
\]

The value `u_1` at `t[1]` is supplied externally.
The function neither constructs nor modifies that starting value:

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

### Type 2

Type 2 is the explicit two-step recurrence

\[
u_{n+1}=u_n+h\mathcal C\!\left(
F(t_n,u_n),\frac{u_n-u_{n-1}}{h}
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

Types 1 and 2 are generically only first-order consistent.
No higher-order theorem is implied by these functions or by the caller's choice of `u_1`.

### Type 5

Type 5 is the implicit one-step recurrence

\[
u_{n+1}=u_n+h\mathcal C\!\left(
F(t_{n+1},u_{n+1}),F(t_n,u_n)
\right),
\qquad n=0,\ldots,N-1.
\]

It is exactly the ellipse method with the fixed scale \(\sigma_n=1\):

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

`euler_scheme_5` accepts the same `atol`, `rtol`, and `max_iter` controls used by the base ellipse method's implicit solve.

## Prescribed scale

For a prescribed scale, the ellipse method is

\[
u_{n+1}
=u_n+h\mathcal C_{\sigma_n}\!\left(
F(t_{n+1},u_{n+1}),F(t_n,u_n)
\right),
\qquad n=0,\ldots,N-1.
\]

Pass `sigma_n` as either a positive real scalar or a callable. A scalar uses
the same scale at every step.

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
sigma_n(n, t_n, u_n, h) -> positive float
```

It is evaluated once at the accepted left endpoint and frozen during that step's implicit solve.
The index `n` permits prescribed sequences. The fourth argument is denoted by
`h` to match the uniform-mesh notation above. Numerically, the solver passes
the represented interval `t[n + 1] - t[n]`, which equals
\((T-t_0)/N\) in exact arithmetic.
Consequently, the scale may depend explicitly on the mesh:

```python
def sigma_n(n, t_n, u_n, h):
    return h**0.5
```

## Nonlinear solve controls

The prescribed-scale method and `euler_scheme_5` solve their implicit update
by fixed-point iteration, initialized with an Euler predictor. The automatic
fourth-order and defect-minimization modes use a local coupled iteration when
the scale depends on the trial right endpoint. These are local nonlinear
solves: convergence is conditional and is not guaranteed for an arbitrary
field or step size. In particular, this API is not intended to replace a
general stiff ODE solver.

`atol` and `rtol` control termination of the nonlinear iteration, and
`max_iter` limits the number of iterations. A failed solve raises
`RuntimeError`; reducing `h` by increasing `n_steps` is usually the first
remedy. Increasing `max_iter` only helps when the iteration is converging too
slowly.

## Automatic scale-selection modes

Set exactly one of `third_order`, `fourth_order`, or `minimize_defect` to request numerical scale selection.
Leave `sigma_n=None` in any of these modes:

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

minimum_defect_result = specular.ellipse_scheme(
    F,
    0.0,
    1.0,
    1.0,
    n_steps=100,
    minimize_defect=True,
)
```

The three flags request different modes.

**`third_order=True`.** This mode numerically selects a positive
`sigma_n` satisfying the left-endpoint defect-cancellation condition.

**`fourth_order=True`.** This is the strict fourth-order mode. It couples
the trial right endpoint and `sigma_n`. In cases E5a and E5b it uses the
positive defect-balancing scale

\[
\Sigma(t,x;s,y)
=
\left(
\frac{-B+\operatorname{sgn}(A)\sqrt D}{2A}
\right)^{1/2},
\]

evaluated at the current point and the trial right endpoint. In every other
classification case, this mode falls back to `sigma_n=1`. It does not use
zero- or infinite-scale sentinels.

**`minimize_defect=True`.** This mode applies the full E1--E6
two-endpoint classification. Depending on the case, it may select a finite
positive scale, the zero-scale limiting method recorded by `0.0`, or the
infinite-scale Crank--Nicolson limit recorded by `inf`. This is a
defect-minimization mode, not an order mode; it does not promise a maximal
convergence order or fourth-order convergence.

All three modes require derivatives of `F` along solution curves of the ODE:

\[
L_F F,\qquad L_F^2 F,
\qquad L_F=\partial_t+F\partial_u.
\]

By default these are estimated numerically from `F`. `derivative_step` may be
used to set the finite-difference step. Centered local-flow samples can evaluate
`F` just outside `[t_0, T]`, so this mode requires `F` to be defined on a
neighborhood of the time interval. Near a hard domain boundary, provide
`derivatives_of_F` instead. For an exact, symbolic, or automatic-differentiation
implementation, pass a callable that maps a length-two NumPy array to another
length-two array:

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

The flags `third_order`, `fourth_order`, and `minimize_defect` are
mutually exclusive, so at most one may be `True`. Supplying `sigma_n`
together with any of them is also an error: a prescribed scale and an
automatically selected scale are different modes.

!!! warning "Conditional order"

    Third-order mode numerically enforces its cancellation condition, while strict fourth-order mode uses the two-endpoint scale above in E5a/E5b and the fixed-scale fallback `sigma_n=1` otherwise.
    Neither order flag provides an unconditional convergence-order guarantee.
    Third-order convergence still depends on the smoothness and boundedness of the selected branch and on sufficiently accurate field derivatives.
    The fourth-order result is also conditional: the current convergence theorem assumes that all sufficiently close pairs in a tube satisfy the manuscript's uniform E5a-or-E5b condition, together with its stated smoothness, boundedness, and nondegeneracy hypotheses.
    Steps that use the `sigma_n=1` fallback do not inherit that fourth-order guarantee.
    The separate `minimize_defect` mode only minimizes the classified two-endpoint defect and carries no maximal- or fourth-order guarantee.
    Finite-difference error can also produce an accuracy plateau as the mesh is refined.
    Rapid variation, or variation that is small relative to a large additive offset in `F`, may require an explicit `derivative_step` or a `derivatives_of_F` callback.

## Result

`ODEResult.t` and `ODEResult.u` contain the initial value and every accepted step.
`ODEResult.sigma` contains the scale associated with each represented interval and therefore has one fewer entry.
For the Type 1, Type 2, and Type 5 Euler functions it is an array of ones, representing the convention \(\mathcal C=\mathcal C_1\).
In the two-step methods this also includes the externally supplied first interval; it does not describe how `u_1` was produced.
In automatic fourth-order mode, the array records the positive E5a/E5b scale or `1.0` when the selector uses its fallback.
This mode does not place zero- or infinite-scale sentinels in `ODEResult.sigma`.
In `minimize_defect` mode, `0.0` records the zero-scale limiting method and `inf` records the Crank--Nicolson infinite-scale limit.
These are result sentinels, not valid `sigma` arguments for the public `scaled_mean()` function.

`ODEResult.number_of_field_evaluations` records the total number of calls to the supplied field `F(t, u)` made by the solver.
Calls made internally by a user-provided `derivatives_of_F` callback are outside this count.

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
