# Optimization

The optimization API combines a direction rule and a step size rule in

\[
x_{n+1}=x_n+\gamma_n d_n.
\]

`direction.py` defines directions, `step_size.py` defines step sizes, and
`solver.py` combines them. SPEG is the built-in direction:

\[
d_n=-\frac{g_n}{\lVert g_n\rVert},
\qquad g_n=\text{specular gradient of }f\text{ at }x_n.
\]

The solver checks the norm of the **unnormalized** gradient against `tol`
before taking a step. A zero gradient therefore stops the iteration before
normalization. The result records the reason for stopping; reaching a
gradient tolerance does not certify a global minimum.

## SPEG

`specular.specular_gradient` accepts a scalar initial point or a nonempty
one-dimensional vector. The objective returns a real scalar. With scalar
input it receives a scalar. The iteration evaluates vector objectives on
NumPy arrays; differentiation samples use the selected backend's array types.

```python
import specular

--8<-- "examples/optimization/quick_start.py:scalar"
```

The following vector example uses a nonsmooth objective:

```python
import numpy as np

--8<-- "examples/optimization/quick_start.py:vector"
```

Both examples use a decreasing schedule. They illustrate numerical
iteration for a fixed budget and do not assert convergence within that
budget. Constant step sizes can oscillate near a minimizer.

The default `h=None` uses the [calculation API's automatic finite-difference
step](calculation.md). Set `h` to a positive scalar to choose it explicitly.
The `gradient` argument can supply an exact or problem-specific
unnormalized specular gradient instead of numerical differentiation. It
must return a scalar or vector matching the initial point.

## Compose rules

The factories return callables with the following interfaces:

```python
direction(n, x)
step_size(n, x, d)
```

The factories bind the objective, gradient callback, and rule options
once. The solver evaluates the direction, obtains a positive step size,
and accepts `x + gamma * d` if the update is valid.

This smooth quadratic example supplies its analytical gradient to both
SPEG and strong Wolfe search:

```python
from specular.optimization import make_direction, make_step_size, minimize

--8<-- "examples/optimization/quick_start.py:composition"
```

For the same built-in rules, `minimize` also accepts their names directly:

```python
result = minimize(
    quadratic,
    initial_point=[1.0, -2.0],
    direction="speg",
    step_size="strong_Wolfe",
    gradient=quadratic_gradient,
    line_search_gradient=quadratic_gradient,
    step_options={"c_1": 1e-4, "c_2": 0.9},
)
```

Custom direction functions use `(n, x)`. Custom step functions passed
directly to `minimize` use `(n, x, d)`. To adapt a schedule that depends
only on `n`, pass it through `make_step_size`:

```python
--8<-- "examples/optimization/quick_start.py:custom"
```

The named factories also accept optional cached `gradient_value`, and
step size rules accept `f_value`. The solver can use these to avoid
repeating evaluations. Custom functions do not need these extra keywords.

## Step size rules

Schedule indices begin at `n=1` for the first attempted update.

| Rule | Step size or acceptance test | Options |
| :--- | :--- | :--- |
| `constant` | \(a\) | `a=1` |
| `not_summable` | \(\frac{a}{\sqrt n}\) | `a=1` |
| `square_summable_not_summable` | \(\frac{a}{b+n}\) | `a=1`, `b=1` |
| `geometric_series` | \(ar^n\) | `a=1`, `r=0.5` |
| `user_defined` | `user_defined_rule(n)` | `user_defined_rule` callable |
| Callable schedule | `schedule(n)` | Bind with `make_step_size(schedule)` |
| Positive number | Constant step | Bind with `make_step_size(0.1)` |
| `armijo` | Sufficient decrease | `c_1=1e-4`, `rho=0.5` |
| `Wolfe` | Sufficient decrease and curvature | `c_1=1e-4`, `c_2=0.9`, `rho=0.5` |
| `strong_Wolfe` | Sufficient decrease and strong curvature | `c_1=1e-4`, `c_2=0.9`, `rho=0.5` |
| `exact` | Bounded numerical line minimization | `tol=1e-8` |

Line searches also accept `t_0=1`, `max_alpha=1e8`, `max_iter=100`, and
`h=None`. Rule names are case-insensitive. The legacy `c_3` option is an
alias for the strong Wolfe curvature constant.

Pass rule options with `step_options` in `minimize`, or as keyword
arguments to `make_step_size`. The SPEG convenience function also accepts
them directly, as in `specular_gradient(f, x0, a=0.1)`. Unknown options are
rejected rather than silently ignored. The solver's `max_iter`
sets the number of optimization updates; `step_options={"max_iter": ...}`
sets a line search's own budget. For `exact`, that budget counts objective
evaluations, including the initial value when it is not cached.

`exact` is a historical rule name. Its implementation numerically searches
`[0, max_alpha]`; it does not promise an exact or global minimizer for an
arbitrary line objective. Choose `max_alpha` for the scale of the problem.
These rules require no additional dependency.

### Classical and specular search derivatives

Ordinary `armijo`, `Wolfe`, and `strong_Wolfe` use a classical gradient
estimated by centered differences by default. This is separate from the
specular gradient that generates the SPEG direction. Set
`line_search_gradient` in the solver to supply an analytical classical
gradient.

The `specular_armijo`, `specular_Wolfe`, and `specular_strong_Wolfe`
variants use the specular gradient in their slope tests by default.
An explicit `line_search_gradient` overrides that choice. When constructing
a step rule directly, use `make_step_size(..., gradient=...)` for the
gradient used by its tests.

Armijo and Wolfe searches require a negative initial slope computed from
their raw gradient and the search direction. Bounded `exact` minimization
uses objective values only. Its `specular_exact` alias performs the same
derivative-free search.

The search derivative is always **unnormalized**. The normalized SPEG
direction alone cannot supply the derivative information for the tests.
Replacing classical derivatives with specular derivatives, or applying
classical tests at nonsmooth points, does not automatically inherit the
classical convergence guarantees. A search can fail if it cannot find an
acceptable step within its bounds and budget.

## Results and execution

`OptimizationResult` exposes:

| Attribute | Meaning |
| :--- | :--- |
| `solution` | Last accepted point |
| `func_val` | Objective value at that point |
| `iteration` | Number of accepted updates |
| `runtime` | Elapsed solver time in seconds |
| `success` | Whether the convergence criterion was met |
| `stop_reason` | Why the iteration stopped |
| `history` | NumPy arrays under `variables`, `values`, and `step_sizes` |
| `method` | Direction name, or `custom` for a callable direction |

The `variables` and `values` histories include the initial point and accepted
iterates, with length `iteration + 1`. `variables` has shape
`(iteration + 1,)` for scalar states or `(iteration + 1, dimension)` for
vectors. `step_sizes` has one entry per accepted update, so its length is
`iteration`. Failed searches
do not append an iterate or change the last accepted solution. Use
`record_history=False` to leave the history arrays empty and avoid storing
the trajectory. The final point and objective remain available in the result.

`last_record()` returns `(solution, func_val, runtime)`. `get_history()`
returns `(history["variables"], history["values"], runtime)`.

Stopping reasons distinguish a gradient below tolerance, an exhausted
iteration budget, a failed line search, a zero direction despite a larger
gradient, an update lost to floating-point precision, and non-finite
proposed points or objective values. Invalid arguments and callback shape
errors raise exceptions.

### Backend selection

The optimizer runs an eager Python loop with NumPy state. Its default
specular differentiation uses the currently selected calculation backend,
then converts values back to NumPy for iteration. Selecting Numba or JAX
does not compile the whole optimizer or make it a JAX-transformable solver.

Use `set_backend` or `use_backend` just as for the calculation and ODE APIs.
For example, after installing the optional Numba backend:

```python
with specular.use_backend("numba"):
    result = specular.specular_gradient(
        sum_abs, [1.0, -2.0], step_size=0.1, max_iter=20
    )
```

The objective must be compatible with the selected backend. In particular,
JAX differentiation needs a JAX-compatible objective, such as one written
with `jax.numpy`. Ordinary centered line-search differences run in NumPy;
specular line searches use the selected calculation backend. Explicit
gradient callbacks perform their own computation. See
[backend support](backend.md) for installation and dtype details.

## API reference

::: specular.optimization.solver.specular_gradient
    options:
      show_root_heading: true

::: specular.optimization.solver.minimize
    options:
      show_root_heading: true

::: specular.optimization.direction.make_direction
    options:
      show_root_heading: true

::: specular.optimization.step_size.make_step_size
    options:
      show_root_heading: true

::: specular.optimization.solver.OptimizationResult
    options:
      show_root_heading: true
