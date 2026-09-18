---
title: specular.optimization.minimize
api_reference: true
---

::: specular.optimization.minimize
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Combine a direction and a step size rule in the iteration
\(x_{n+1}=x_n+\gamma_n d_n\). Also available as `specular.minimize`.

## Parameters

| Parameter | Description |
| :--- | :--- |
| `objective_function` | Callable returning a real scalar. Its value at the initial point must be finite. |
| `initial_point` | Finite real scalar or nonempty one-dimensional vector. |
| `direction="speg"` | `"speg"`, its alias `"specular_gradient"`, or a callable `(n, x)` returning a finite direction with the same shape as `x`. |
| `step_size="constant"` | Named step rule, positive finite scalar, or callable `(n, x, d)` returning a positive finite scalar. |
| `gradient=None` | Raw gradient callback `gradient(x)` for SPEG and the stopping test. If omitted, numerical specular differentiation uses the selected backend. |
| `line_search_gradient=None` | Raw gradient callback for a named line search. Overrides its default classical or specular derivative. |
| `h=None` | Positive finite scalar interval for numerical specular differentiation; omitted intervals are chosen automatically. |
| `max_iter=1000` | Nonnegative integer limiting accepted updates. Zero returns the initial point without a gradient check. |
| `tol=1e-6` | Finite, nonnegative threshold for the raw gradient norm. |
| `record_history=True` | Whether to record trajectory arrays. Must be a Boolean. |
| `step_options=None` | Mapping of named or numeric step rule options; see [`make_step_size`](make_step_size.md). |

## Returns

An [`OptimizationResult`](OptimizationResult.md). With history enabled,
point and objective histories have `iteration + 1` entries, and the step
history has `iteration` entries.

## Notes

Indices passed to rule callables start at one. A custom direction is not
normalized automatically. A schedule accepting only `n` must first be
adapted with [`make_step_size`](make_step_size.md). For callable step rules,
configure any gradient on the callable itself; `step_options` and
`line_search_gradient` cannot be supplied to this solver for that callable.

The stopping test uses the raw gradient, including when the direction is
custom. A custom direction's size alone is therefore not a convergence
criterion. The solver also checks the gradient at the last permitted
iterate when `max_iter` is positive.

The solver runs an eager Python loop with NumPy state. Scalar objectives
receive scalar points; vector objectives receive NumPy arrays during
iteration and backend-specific arrays during numerical differentiation.
Selecting Numba or JAX changes the calculation backend without compiling
the complete solver. See the [backend guide](../../user-guide/optimization.md#backend-selection).

Ordinary Armijo and Wolfe searches use classical centered differences by
default. The `specular_` variants instead use the raw specular gradient.
`line_search_gradient` overrides either choice. A line search's own
finite-difference interval can be configured with `step_options={"h": ...}`
when it uses ordinary centered differences. Default specular searches share
the solver's `h`; a supplied `line_search_gradient` controls its own differentiation.
Specular substitutions do not automatically inherit classical Wolfe
convergence guarantees.

A [`LineSearchError`](LineSearchError.md) is recorded as a failed search
in the result. A non-finite proposed point or objective, a zero direction,
or a step lost to working precision also stops without replacing the last
accepted point. Invalid arguments and callback shape errors raise exceptions.

## Examples

```python
import numpy as np

from specular.optimization import minimize


def quadratic(x):
    return np.dot(x, x)


def quadratic_gradient(x):
    return 2.0 * x


result = minimize(
    quadratic,
    initial_point=[1.0, -2.0],
    direction="speg",
    step_size="strong_Wolfe",
    gradient=quadratic_gradient,
    line_search_gradient=quadratic_gradient,
    max_iter=100,
)
print(result.solution, result.stop_reason)
```
