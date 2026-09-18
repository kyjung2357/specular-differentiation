---
title: specular.optimization.make_step_size
api_reference: true
---

::: specular.optimization.make_step_size
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Create a decreasing schedule, a constant step, or a bounded line search.

## Parameters

| Parameter | Description |
| :--- | :--- |
| `rule="constant"` | Rule name, positive finite scalar, or schedule callable accepting `n`. Names are case-insensitive. |
| `f=None` | Scalar-valued objective, required for line searches. |
| `gradient=None` | Optional raw gradient callback for line-search slopes. Takes precedence over the selected rule's default derivative. |
| `**options` | Options specific to the rule, listed below. Unknown options raise `TypeError`. |

### Schedules

| Rule | Step for \(n\geq1\) | Options |
| :--- | :--- | :--- |
| `constant` | \(a\) | `a=1`, with `a > 0` |
| `not_summable` | \(a/\sqrt n\) | `a=1`, with `a > 0` |
| `square_summable_not_summable` | \(a/(b+n)\) | `a=1`, `b=1`, with `a > 0`, `b >= 0` |
| `geometric_series` | \(ar^n\) | `a=1`, `r=0.5`, with `a > 0`, `0 < r < 1` |
| `user_defined` | `user_defined_rule(n)` | `user_defined_rule` callable |
| Callable | `rule(n)` | No additional options |
| Positive number | `rule` | No additional options |

All numeric schedule parameters must be finite. Every evaluated step must
remain finite and strictly positive; for example, a geometric schedule that
underflows to zero raises an error.

### Line searches

| Rule | Acceptance or minimization condition | Specific options |
| :--- | :--- | :--- |
| `armijo` | Sufficient decrease | `c_1=1e-4`, `rho=0.5` |
| `Wolfe` | Sufficient decrease and curvature | `c_1=1e-4`, `c_2=0.9`, `rho=0.5` |
| `strong_Wolfe` | Sufficient decrease and strong curvature | `c_1=1e-4`, `c_2=0.9`, `rho=0.5` |
| `exact` | Bounded numerical minimization | `tol=1e-8` |

Each search accepts `t_0=1`, `max_alpha=1e8`, `max_iter=100`, and `h=None`.
The first two must be finite and positive; `max_iter` must be a positive
integer. An explicit `h` must be finite and positive. Armijo and Wolfe
require `0 < c_1 < 1` and `0 < rho < 1`; Wolfe also requires
`c_1 < c_2 < 1`. Strong Wolfe accepts the legacy `c_3` alias instead of
`c_2`, but not both. The bounded minimizer requires positive finite `tol`.

Each search name also accepts the `specular_` prefix. For Armijo and Wolfe
this changes the default derivative to the specular gradient.
`specular_exact` is identical to `exact`.

## Returns

A callable with the interface:

```python
step(n, x=None, d=None, *, gradient_value=None, f_value=None)
```

The returned value is a positive finite scalar. Indices start at one.
Schedules only need `n`; searches also need finite scalar or nonempty
one-dimensional points `x` and directions `d` with matching shapes.

## Notes

Armijo and Wolfe searches require a negative initial slope computed from
the **raw** gradient and the direction. Ordinary rules use centered
differences when no gradient callback is supplied; specular rules use the
selected calculation backend. `gradient_value` and `f_value` can cache the
appropriate raw gradient and objective at `x`. A normalized direction must
not be supplied as the raw gradient.

`exact` uses derivative-free golden-section minimization on
`[0, max_alpha]`. Its `max_iter` counts objective evaluations, including the
initial value if it is not cached. The name is historical: it does not
promise symbolic exactness or a global minimum for arbitrary objectives.
Choose `max_alpha` for the problem's scale.

Failed searches raise [`LineSearchError`](LineSearchError.md), without an
unchecked fallback step. Classical smooth Wolfe convergence guarantees do
not automatically extend to specular slopes or nonsmooth objectives.
See the [optimization user guide](../../user-guide/optimization.md#step-size-rules)
for composition examples and derivative choices.

## Examples

```python
from specular.optimization import make_step_size

schedule = make_step_size("square_summable_not_summable", a=0.5, b=1.0)
print(schedule(1))
# 0.25

search = make_step_size(
    "armijo",
    f=lambda x: x * x,
    gradient=lambda x: 2.0 * x,
)
print(search(1, x=2.0, d=-1.0))
# 1.0
```
