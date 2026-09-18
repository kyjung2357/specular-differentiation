---
title: specular.ode.ODEResult
api_reference: true
---

::: specular.ode.ODEResult
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false
      merge_init_into_class: false

Container for the numerical data returned by every scalar ODE method.

## Attributes

| Name | Type | Description |
| --- | --- | --- |
| `t` | NumPy float64 array, shape `(n_steps + 1,)` | Time nodes, including both endpoints. |
| `u` | NumPy float64 array, shape `(n_steps + 1,)` | Initial value followed by the solution values at the time nodes. For Types 1/2, `u[1]` is the externally supplied starter. |
| `sigma` | NumPy float64 array, shape `(n_steps,)` | Scale associated with each represented time interval. |
| `number_of_field_evaluations` | int | Number of calls to `F(t, u)` made by the solver. |

## Notes

This is a frozen, slotted dataclass. Its attribute bindings cannot be reassigned,
but the contained NumPy arrays are mutable. Results contain numerical data;
plotting, event handling, and dense output are outside this API.

For all three Euler methods, `sigma` contains ones. In Types 1/2, this convention
also covers the first interval and does not describe how `u_1` was produced.

In automatic fourth-order mode, `sigma` records the positive E5a/E5b scale or
the fallback `1.0`. Only `minimize_defect` may use `0.0` for the zero-scale
limiting method or `inf` for the infinite-scale Crank--Nicolson limit. These
are result sentinels, not valid scales for the public `scaled_mean` function.
See [Result](../../user-guide/ode.md#result).

Field calls made internally by a user-provided `derivatives_of_F` callback
are not included in `number_of_field_evaluations`.

## Examples

```python
import specular.ode

result = specular.ode.ellipse_scheme(
    lambda t, u: -u, 0.0, 1.0, 1.0,
    n_steps=100, sigma_n=1.0,
)
print(result.t.shape, result.u.shape, result.sigma.shape)
print(result.number_of_field_evaluations)
```
