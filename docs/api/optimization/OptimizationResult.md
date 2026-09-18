---
title: specular.optimization.OptimizationResult
api_reference: true
---

::: specular.optimization.OptimizationResult
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false
      merge_init_into_class: false

Store the last accepted point and numerical termination information from
[`minimize`](minimize.md) or [`specular_gradient`](specular_gradient.md).
Also available as `specular.OptimizationResult`.

## Attributes

| Attribute | Description |
| :--- | :--- |
| `solution` | Last accepted point, as a float or one-dimensional NumPy array. |
| `func_val` | Objective value at `solution`, as a float. |
| `iteration` | Number of accepted updates. |
| `runtime` | Elapsed solver time in seconds. |
| `history` | Dictionary of NumPy arrays with keys `variables`, `values`, and `step_sizes`. |
| `stop_reason` | Text describing why the solver stopped. |
| `success` | Whether the raw gradient norm met the requested tolerance. |
| `method` | Direction name, or `"custom"` for a callable direction. Defaults to `"speg"` when constructing a result manually. |

## Methods

| Method | Returns |
| :--- | :--- |
| `last_record()` | `(solution, func_val, runtime)` |
| `get_history()` | `(history["variables"], history["values"], runtime)` |

## Notes

With history enabled, `variables` and `values` include the initial point
and every accepted iterate. `variables` has shape `(iteration + 1,)` for a
scalar problem and `(iteration + 1, dimension)` for a vector problem.
`step_sizes` has `iteration` entries. All three arrays are empty when
`record_history=False`.

The result is a frozen dataclass: fields cannot be reassigned, but contained
NumPy arrays and the history dictionary are still mutable. The accessors
return the stored objects, without making copies.

`success=True` means the computed raw gradient norm met the tolerance; it
does not certify a global minimizer. Stopping reasons distinguish gradient
tolerance, an exhausted update budget, a failed line search, a zero
direction with a larger gradient, lost floating-point progress, and a
non-finite proposed point or objective. Failed updates do not change the
last accepted point or append history.

## Examples

```python
from specular.optimization import specular_gradient

result = specular_gradient(
    lambda x: x * x,
    initial_point=2.0,
    gradient=lambda x: 2.0 * x,
    step_size=1.0,
    max_iter=5,
)
print(result.solution, result.iteration, result.success)
# 0.0 2 True

points, values, runtime = result.get_history()
print(points)
# [2. 1. 0.]
```
