---
title: specular.optimization.specular_gradient
api_reference: true
---

::: specular.optimization.specular_gradient
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Run SPEG, using the normalized negative specular gradient as the direction.
Also available as `specular.specular_gradient`.

## Parameters

| Parameter | Description |
| :--- | :--- |
| `objective_function` | Callable returning a real scalar objective value. |
| `initial_point` | Finite real scalar or nonempty one-dimensional vector. |
| `step_size="constant"` | Named rule, positive constant, or callable accepting `(n, x, d)`. See [`make_step_size`](make_step_size.md). |
| `max_iter=1000` | Nonnegative integer limiting the number of accepted updates. |
| `tol=1e-6` | Finite, nonnegative tolerance for the unnormalized gradient norm. |
| `h=None` | Positive finite scalar differentiation interval, or automatic backend-dependent intervals when omitted. |
| `gradient=None` | Optional raw specular gradient callback `gradient(x)`, matching the shape of `initial_point`. |
| `line_search_gradient=None` | Optional raw gradient callback for a named line search's slope tests. |
| `record_history=True` | Whether to record the initial point and accepted iterates. |
| `step_options=None` | Mapping of options for a named or numeric step rule. |
| `**step_parameters` | Additional named step options, such as `a=0.5` or `b=1.0`. |

## Returns

An [`OptimizationResult`](OptimizationResult.md) containing the last accepted
point, its objective value, the stopping reason, and optional history.

## Notes

This convenience wrapper delegates to [`minimize`](minimize.md) with
`direction="speg"`. It tests the norm of the raw gradient before
normalization. A zero gradient stops the iteration; satisfying `tol` does
not certify a global minimum.

Options may be supplied through `step_options` or directly as keywords.
Supplying the same option through both raises `TypeError`. Set
`step_options={"max_iter": ...}` to configure the line search budget
separately from the solver's `max_iter`.

Ordinary Armijo and Wolfe rules use classical centered differences unless
`line_search_gradient` is supplied. Rules prefixed with `specular_` use the
raw specular gradient by default. The two derivative choices have different
meanings; see the [optimization user guide](../../user-guide/optimization.md#classical-and-specular-search-derivatives).

## Examples

```python
import specular

result = specular.specular_gradient(
    abs,
    initial_point=1.0,
    step_size="square_summable_not_summable",
    a=0.5,
    b=1.0,
    max_iter=200,
)
print(result.solution, result.func_val, result.stop_reason)
```
