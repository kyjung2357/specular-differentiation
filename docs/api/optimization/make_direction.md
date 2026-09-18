---
title: specular.optimization.make_direction
api_reference: true
---

::: specular.optimization.make_direction
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Create a SPEG or custom direction callable.

## Parameters

| Parameter | Description |
| :--- | :--- |
| `method="speg"` | `"speg"`, its alias `"specular_gradient"`, or a custom callable accepting `(n, x)`. |
| `f=None` | Scalar-valued objective for numerical specular differentiation. Required for a named method unless `gradient` is supplied. |
| `gradient=None` | Raw specular gradient callback `gradient(x)`. Takes precedence over numerical differentiation of `f`. |
| `h=None` | Concrete, finite, positive scalar differentiation interval. If omitted, the selected backend chooses intervals automatically. |

## Returns

A callable with the interface:

```python
direction(n, x, *, gradient_value=None)
```

It returns a finite real scalar or a one-dimensional NumPy array with the
same shape as `x`. The iteration index `n` must be an integer at least one.

## Notes

SPEG returns the negative unit vector of the raw specular gradient, or zero
when that gradient is exactly zero. `gradient_value` can reuse a previously
computed raw gradient. This cached value must match the shape of `x`.

A custom `method(n, x)` is used directly, without normalization, and ignores
`gradient_value`. Callbacks receive private copies of vector points.
Both points and returned directions must be finite scalars or nonempty
one-dimensional vectors with exactly matching shapes.

Numerical differentiation uses the currently selected
[calculation backend](../backends/index.md). An explicit gradient callback
performs its own computation.

## Examples

```python
import numpy as np

from specular.optimization import make_direction

direction = make_direction("speg", gradient=lambda x: 2.0 * x)
print(direction(1, np.array([3.0, 4.0])))
# [-0.6 -0.8]

custom_direction = make_direction(lambda n, x: -2.0 * x)
print(custom_direction(1, 3.0))
# -6.0
```
