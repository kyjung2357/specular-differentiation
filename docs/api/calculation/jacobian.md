---
title: specular.calculation.jacobian
api_reference: true
---

::: specular.calculation.jacobian
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Evaluate a specular Jacobian with the selected backend. Also available as
`specular.jacobian`.

## Parameters

| Name | Description |
| :--- | :--- |
| `f` | Callable taking a one-dimensional real vector and returning a nonempty one-dimensional real vector of fixed shape. |
| `x` | Nonempty one-dimensional array-like of finite real coordinates. |
| `h` | Finite, positive, concrete real scalar shared by all coordinates, or `None` for automatic coordinatewise selection. Default: `None`. |

## Returns

A matrix of shape `(m, n)`, where `n` is the input dimension and `m` is the
output dimension. Entry `(i, j)` is the specular derivative of output
component `i` with respect to input coordinate `j`. NumPy and Numba return
NumPy arrays; JAX returns JAX arrays.

## Notes

The center value is evaluated once, with two additional samples per input
coordinate. Automatic steps are selected separately for each coordinate.
Validation and backend constraints match those of
[`gradient`](gradient.md).

The JAX backend samples coordinates in bounded batches. The returned matrix
still requires storage proportional to `m * n`. See the
[backend guide](../../user-guide/backends.md) for dtype and callback requirements.

## Examples

```python
import numpy as np
from specular.calculation import jacobian

def f(x):
    return np.array([x[0] + 2.0 * x[1], 3.0 * x[0] - x[1]])

print(jacobian(f, [1.0, 2.0]))
# approximately [[1, 2], [3, -1]]
```
