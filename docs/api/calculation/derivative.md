---
title: specular.calculation.derivative
api_reference: true
---

::: specular.calculation.derivative
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Evaluate a specular derivative with the selected backend. Also available
as `specular.derivative`.

## Parameters

| Name | Description |
| :--- | :--- |
| `f` | Callable taking a real scalar and returning a real scalar or nonempty one-dimensional vector of fixed shape. |
| `x` | Finite real scalar at which to evaluate the derivative. |
| `h` | Finite, positive, concrete real step size, or `None` for automatic selection. Default: `None`. |

## Returns

A scalar for scalar-valued `f`, or a vector of shape `(m,)` when `f` returns
a vector of length `m`. NumPy and Numba use Python floats or NumPy arrays;
JAX uses JAX arrays.

## Notes

The function samples `f` at `x`, `x - h`, and `x + h`, evaluates the center
once, and combines the one-sided secant slopes using the specular mean.
For vector-valued `f`, this is done componentwise.

If `h` is omitted, the backend selects
`eps(dtype)**(1/3) * max(1, abs(x))`. An explicit step is validated before
`f` is evaluated. Steps that cannot produce distinct finite samples are
rejected for concrete inputs; dynamically traced JAX inputs cannot be
checked in the same way. A representable step can still be too small for
an accurate approximation.

The callback must be compatible with the selected backend. Under `jax.jit`,
an explicit `h` must be static. See the
[calculation guide](../../user-guide/calculation.md) and
[backend guide](../../user-guide/backends.md).

## Examples

```python
import numpy as np
from specular.calculation import derivative

print(derivative(lambda x: max(x, 0.0), 0.0))  # approximately sqrt(2) - 1
print(derivative(lambda x: np.array([x, x * x]), 2.0))  # approximately [1, 4]
```
