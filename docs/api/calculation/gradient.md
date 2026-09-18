---
title: specular.calculation.gradient
api_reference: true
---

::: specular.calculation.gradient
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Evaluate the specular gradient of a scalar-valued function. Also available
as `specular.gradient`.

## Parameters

| Name | Description |
| :--- | :--- |
| `f` | Callable taking a one-dimensional real vector and returning a real scalar. |
| `x` | Nonempty one-dimensional array-like of finite real coordinates. |
| `h` | Finite, positive, concrete real scalar shared by all coordinates, or `None` for automatic coordinatewise selection. Default: `None`. |

## Returns

A vector of shape `(n,)`, where `n` is the length of `x`. The return value
is a NumPy array for NumPy and Numba, or a JAX array for JAX.

## Notes

Each component is the specular derivative along one coordinate direction.
The center value is evaluated once. With `h=None`, coordinate `j` uses
`eps(dtype)**(1/3) * max(1, abs(x[j]))`.

An explicit `h` is validated before evaluating `f`. Concrete samples must
be finite and distinct; a very small valid step can still lose accuracy.
The callback must support the selected backend, and JAX requires a static
explicit step under transformations. See the
[calculation guide](../../user-guide/calculation.md) for these limitations.

The result is the unnormalized gradient. Normalization for SPEG is handled
by [`make_direction`](../optimization/make_direction.md).

## Examples

```python
import numpy as np
from specular.calculation import gradient

print(gradient(lambda x: np.sum(x * x), [1.0, 2.0, 3.0]))
# approximately [2, 4, 6]
```
