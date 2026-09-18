---
title: specular.calculation.scaled_mean
api_reference: true
---

::: specular.calculation.scaled_mean
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Evaluate the scaled angular mean elementwise. Also available as
`specular.scaled_mean`.

## Parameters

| Name | Description |
| :--- | :--- |
| `alpha`, `beta` | Real scalars or broadcast-compatible arrays of slopes. |
| `sigma` | Concrete, finite, positive real scalar. Default: `1.0`. |

## Returns

A scalar or array with the broadcast shape of `alpha` and `beta`.
NumPy and Numba return a Python float for scalar results and a NumPy array
otherwise; JAX returns a JAX array. Arithmetic uses the selected backend's
calculation dtype.

## Notes

The scale parameter defines

\[
\mathcal C_\sigma(\alpha,\beta)
=\sigma\mathcal C(\alpha/\sigma,\beta/\sigma).
\]

Inputs are promoted before arithmetic. Scale-safe formulas recover
representable results when direct rescaling would overflow or underflow.
The exact identities

\[
\mathcal C_\sigma(\alpha,\alpha)=\alpha,
\qquad \mathcal C_\sigma(\alpha,-\alpha)=0
\]

are preserved for finite slopes even in those regimes. Other results remain
subject to the backend dtype's representable range and device treatment
of subnormal values. Under JAX transformations, `alpha` and `beta` may be
traced, but `sigma` must remain static.

See the [calculation guide](../../user-guide/calculation.md) for background.

## Examples

```python
from specular.calculation import scaled_mean

print(scaled_mean(1.0, -0.5, sigma=2.0))
print(scaled_mean([1.0, 2.0], [-1.0, 2.0]))  # [0. 2.]
```
