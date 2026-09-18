---
title: specular.ode.euler_scheme_5
api_reference: true
---

::: specular.ode.euler_scheme_5
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Apply the implicit one-step specular Euler method of Type 5 to a scalar
initial-value problem. This is [`ellipse_scheme`](ellipse_scheme.md) with
the fixed scale \(\sigma_n=1\).

## Parameters

| Name | Type and default | Description |
| --- | --- | --- |
| `F` | callable | Field `F(t, u)` returning a finite real scalar. |
| `t_0` | real scalar | Finite initial time. |
| `T` | real scalar | Finite final time, strictly greater than `t_0`. |
| `u_0` | real scalar | Finite initial value at `t_0`. |
| `n_steps` | positive integer, keyword-only | Number of uniform time intervals. |
| `atol` | real scalar, default `1e-12` | Finite nonnegative absolute tolerance for the nonlinear solve. |
| `rtol` | real scalar, default `1e-10` | Finite nonnegative relative tolerance for the nonlinear solve. At least one of `atol` and `rtol` must be positive. |
| `max_iter` | positive integer, default `100` | Limit for each nonlinear iteration or root-refinement phase. Bracket searches may make additional field evaluations. |

All parameters after `u_0` are keyword-only.

## Returns

[`ODEResult`](ODEResult.md): NumPy float64 arrays `t` and `u` of shape `(n_steps + 1,)`, an array `sigma`
of `n_steps` ones, and the number of solver calls to `F`.

## Notes

The method solves

\[
u_{n+1}=u_n+h\mathcal C_1\!\left(
F(t_{n+1},u_{n+1}),F(t_n,u_n)\right)
\]

at each step, where \(h=(T-t_0)/N\) and \(N=\texttt{n_steps}\).
Unlike the two-step Euler methods, it does not require an external `u_1`.
See [Type 5](../../user-guide/ode.md#type-5).

Fixed-point iteration uses an Euler predictor, with a local bracket and
bisection fallback. Convergence is conditional; a failed solve raises
`RuntimeError`. See [Nonlinear solve controls](../../user-guide/ode.md#nonlinear-solve-controls).
Angular means use the selected backend; the outer iteration remains scalar
and the stored arrays use float64.

## Examples

```python
import specular.ode

result = specular.ode.euler_scheme_5(
    lambda t, u: -u,
    0.0,
    1.0,
    1.0,
    n_steps=100,
)
print(result.u[-1])
```
