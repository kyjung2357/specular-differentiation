---
title: specular.ode.ellipse_scheme
api_reference: true
---

::: specular.ode.ellipse_scheme
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Solve a scalar initial-value problem on `[t_0, T]` with the implicit specular
ellipse method.

## Parameters

| Name | Type and default | Description |
| --- | --- | --- |
| `F` | callable | Field `F(t, u)` returning a finite real scalar. |
| `t_0` | real scalar | Finite initial time. |
| `T` | real scalar | Finite final time, strictly greater than `t_0`. |
| `u_0` | real scalar | Finite initial value at `t_0`. |
| `n_steps` | positive integer, keyword-only | Number of time intervals; the result has `n_steps + 1` nodes. |
| `sigma_n` | positive real scalar or callable, default `None` | Required in the base mode. A callable `sigma_n(n, t_n, u_n, h)` returns a finite positive scale and is evaluated once at the accepted left endpoint. Here `h` is the represented interval `t[n + 1] - t[n]`. Must be `None` in an automatic mode. |
| `third_order` | bool, default `False` | Select a positive scale using numerical left-endpoint defect cancellation. |
| `fourth_order` | bool, default `False` | Use the coupled two-endpoint scale in cases E5a/E5b and scale `1.0` in the other cases. |
| `minimize_defect` | bool, default `False` | Apply the full E1--E6 two-endpoint defect-minimization rule, including zero- and infinite-scale limits. |
| `derivatives_of_F` | callable or `None`, default `None` | In an automatic mode, optionally map the NumPy array `[t, u]` to `[L_F F, L_F^2 F]`, also of shape `(2,)`. Omit to estimate these derivatives numerically. |
| `derivative_step` | positive real scalar or `None`, default `None` | Finite-difference step for automatic modes. Cannot be combined with `derivatives_of_F`. |
| `atol` | real scalar, default `1e-12` | Finite nonnegative absolute tolerance for the nonlinear solve. |
| `rtol` | real scalar, default `1e-10` | Finite nonnegative relative tolerance for the nonlinear solve. At least one of `atol` and `rtol` must be positive. |
| `max_iter` | positive integer, default `100` | Limit for each local nonlinear iteration or root-refinement phase. Bracket searches may make additional field evaluations. |

All parameters after `u_0` are keyword-only. The three automatic-mode flags are
mutually exclusive. Derivative controls are valid only in an automatic mode.
Here \(L_F=\partial_t+F\partial_u\) denotes differentiation along solution curves.

## Returns

[`ODEResult`](ODEResult.md): NumPy float64 arrays `t` and `u` of shape `(n_steps + 1,)`, a scale array
`sigma` of shape `(n_steps,)`, and the number of solver calls to `F`.

## Notes

For a prescribed scale, the update is

\[
u_{n+1}=u_n+h\mathcal C_{\sigma_n}\!\left(
F(t_{n+1},u_{n+1}),F(t_n,u_n)\right).
\]

The scale is frozen during each implicit solve. Angular means use the selected
calculation backend; the outer iteration uses Python floats and NumPy float64
arrays. See [Prescribed scale](../../user-guide/ode.md#prescribed-scale).

Local nonlinear convergence is conditional. A failed solve raises
`RuntimeError`; a smaller time step may help. Exceptions from `F` are preserved.
See [Nonlinear solve controls](../../user-guide/ode.md#nonlinear-solve-controls).

The order flags do not guarantee their named order for arbitrary fields.
Third-order convergence requires a sufficiently smooth, bounded selected branch
and accurate derivatives. The fourth-order theorem additionally requires the
uniform E5a-or-E5b condition near the solution and the stated nondegeneracy
assumptions; steps using the scale-1 fallback do not inherit that guarantee.
`minimize_defect` is a defect-minimization mode with no fourth-order or maximal-order
guarantee. See [Automatic scale-selection modes](../../user-guide/ode.md#automatic-scale-selection-modes).

Numerical derivative estimation may sample `F` outside `[t_0, T]`; the field must
be defined on a neighborhood of the interval, or exact derivatives must be
supplied. Only `minimize_defect` records the result sentinels `0.0` and `inf`
for limiting methods. They are not valid prescribed scales or public
`scaled_mean` scale arguments.

## Examples

```python
import specular.ode

result = specular.ode.ellipse_scheme(
    lambda t, u: -u,
    0.0,
    1.0,
    1.0,
    n_steps=100,
    sigma_n=1.0,
)
print(result.t[-1], result.u[-1])
```
