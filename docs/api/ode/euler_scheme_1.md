---
title: specular.ode.euler_scheme_1
api_reference: true
---

::: specular.ode.euler_scheme_1
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Apply the explicit two-step specular Euler method of Type 1 to a scalar
initial-value problem.

## Parameters

| Name | Type | Description |
| --- | --- | --- |
| `F` | callable | Field `F(t, u)` returning a finite real scalar. |
| `t_0` | real scalar | Finite initial time. |
| `T` | real scalar | Finite final time, strictly greater than `t_0`. |
| `u_0` | real scalar | Finite initial value at `t_0`. |
| `u_1` | real scalar | Finite, externally computed starting value at the first time node `t[1]`. |
| `n_steps` | positive integer, keyword-only | Number of uniform time intervals. |

## Returns

[`ODEResult`](ODEResult.md): NumPy float64 arrays `t` and `u` of shape `(n_steps + 1,)`, an array `sigma`
of `n_steps` ones, and the number of solver calls to `F`.

## Notes

With \(h=(T-t_0)/N\), \(N=\texttt{n_steps}\), and
\(\mathcal C=\mathcal C_1\), the recurrence is

\[
u_{n+1}=u_n+h\mathcal C\!\left(
F(t_n,u_n),F(t_{n-1},u_{n-1})\right),\qquad n=1,\ldots,N-1.
\]

The solver keeps the supplied `u_1` unchanged. For `n_steps=1`, it returns
`u_0` and `u_1` without evaluating `F`. The all-ones `sigma` array describes the
unscaled-mean convention, including the externally supplied first interval;
it does not describe how the caller produced `u_1`.

This method is generically first-order consistent. A more accurate starter
does not give an unconditional higher-order theorem. See
[Type 1](../../user-guide/ode.md#type-1). Angular means use the selected backend;
the outer iteration remains scalar and the stored arrays use float64.

## Examples

```python
import specular.ode

def F(t, u):
    return -u

t_0, T, u_0 = 0.0, 1.0, 1.0
n_steps = 100
h = (T - t_0) / n_steps
u_1 = u_0 + h * F(t_0, u_0)
result = specular.ode.euler_scheme_1(
    F, t_0, T, u_0, u_1, n_steps=n_steps,
)
print(result.u[-1])
```
