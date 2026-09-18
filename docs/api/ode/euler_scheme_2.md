---
title: specular.ode.euler_scheme_2
api_reference: true
---

::: specular.ode.euler_scheme_2
    options:
      heading_level: 1
      show_root_heading: true
      show_root_toc_entry: true
      show_root_full_path: true
      show_docstring_description: false
      show_source: true
      members: false

Apply the explicit two-step specular Euler method of Type 2 to a scalar
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
F(t_n,u_n),\frac{u_n-u_{n-1}}{h}\right),\qquad n=1,\ldots,N-1.
\]

The implementation uses each represented time interval in the update and
backward difference. The solver keeps `u_1` unchanged; with `n_steps=1`,
it returns the two supplied values without evaluating `F`. The first entry
of `sigma` records the unscaled convention, not the caller's starter method.

Type 2 is generically first-order consistent and differs from the second-order
ellipse configuration labelled SE2 on the numerical-examples page. See
[Type 2](../../user-guide/ode.md#type-2). Angular means use the selected backend;
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
result = specular.ode.euler_scheme_2(
    F, t_0, T, u_0, u_1, n_steps=n_steps,
)
print(result.u[-1])
```
