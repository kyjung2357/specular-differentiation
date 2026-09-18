# specular.ode

Scalar specular methods for initial-value problems \(u'(t)=F(t,u(t))\).
These methods return numerical data on a uniform time grid; vector-valued
states are not supported.

| Function or class | Description |
| --- | --- |
| [`ellipse_scheme`](ellipse_scheme.md) | Implicit specular ellipse method with prescribed or automatic scale selection. |
| [`euler_scheme_1`](euler_scheme_1.md) | Explicit two-step specular Euler method of Type 1. |
| [`euler_scheme_2`](euler_scheme_2.md) | Explicit two-step specular Euler method of Type 2. |
| [`euler_scheme_5`](euler_scheme_5.md) | Implicit one-step specular Euler method of Type 5. |
| [`ODEResult`](ODEResult.md) | Time nodes, solution values, scales, and field-evaluation count. |

The same exports are available directly from `specular`, for example
`specular.ellipse_scheme(...)`.
For the recurrences, scale-selection modes, and conditional convergence-order
assumptions, see [Scalar ODE methods](../../user-guide/ode.md) in the User Guide.
