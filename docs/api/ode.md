# Scalar ODE schemes

The ODE API solves scalar initial-value problems

\[
y'(t)=f(t,y(t)), \qquad y(t_0)=y_0.
\]

It does not currently support vector-valued states. The three schemes are
available as `specular.ellipse_scheme`,
`specular.ellipse_scheme_3rd_order`, and
`specular.ellipse_scheme_4th_order`; importing `specular` does not eagerly load
the ODE implementation.

The result contains numerical data only. Plotting, table generation, event
handling, and dense output are not part of this API.

## Scale policies

Every scheme requires `sigma` to be either a positive real scalar or a
callable returning a positive real scalar. A scalar uses the same scale at
every evaluation.

For `ellipse_scheme` and `ellipse_scheme_3rd_order`, a callable has the
contract

```python
sigma(n, t_n, y_n, h) -> positive float
```

It is evaluated at the accepted left endpoint and is fixed during that
step's implicit solve. The step index `n` makes prescribed sequences easy to
represent, while the explicit `h` argument permits mesh-dependent policies:

```python
def mesh_scale(n, t_n, y_n, h):
    return h**0.5
```

For `ellipse_scheme_4th_order`, a callable has the coupled two-endpoint
contract

```python
sigma(n, t_n, y_n, t_next, y_trial, h) -> positive float
```

The policy is reevaluated as the trial right-endpoint value changes during the
implicit solve. Supplying a scalar is computationally valid, but it does not
enforce the coupled cancellation needed for a general fourth-order result.

## Order guarantees

The order in a function name describes the mathematical selector contract,
not an order that can be inferred from an arbitrary scale policy.

- `ellipse_scheme` implements the base specular ellipse scheme and allows any
  positive scale sequence.
- `ellipse_scheme_3rd_order` requires its left-endpoint policy to satisfy the
  third-order defect-cancellation assumption. With the same policy,
  `ellipse_scheme` and `ellipse_scheme_3rd_order` perform the same numerical
  update.
- `ellipse_scheme_4th_order` requires a branch-selected two-endpoint policy
  satisfying the coupled fourth-order cancellation assumption.

Policies depending on `h` are accepted by all three functions. Their
convergence order depends on that policy. In particular, allowing the scale to
approach zero or infinity with `h` falls outside the general bounded-selector
assumptions, so the third- or fourth-order function name alone is not a proof
of that order.

## API reference

::: specular.ode.ODEResult
    options:
      show_root_heading: true

::: specular.ode.ellipse_scheme
    options:
      show_root_heading: true

::: specular.ode.ellipse_scheme_3rd_order
    options:
      show_root_heading: true

::: specular.ode.ellipse_scheme_4th_order
    options:
      show_root_heading: true
