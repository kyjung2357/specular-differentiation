# Specular Ellipse method

Here, SE denotes the specular ellipse method.
SE2 uses `sigma_n = 1`, SE3 uses the pointwise third-order scale, and SE4 uses the two-point fourth-order scale.

The scalar ODE examples include:

- `ellipse_exactness.py`: exact tracing after fitting the scale to an ellipse;
- `quadratic_decay_defect_cancellation.py`: the defect quantities cancelled by SE3 and SE4 for `u' = -u^2`;
- `quadratic_decay_convergence.py`: CN, SE2, SE3, SE4, RK3, and RK4 compared for `u' = -u^2`;
- `autonomous_large_scale.py`: fixed and diverging SE scales compared with CN on one-step E3b1 pairs for `u' = 1-u^2`;
- `inverse_equation_small_scale.py`: mesh-dependent scales for `u' = 1/u`;
- `pendulum_fourth_order.py`: SE2, SE3, and SE4 compared with RK4 on normalized pendulum branches.
- `pendulum_fourth_order_diagnostics.py`: accepted-step scale, discriminant, defect-residual, and classification diagnostics for the SE4 pendulum calculation.

Run them from the repository root:

```console
python examples/ode/ellipse_exactness.py
python examples/ode/quadratic_decay_defect_cancellation.py
python examples/ode/quadratic_decay_convergence.py
python examples/ode/autonomous_large_scale.py
python examples/ode/inverse_equation_small_scale.py
python examples/ode/pendulum_fourth_order.py
python examples/ode/pendulum_fourth_order_diagnostics.py
```

The examples use NumPy and the public `specular` API.
They also use Matplotlib from the `dev` extra, print compact numerical summaries, open comparison figures, and save PDFs under `figures/`.

The scales used by SE3 and SE4 enforce the corresponding defect conditions.
Their observed orders are conditional on the regularity and boundedness of the selected scale branch; they are not guarantees for an arbitrary scalar field.

The SE4 pendulum example uses `[0, 0.8]`, as in the manuscript.
Its first accepted pair is in case E5a, and subsequent accepted pairs are in case E5b in the current theorem statement.
