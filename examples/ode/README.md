# Scalar ODE examples

These scripts cover seven numerical themes associated with the manuscript:

- `ellipse_exactness.py`: exact tracing after fitting the scale to an ellipse;
- `defect_local_error.py`: the curvature-defect profile and its cancelling
  scale for `u' = -u^2`;
- `quadratic_decay_third_order.py`: a third-order state-dependent scale for
  `u' = -u^2`;
- `quadratic_decay_fourth_order.py`: fixed, pointwise third-order, and
  defect-balanced fourth-order SE scales compared with RK4 for `u' = -u^2`;
- `autonomous_large_scale.py`: fixed and diverging SE scales compared with CN
  on one-step E3b1 pairs for `u' = 1-u^2`;
- `inverse_equation_small_scale.py`: mesh-dependent scales for `u' = 1/u`;
- `pendulum_fourth_order.py`: fixed, pointwise third-order, and defect-balanced
  fourth-order SE scales compared with RK4 on normalized pendulum branches.

Run them from the repository root:

```console
python examples/ode/ellipse_exactness.py
python examples/ode/defect_local_error.py
python examples/ode/quadratic_decay_third_order.py
python examples/ode/quadratic_decay_fourth_order.py
python examples/ode/autonomous_large_scale.py
python examples/ode/inverse_equation_small_scale.py
python examples/ode/pendulum_fourth_order.py
```

The examples use NumPy and the public `specular` API. They also use Matplotlib
from the `dev` extra, print compact numerical summaries, open comparison
figures, and save PDFs under `figures/`.

The selected third- and fourth-order scales enforce the corresponding defect
conditions. Their observed orders are conditional on the regularity and
boundedness of the selected scale branch; they are not guarantees for an
arbitrary scalar field.

The fourth-order pendulum example uses `[0, 0.8]`, as in the manuscript. Its
first step is case E5(i), and subsequent steps follow the E5(ii) branch in the
current theorem statement.
