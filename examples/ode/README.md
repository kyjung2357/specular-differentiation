# Scalar ODE examples

These scripts reproduce the four numerical themes reported in the manuscript:

- `ellipse_exactness.py`: exact tracing after fitting the scale to an ellipse;
- `inverse_equation_small_scale.py`: mesh-dependent scales for `u' = 1/u`;
- `pendulum_third_order.py`: third-order pointwise cancellation on normalized
  pendulum branches;
- `pendulum_fourth_order.py`: fourth-order defect balance on normalized
  pendulum branches.

Run them from the repository root:

```console
python examples/ode/ellipse_exactness.py
python examples/ode/inverse_equation_small_scale.py
python examples/ode/pendulum_third_order.py
python examples/ode/pendulum_fourth_order.py
```

The examples use only NumPy and the public `specular` API. They print compact
tables and do not create data or figure files.

The selected third- and fourth-order scales enforce the corresponding defect
conditions. Their observed orders are conditional on the regularity and
boundedness of the selected scale branch; they are not guarantees for an
arbitrary scalar field.

The fourth-order pendulum example uses `[0, 0.8]`, as in the manuscript. Its
first step is case E5(i), and subsequent steps follow the E5(ii) branch in the
current theorem statement.
