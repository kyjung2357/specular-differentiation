# Specular Ellipse method

Here, SE denotes the specular ellipse method. In the numerical-results
notation, SE2 uses `sigma_n = 1`, SE3 uses either a pointwise cancelling scale
or a problem-specific vanishing-scale family, and SE4 uses a two-point
defect-balancing scale. These order labels are unrelated to the Type 1, Type 2,
and Type 5 names of the unscaled specular Euler methods.

The scalar ODE examples include:

- `ellipse_exactness.py`: exact tracing after fitting the scale to an ellipse;
- `quadratic_decay_defect_cancellation.py`: the defect quantities cancelled by SE3 and SE4 for `u' = -u^2`;
- `quadratic_decay_convergence.py`: CN, SE2, SE3, SE4, RK3, and RK4 compared for `u' = -u^2`;
- `autonomous_large_scale.py`: fixed and diverging SE scales compared with CN on one-step E3b1 pairs for `u' = 1-u^2`;
- `inverse_equation_small_scale.py`: mesh-dependent scales for `u' = 1/u`;
- `pendulum_fourth_order.py`: SE2, SE3, and SE4 compared with RK4 on normalized pendulum branches.

Run them from the repository root:

```console
python examples/ode/ellipse_exactness.py
python examples/ode/quadratic_decay_defect_cancellation.py
python examples/ode/quadratic_decay_convergence.py
python examples/ode/autonomous_large_scale.py
python examples/ode/inverse_equation_small_scale.py
python examples/ode/pendulum_fourth_order.py
```

The examples use NumPy and the public `specular` API.
They also use Matplotlib from the `dev` extra, print compact numerical
summaries, and save PDFs under `figures/`.

The pointwise and two-point selectors enforce their corresponding finite-scale
defect conditions where such a scale exists. The vanishing- and diverging-scale
experiments instead study boundary regimes. All observed orders are conditional
on the problem and on the hypotheses of the applicable result; they are not
guarantees for an arbitrary scalar field.

The SE4 pendulum example uses `[0, 0.8]`, as in the manuscript.
For the reported `h = 1e-2` calculation, its first accepted pair is in case
E5a and the subsequent accepted pairs are in case E5b. This accepted-path
check does not verify the theorem's uniform-neighborhood hypothesis.
