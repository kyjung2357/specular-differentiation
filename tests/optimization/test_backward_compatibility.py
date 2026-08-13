import numpy as np

import specular
from specular.optimization import StepSize, gradient_method


def quadratic(x):
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def test_legacy_step_size_constructor_and_top_level_export():
    step = specular.StepSize("square_summable_not_summable", [1.0, 2.0])

    assert isinstance(step, StepSize)
    assert step(1) == 1.0 / 3.0


def test_legacy_gradient_method_is_specular_and_history_is_callable():
    result = gradient_method(
        f=quadratic,
        x_0=np.array([1.0, 1.0]),
        step_size=StepSize("constant", 0.1),
        form="specular gradient",
        max_iter=5,
        print_bar=False,
    )

    variables, values, runtime = result.history()

    assert result.method == "specular_gradient"
    assert len(variables) == len(values)
    assert runtime == result.runtime


def test_stochastic_zero_component_does_not_report_convergence(monkeypatch):
    monkeypatch.setattr(np.random, "randint", lambda _: 1)

    result = gradient_method(
        f=quadratic,
        x_0=np.array([1.0, 0.0]),
        step_size=StepSize("constant", 0.1),
        form="stochastic",
        f_j=[lambda x: float(x[0] ** 2), lambda x: float(x[1] ** 2)],
        max_iter=1,
        print_bar=False,
    )

    assert result.stop_reason != "gradient norm below tolerance"
