"""Run the examples shown on the Optimization API page."""

import numpy as np

import specular
from specular.optimization import make_direction, make_step_size, minimize


# --8<-- [start:scalar]
scalar_result = specular.specular_gradient(
    abs,
    initial_point=1.0,
    step_size="square_summable_not_summable",
    a=0.5,
    b=1.0,
    max_iter=200,
)
print("Scalar SPEG:", scalar_result.solution, scalar_result.stop_reason)
# --8<-- [end:scalar]


# --8<-- [start:vector]
def sum_abs(x):
    return np.sum(np.abs(x))


vector_result = specular.specular_gradient(
    sum_abs,
    initial_point=[1.0, -2.0],
    step_size="square_summable_not_summable",
    a=0.5,
    b=1.0,
    max_iter=500,
)
print("Vector SPEG:", vector_result.solution, vector_result.func_val)
# --8<-- [end:vector]


# --8<-- [start:composition]
def quadratic(x):
    return np.dot(x, x)


def quadratic_gradient(x):
    return 2.0 * x


direction = make_direction("speg", gradient=quadratic_gradient)
step = make_step_size(
    "strong_Wolfe", f=quadratic, gradient=quadratic_gradient
)
composed_result = minimize(
    quadratic,
    initial_point=[1.0, -2.0],
    direction=direction,
    step_size=step,
    gradient=quadratic_gradient,
    max_iter=100,
)
print("Composed SPEG:", composed_result.solution, composed_result.stop_reason)
# --8<-- [end:composition]


# --8<-- [start:custom]
def descent(n, x):
    return -2.0 * x


schedule = make_step_size(lambda n: 0.1)
custom_result = minimize(
    quadratic,
    initial_point=[1.0, -2.0],
    direction=descent,
    step_size=schedule,
    gradient=quadratic_gradient,
    max_iter=100,
    record_history=False,
)
print("Custom rules:", custom_result.solution, custom_result.stop_reason)
# --8<-- [end:custom]
