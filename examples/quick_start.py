"""Run the examples shown on the Quick start documentation page."""

# --8<-- [start:version]
import specular

print("specular version:", specular.__version__)
# --8<-- [end:version]


# --8<-- [start:derivative]
def relu(x: float) -> float:
    """Return the rectified linear unit."""
    return max(x, 0.0)


value = specular.derivative(relu, x=0.0)
print("ReLU derivative at 0:", value)
# --8<-- [end:derivative]


# --8<-- [start:optimization]
result = specular.specular_gradient(
    abs,
    initial_point=1.0,
    step_size="square_summable_not_summable",
    a=0.5,
    b=1.0,
    max_iter=200,
)
print(result.solution, result.func_val, result.stop_reason)
# --8<-- [end:optimization]


# --8<-- [start:ellipse]
def F(t, u):
    return -u


result = specular.ellipse_scheme(
    F,
    0.0,
    1.0,
    1.0,
    n_steps=100,
    sigma_n=1.0,
)
print(result.t[-1], result.u[-1])
# --8<-- [end:ellipse]


# --8<-- [start:backend-status]
available = specular.available_backends()
print("current backend:", specular.get_backend())
print("available backends:", available)
# --8<-- [end:backend-status]

if "numba" not in available:
    raise SystemExit(
        'Install "specular-differentiation[numba]" for the Numba examples.'
    )


# --8<-- [start:persistent-backend]
specular.set_backend("numba")
try:
    print("selected backend:", specular.get_backend())
    print("ReLU derivative at 0:", specular.derivative(relu, x=0.0))
finally:
    specular.set_backend("numpy")

print("restored backend:", specular.get_backend())
# --8<-- [end:persistent-backend]


# --8<-- [start:temporary-backend]
with specular.use_backend("numba"):
    print("inside context:", specular.get_backend())
    print("ReLU derivative at 0:", specular.derivative(relu, x=0.0))

print("after context:", specular.get_backend())
# --8<-- [end:temporary-backend]
