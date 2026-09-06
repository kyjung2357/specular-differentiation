# Quick start

- [Import Specular Differentiation](#import-specular-differentiation)
    - [Compute a specular derivative](#compute-a-specular-derivative)
- [Optimization](#optimization)
- [ODE](#ode)
- [Backends](#backends)

## Import Specular Differentiation

Import `specular` and check the installed package version.

```python
import specular

print("specular version:", specular.__version__)
```

### Compute a specular derivative

Define the ReLU function and evaluate its specular derivative at the kink `x = 0`.
The result is $\sqrt{2}-1$, approximately `0.41421356237309503`.

```python
import specular


def relu(x):
    return max(x, 0.0)


value = specular.derivative(relu, x=0.0)
print(value)
```

## Optimization

SPEG combines the normalized negative specular gradient with a step size rule:

```python
import specular

result = specular.specular_gradient(
    abs,
    initial_point=1.0,
    step_size="square_summable_not_summable",
    a=0.5,
    b=1.0,
    max_iter=200,
)
print(result.solution, result.func_val, result.stop_reason)
```

The optimization API separates direction rules, step size rules, and the
iteration wrapper. It supports decreasing schedules, user-defined rules,
Armijo and Wolfe searches, and bounded numerical line minimization. The
[Optimization API](api/optimization.md) describes their options and the
distinction between classical and specular line-search derivatives.

## ODE

The specular ellipse method solves scalar ordinary differential equations.
For example, solve $u'(t)=-u(t)$ with $u(0)=1$ on $[0,1]$:

```python
import specular


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
```

The result contains the time points in `t` and the numerical solution in `u`.
The [ODE API](api/ode.md#prescribed-scale) describes the specular ellipse
method's scale parameter and options.

## Backends

`get_backend()` returns the backend selected for the current execution
context. `available_backends()` returns the installed backends that can be
selected. NumPy is the default.

```python
import specular

print("current backend:", specular.get_backend())
print("available backends:", specular.available_backends())
```

NumPy is included in the standard installation. To run the optional Numba
example below, install the Numba extra:

```bash
pip install "specular-differentiation[numba]"
```

`use_backend()` limits the selection to a `with` block and restores the
previous backend automatically when the block ends. Using `relu` defined
above:

```python
with specular.use_backend("numba"):
    print(specular.derivative(relu, x=0.0))

print("restored backend:", specular.get_backend())
```

To keep using a backend in the current context, call
`specular.set_backend("numba")`. The [Backend API](api/backend.md) covers
persistent selection, JAX support, and precision settings.
