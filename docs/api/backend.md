# Backend

The package is organized around a backend system.
The standard installation uses the NumPy implementation, while accelerated
backends are optional and selected through `specular.change_backend(...)`.

## Backend selection

```python
import specular

specular.backend_info()
specular.change_backend("cpu_jax")
```

Optional backend dependencies such as JAX, Numba, and PyTorch are not imported when `import specular` is executed. They are checked when the user explicitly selects the corresponding backend or calls `specular.backend_info()`.

## Backend support

| Backend | Calculation | ODE | Optimization |
|:---:|:---:|:---:|:---:|
| NumPy | supported | supported  | supported (recommended) |
| Numba | supported | supported (recommended) | not supported |
| JAX | supported | supported | experimental  |
| PyTorch | experimental | experimental | not supported |

TensorFlow is not supported for the Python 3.14 target.

## Numba backend

The Numba backend is loaded only after `specular.change_backend("cpu_numba")` and accelerates the NumPy-style finite-difference implementation when available.

```python
import specular

specular.change_backend("cpu_numba")
```

## JAX backend

To use the **JAX** backend, install the JAX extra and select the backend explicitly:

```python
import jax.numpy as jnp
import specular

specular.change_backend("cpu_jax")

ReLU = lambda x: jnp.maximum(x, 0)
specular.derivative(ReLU, 0.0)
```

```text
Array(0.41421354, dtype=float32)
```

To enable 64-bit precision (double precision), update the **JAX** configuration as follows:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import specular

specular.change_backend("cpu_jax")

ReLU = lambda x: jnp.maximum(x, 0)
specular.derivative(ReLU, 0.0)
```

```text
Array(0.41421356, dtype=float64)
```

The JAX backend is not a bitwise-equivalent implementation of the NumPy backend:
it uses automatic differentiation at shifted points, while NumPy/Numba use
one-sided finite differences. This distinction is intentional.

See the [official homepage](https://docs.jax.dev/en/latest/index.html) of JAX.

Requirement: objective functions should use `jax.numpy` instead of standard `numpy`.

The difference between the NumPy backend and the JAX backend lies in how they compute the one-sided derivatives.
The NumPy and Numba backends approximate them from function values using finite differences, whereas the JAX backend computes them by applying automatic differentiation at shifted points.
Then, they use the function `A` to complete the calculation of specular differentiation.

```python
import jax.numpy as jnp
import specular

specular.change_backend("cpu_jax")

ReLU = lambda x: jnp.maximum(x, 0)
specular.derivative(ReLU, 0.0)
```

To enable 64-bit precision, update the JAX configuration before defining the functions that will be evaluated:

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import specular

specular.change_backend("cpu_jax")

ReLU = lambda x: jnp.maximum(x, 0)
specular.derivative(ReLU, 0.0)
```

For a detailed comparison of the algorithms, see:

* [`examples/optimization/jax/main.py`](https://github.com/kyjung2357/specular-differentiation/blob/main/examples/optimization/jax/main.py): A basic implementation using the JAX backend.
* [`examples/optimization/2026-Jung/main_jax.py`](https://github.com/kyjung2357/specular-differentiation/blob/main/examples/optimization/2026-Jung/main_jax.py): The JAX version of the optimization experiment.

## API Reference

- [`specular.backend`](backend/specular-backend.md)
