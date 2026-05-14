# 2.4. Backend

The package is organized around a backend system.
The standard installation uses the NumPy implementation, while accelerated
backends are optional and selected through `specular.change_backend(...)`.

## 2.4.1. Backend selection

```python
import specular

specular.backend_info()
specular.change_backend("cpu_jax")
```

Heavy machine-learning frameworks such as JAX, Numba, and PyTorch are not imported when `import specular` is executed. They are checked only when the user explicitly selects the corresponding backend.

## 2.4.2. Backend support

| Backend | Calculation | ODE | Optimization |
|:---:|:---:|:---:|:---:|
| NumPy | supported | supported  | supported (recommended) |
| Numba | supported | supported (recommended) | not supported |
| JAX | supported | supported | experimental  |
| PyTorch | experimental | experimental | not supported |

TensorFlow is not supported for the Python 3.14 target.

## 2.4.3. JAX backend

See the [official homepage](https://docs.jax.dev/en/latest/index.html) of JAX.

Requirement: objective functions should use `jax.numpy` instead of standard `numpy`.

The difference between numpy backend and JAX backend is in that how they calculate differentiation. 
The difference between the NumPy backend and the JAX backend lies in how they compute the one-sided derivatives. The NumPy backend approximates them from function values using finite differences, whereas the JAX backend computes them by applying automatic differentiation at shifted points.
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

## 2.4.4. API Reference

::: specular.backend
    handler: python
    options:
      show_root_heading: true
      show_source: true

---
::: specular.calculation
    handler: python
    options:
      show_root_heading: true
      show_source: true

---
::: specular.optimization.solver
    handler: python
    options:
      show_root_heading: true
      show_source: true

