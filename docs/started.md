# 1. Getting Started

## 1.1. User installation

**Standard Installation (NumPy backend)**

The package is available on PyPI:

```bash
pip install specular-differentiation
```

Check the version:

```python
import specular

print("version: ", specular.__version__)
```

```text
version:  1.1.0
```

**ODE solvers**

```bash
pip install "specular-differentiation[ode]"
```

**Optimization routines**

```bash
pip install "specular-differentiation[optimization]"
```

**Numba backend**

```bash
pip install "specular-differentiation[numba]"
```

If Numba is installed and available, the package may use the Numba-accelerated CPU backend.

**JAX backend**

By default, the package uses the NumPy backend (CPU). 
To enable hardware acceleration, you can install the package with the JAX backend (GPU/TPU). 
This adds the following dependencies:

* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.4):

```bash
pip install "specular-differentiation[jax]"
```

> [!NOTE]
> This feature is experimental for now. See [2.4 Backend](api/backend.md).

**Developer installation**

To install all dependencies including tests, docs, and examples.
This adds the following dependencies:

* optional extras: `ode`, `optimization`, `numba`, and `jax`
* **[SciPy](https://scipy.org/)** (`scipy` >= 1.10.0)
* **[PyTorch](https://pytorch.org/)** (`torch` >= 2.0.0)
* **[Pytest](https://docs.pytest.org/en/stable/)** (`pytest` >= 7.0)

```bash
pip install -e ".[dev]"
```

## 1.2. Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
import specular

ReLU = lambda x: max(x, 0)
specular.derivative(ReLU, x=0)
```

```text
0.41421356237309515
```

## 1.3. Backend Usage

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
