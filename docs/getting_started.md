# Getting Started

## User installation

**Standard Installation (NumPy backend)**

The package is available on PyPI:

```bash
pip install specular-differentiation
```

Check the version:

```python
import specular

print("version: ", specular.__version__)
# Output: version:  1.0.0
```

**Advanced Installation (JAX backend)**

By default, the package uses the NumPy backend (CPU). 
To enable hardware acceleration, you can install the package with the JAX backend (GPU/TPU). 
This adds the following dependencies:

* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.4):

> [!NOTE]
> This feature is experimental for now. See [Notes](/docs/api_reference/jax.md).

```bash
pip install "specular-differentiation[jax]"
```

**Developer installation**

To install all dependencies including tests, docs, and examples.
This adds the following dependencies:

* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.4):
* **[SciPy](https://scipy.org/)** (`scipy` >= 1.10.0)
* **[PyTorch](https://pytorch.org/)** (`torch` >= 2.0.0)
* **[Pytest](https://docs.pytest.org/en/stable/)** (`pytest` >= 7.0)

```bash
pip install -e ".[dev]"
```

## Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
import specular

ReLU = lambda x: max(x, 0)
specular.derivative(ReLU, x=0)
# Output: 0.41421356237309515
```

## JAX Backend Usage

To leverage **JAX** for hardware acceleration instead of the standard NumPy backend, import `specular.jax`:

```python
import specular.jax as sjax

ReLU = lambda x: jax.numpy.maximum(x, 0)
sjax.derivative(ReLU, 0.0)
# Output: Array(0.41421354, dtype=float32)
```

To enable 64-bit precision (double precision), update the **JAX** configuration as follows:

```python
import jax
jax.config.update("jax_enable_x64", True)

import specular.jax as sjax

ReLU = lambda x: jax.numpy.maximum(x, 0)
sjax.derivative(ReLU, 0.0)
# Output: Array(0.41421356, dtype=float64)
```

## Continue reading

### [API Reference](./api_reference/README.md)

* [1. Calculation](./api_reference/calculation.md)
* [2. ODE](./api_reference/ode.md)
* [3. Optimization](./api_reference/optimization.md)
* [4. JAX backend](./api_reference/jax.md)

### [Examples](/examples/README.md)

* [2026-Jung](/examples/ode/2026-Jung/)
* [2024-Jung-Oh](/examples/optimization/2024-Jung-Oh/)
* [2026-Jung](/examples/optimization/2026-Jung/)

