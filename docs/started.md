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
# Output: version:  1.0.0
```

**Advanced Installation (JAX backend)**

By default, the package uses the NumPy backend (CPU). 
To enable hardware acceleration, you can install the package with the JAX backend (GPU/TPU). 
This adds the following dependencies:

* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.4):

```bash
pip install "specular-differentiation[jax]"
```

> [!NOTE]
> This feature is experimental for now. See [Notes](api/jax.md).

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

## 1.2. Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
import specular

ReLU = lambda x: max(x, 0)
specular.derivative(ReLU, x=0)
# Output: 0.41421356237309515
```

## 1.3. JAX Backend Usage

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

