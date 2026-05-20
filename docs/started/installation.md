# User installation

## Standard Installation

The package is available on PyPI:

```bash
pip install specular-differentiation
```

Requirements:

* **Python** >= 3.14
* `numpy` >= 2.4

The package is distributed on PyPI as `specular-differentiation` and imported in Python as `specular`.
Check the version:

```python
import specular

print("version: ", specular.__version__)
```

```text
version:  1.3.0
```

## Optional features

Additional features are provided as optional extras:

```bash
pip install "specular-differentiation[ode]"             # ODE solvers
pip install "specular-differentiation[optimization]"    # optimization routines
pip install "specular-differentiation[numba]"           # Numba backend
pip install "specular-differentiation[jax]"             # JAX backend
pip install "specular-differentiation[torch]"           # PyTorch backend
```

This adds the following dependencies:

* **[Numba](https://numba.pydata.org/)** (`numba` >= 0.65)
* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.10)
* **[PyTorch](https://pytorch.org/)** (`torch` >= 2.12)

## Developer installation

To install all dependencies including tests, docs, and examples.

```bash
pip install -e ".[dev]"
```

This adds the following dependencies:

* **[SciPy](https://scipy.org/)** (`scipy` >= 1.17)
* **[Pytest](https://docs.pytest.org/en/stable/)** (`pytest` >= 9.0)

## Backend support

By default, specular-differentiation uses the NumPy backend (CPU).
To enable hardware acceleration, you can install the package with different backends.

| Backend | Calculation | ODE | Optimization |
|:---:|:---:|:---:|:---:|
| NumPy | supported | supported | supported |
| Numba | supported | supported | not supported |
| JAX | supported | supported | experimental |
| PyTorch | experimental | experimental | not supported |

> [!NOTE]
> TensorFlow is not supported for the Python 3.14 target.
