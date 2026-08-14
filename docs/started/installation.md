# User installation

## Standard Installation

The package is available on PyPI:

```bash
pip install specular-differentiation
```

Requirements:

* **Python** >= 3.14
* `numpy` >= 2.4

The package is distributed on PyPI as `specular-differentiation` and imported
in Python as `specular`.

## Optional features

Additional features are provided as optional extras:

```bash
pip install "specular-differentiation[numba]"           # Numba backend
pip install "specular-differentiation[jax]"             # JAX backend
```

This adds the following dependencies:

* **[Numba](https://numba.pydata.org/)** (`numba` >= 0.65)
* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.10)

## Developer installation

To install the optional backends and development dependencies for tests:

```bash
pip install -e ".[dev]"
```

The `dev` extra also installs SciPy, Pytest, and IPython.

Documentation tools are available separately with `pip install -e ".[docs]"`.

## Backend support

By default, specular-differentiation uses the NumPy backend. Installing an
optional dependency does not change the backend automatically.

| Backend | Calculation |
|:---:|:---:|
| NumPy | supported |
| Numba | supported |
| JAX | supported |

Select an installed backend explicitly:

```python
import specular

specular.set_backend("numba")
```
