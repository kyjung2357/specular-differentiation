# Installation

## Requirements

Specular Differentiation requires **Python 3.14 or later** and **NumPy 2.4 or
later**. NumPy is installed automatically with the package and is the default
backend.

## Install from PyPI

Install Specular Differentiation using pip:

```bash
pip install specular-differentiation
```

The package is distributed as `specular-differentiation` and imported in
Python as `specular`.

The standard installation includes `scaled_mean`, `derivative`, `gradient`,
and `jacobian`, as well as the scalar ODE methods and the SPEG optimizer.

## Verify the installation

Import `specular` and check the installed package version:

```python
import specular

print("specular version:", specular.__version__)
```

## Optional backends

Install the corresponding extra to use Numba or JAX:

```bash
pip install "specular-differentiation[numba]"
pip install "specular-differentiation[jax]"
```

The Numba extra requires `numba >= 0.65`. The JAX extra requires
`jax >= 0.10` and `jaxlib >= 0.10`. pip installs these dependencies
automatically.

Installing an extra makes that backend available; NumPy remains the default.
See [Backends](backends.md) for backend selection and precision settings.

## Development and documentation

For development, run the following command from the root of a local checkout
of the [repository](https://github.com/kyjung2357/specular-differentiation):

```bash
pip install -e ".[dev]"
```

The editable installation uses the local source and includes both optional
backends, the test dependencies, and the example tools.

Documentation tools can be installed separately from the same directory:

```bash
pip install -e ".[docs]"
```
