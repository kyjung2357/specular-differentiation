# Specular Differentiation

`specular-differentiation` is a Python package for computing specular derivatives,
directional derivatives, gradients, Jacobians, ODE solvers, and optimization methods.

## Installation

```bash
pip install specular-differentiation
```

Requirements:

* **Python** >= 3.14
* `numpy` >= 2.4

Optional features:

```bash
pip install "specular-differentiation[ode]"             # ODE solvers
pip install "specular-differentiation[optimization]"    # optimization routines
pip install "specular-differentiation[numba]"           # Numba backend
pip install "specular-differentiation[jax]"             # JAX backend
pip install "specular-differentiation[torch]"           # PyTorch backend
```

## Documentation

* [Getting started](started.md)
* [API reference](api/index.md)
* [Examples](examples/index.md)

## Backend support

| Backend | Calculation | ODE | Optimization |
|:---:|:---:|:---:|:---:|
| NumPy | supported | supported | supported |
| Numba | supported | supported | not supported |
| JAX | supported | supported | experimental |
| PyTorch | experimental | experimental | not supported |

TensorFlow is not supported for the Python 3.14 target.

## Project Information

For the full project overview, citation information, figures, and references, see the
[GitHub README](https://github.com/kyjung2357/specular-differentiation#readme).
