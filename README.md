# Specular Differentiation

[![PyPI version](https://badge.fury.io/py/specular-differentiation.svg)](https://badge.fury.io/py/specular-differentiation)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18246734.svg)](https://doi.org/10.5281/zenodo.18246734)
[![License](https://img.shields.io/pypi/l/specular-differentiation.svg)](https://pypi.org/project/specular-differentiation/)
[![CodeFactor](https://www.codefactor.io/repository/github/kyjung2357/specular-differentiation/badge)](https://www.codefactor.io/repository/github/kyjung2357/specular-differentiation)
[![CodeQL Advanced](https://github.com/kyjung2357/specular-differentiation/actions/workflows/codeql.yml/badge.svg)](https://github.com/kyjung2357/specular-differentiation/actions/workflows/codeql.yml)
[![Docs](https://img.shields.io/github/deployments/kyjung2357/specular-differentiation/github-pages?label=docs&logo=github)](https://kyjung2357.github.io/specular-differentiation)

The Python package `specular` implements *specular differentiation* which generalizes classical differentiation.
This implementation strictly follows the definitions, notations, and results in [[1]](#references) and [[2]](#references).

A specular derivative (the red line) can be understood as the average of the inclination angles of the right and left derivatives. 
In contrast, a symmetric derivative (the purple line) is the average of the right and left derivatives.
Their difference is illustrated as in the following figure.

![specular-derivative-animation](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/specular-derivative-animation.gif)

## Table of Contents
* [Introduction](#installation)
* [Applications](#applications)
* [Documentation](#documentation)
* [LaTeX macro](#latex-macro)
* [Citing specular-differentiation](#citing-specular-differentiation)
* [References](#references)

## Installation

### Requirements

`specular-differentiation` requires:

* **Python** >= 3.11
* `numpy` >= 2.0

Additional features are available through optional dependencies:

* `ode`: `matplotlib`, `pandas`, `tqdm`
* `optimization`: `matplotlib`, `tqdm`
* `numba`: `numba`
* `jax`: `jax`, `jaxlib`

### User installation

**Standard Installation (NumPy backend)**

```bash
pip install specular-differentiation
```

This installs the core specular differentiation API, including `A`, `derivative`,
`directional_derivative`, `partial_derivative`, `gradient`, and `jacobian`.

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

```bash
pip install "specular-differentiation[jax]"
```

See [the documentation](docs/api/jax.md) for JAX-specific usage.

**Developer Installation**

```bash
pip install -e ".[dev]"
```

### Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
import specular

ReLU = lambda x: max(x, 0)
print(specular.derivative(ReLU, x=0))
```

```text
0.41421356237309515
```

### Backend support

The package is organized around a backend system. 
NumPy is the default backend, while accelerated backends are optional and may require extra dependencies.

| Backend | Calculation | ODE | Optimization |
|:---:|:---:|:---:|:---:|
| NumPy | supported | supported  | supported |
| Numba | supported | supported (recommended) | Not supported |
| JAX | supported | supported | supported (recommended) |
| TensorFlow | supported | supported | not supported |
| PyTorch | supported | supported | not supported |


## Applications

Specular differentiation is defined in normed vector spaces, allowing for applications in higher-dimensional Euclidean spaces. 
The `specular` package includes the following applications.

### [Ordinary differential equation](docs/api/ode.md)

* **Directory**: `examples/ode/`
* **References**: [[1]](#references), [[3]](#references), [[4]](#references)

In [[1]](#references), seven schemes are proposed for solving ODEs numerically:

* the *specular Euler* scheme of Type 1~6
* the *specular trigonometric* scheme
* the *specular Huen* scheme

The following example shows that the specular Euler schemes of Type 5 and 6 yield more accurate numerical solutions than classical schemes: the explicit and implicit Euler schemes and the Crank-Nicolson scheme.

![ODE-example-1](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/ODE-example-1.png)

![ODE-example-2](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/ODE-example-2.png)

### [Optimization](docs/api/optimization.md)

* **Directory**: `examples/optimization/`
* **References**: [[2]](#references), [[5]](#references)

In [[2]](#references), three methods are proposed for optimizing nonsmooth convex objective functions:

* the *specular gradient (SPEG)* method
* the *stochastic specular gradient (S-SPEG)* method
* the *hybrid specular gradient (H-SPEG)* method

The following example compares the three proposed methods with the classical methods: [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) (GD), [Adaptive Moment Estimation](https://arxiv.org/abs/1412.6980) (Adam), and [Broyden-Fletcher-Goldfarb-Shanno](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) (BFGS).

![optimization-example](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/optimization-example.png)

## [Documentation](https://kyjung2357.github.io/specular-differentiation/)

### [Getting Started](https://kyjung2357.github.io/specular-differentiation/started/)
### [API Reference](https://kyjung2357.github.io/specular-differentiation/api/)
### [Examples](https://kyjung2357.github.io/specular-differentiation/examples/)

## LaTeX macro

To use the specular differentiation symbol in your LaTeX document, add the following code to your preamble (before `\begin{document}`):

```latex
% Required packages
\usepackage{graphicx}
\usepackage{bm}

% Definition of Specular Differentiation symbol
\newcommand\sd[1][.5]{\mathbin{\vcenter{\hbox{\scalebox{#1}{\,$\bm{\wedge}$}}}}}
```

### Usage examples 

Use the symbol in your document (after `\begin{document}`):

```latex
% A specular derivative in the one-dimensional Euclidean space
$f^{\sd}(x)$

% A specular directional derivative in normed vector spaces
$\partial^{\sd}_v f(x)$
```

## Citing specular-differentiation

To cite this repository:

```bibtex
@software{Jung_specular-differentiation_2026,
  author = {Jung, Kiyuob},
  doi = {10.5281/zenodo.18246734},
  license = {MIT},
  month = jan,
  title = {{specular-differentiation}},
  url = {https://github.com/kyjung2357/specular-differentiation},
  version = {1.1.0},
  year = {2026},
}
```

## References

[1] K. Jung. [*Nonlinear numerical schemes using specular differentiation for initial value problems of first-order ordinary differential equations*](https://arxiv.org/abs/2601.09900). arXiv preprint arXiv:2601.09900, 2026.

[2] K. Jung. [*Specular differentiation in normed vector spaces and its applications to nonsmooth convex optimization*](https://arxiv.org/abs/2601.10950). arXiv preprint arXiv:2601.10950, 2026. 

[3] K. Jung and J. Oh. [*The specular derivative*](https://arxiv.org/abs/2210.06062). arXiv preprint arXiv:2210.06062, 2022.

[4] K. Jung and J. Oh. [*The wave equation with specular derivatives*](https://arxiv.org/abs/2210.06933). arXiv preprint arXiv:2210.06933, 2022.

[5] K. Jung and J. Oh. [*Nonsmooth convex optimization using the specular gradient method with root-linear convergence*](https://arxiv.org/abs/2412.20747). arXiv preprint arXiv:2210.06933, 2024.