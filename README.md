# Specular Differentiation

[![PyPI version](https://badge.fury.io/py/specular-differentiation.svg)](https://badge.fury.io/py/specular-differentiation)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)
[![License](https://img.shields.io/pypi/l/specular-differentiation.svg)](https://pypi.org/project/specular-differentiation/)
[![CodeFactor](https://www.codefactor.io/repository/github/kyjung2357/specular-differentiation/badge)](https://www.codefactor.io/repository/github/kyjung2357/specular-differentiation)
[![CodeQL Advanced](https://github.com/kyjung2357/specular-differentiation/actions/workflows/codeql.yml/badge.svg)](https://github.com/kyjung2357/specular-differentiation/actions/workflows/codeql.yml)

The Python package `specular` implements *specular differentiation* which generalizes classical differentiation.
This implementation strictly follows the definitions, notations, and results in [[1]](#references) and [[2]](#references).

A specular derivative (the red line) can be understood as the average of the inclination angles of the right and left derivatives. 
In contrast, a symmetric derivative (the purple line) is the average of the right and left derivatives.
Their difference is illustrated as in the following figure.

![specular-derivative-animation](docs/figures/specular-derivative-animation.gif)

Also, `specular` includes the following applications:

* [**Ordinary differential equation**](examples/README.md#ordinary-differential-equation)
  * **Directory**: `examples/ode/`
  * **References**: [[1]](#references), [[3]](#references), [[4]](#references)

* [**Optimization**](examples/README.md#optimization)
  * **Directory**: `examples/optimization/`
  * **References**: [[2]](#references), [[5]](#references)

## Installation

### Requirements

`specular-differentiation` requires:

* **Python** >= 3.11
* `ipython` >= 8.12.3
* `matplotlib` >= 3.10.8
* `numpy` >= 2.4.0
* `pandas` >= 2.3.3
* `tqdm` >= 4.67.1

### User installation

**Standard Installation (NumPy backend)**

The package is available on PyPI:

```bash
pip install specular-differentiation
```

**Advanced Installation (JAX backend)**

By default, the package uses the NumPy backend (CPU). 
To enable hardware acceleration, you can install the package with the JAX backend (GPU/TPU). 
This adds the following dependencies:

* **[JAX](https://docs.jax.dev/en/latest/index.html)** (`jax`, `jaxlib` >= 0.4):

> [!IMPORTANT]
> This feature is experimental for now. See [Notes](docs/notes_for_jax.ipynb).

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

### Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
>>> import specular
>>> 
>>> ReLU = lambda x: max(x, 0)
>>> specular.derivative(ReLU, x=0)
0.41421356237309515
```

### Tutorial

Detailed usage examples can be found in [documentation](docs/).

## Applications

Specular differentiation is defined in normed vector spaces, allowing for applications in higher-dimensional Euclidean spaces. 
Two applications are provided in this repository.

### [Ordinary differential equation](examples/README.md#ordinary-differential-equation)

In [[1]](#references), seven schemes are proposed for solving ODEs numerically:

* the *specular Euler* scheme of Type 1~6
* the *specular trigonometric* scheme

The following example shows that the specular Euler schemes of Type 5 and 6 yield more accurate numerical solutions than classical schemes: the explicit and implicit Euler schemes and the Crank-Nicolson scheme.

![ODE-example](docs/figures/ODE-example.png)

### [Optimization](examples/README.md#optimization)

In [[2]](#references), three methods are proposed for optimizing nonsmooth convex objective functions:

* the *specular gradient (SPEG)* method
* the *stochastic specular gradient (S-SPEG)* method
* the *hybrid specular gradient (H-SPEG)* method

The following example compares the three proposed methods with the classical methods: [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) (GD), [Adaptive Moment Estimation](https://arxiv.org/abs/1412.6980) (Adam), and [Broyden-Fletcher-Goldfarb-Shanno](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) (BFGS).

![optimization-example](docs/figures/optimization-example.png)

## LaTeX symbol

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

## References

[1] K. Jung. *Nonlinear numerical schemes using specular differentiation for initial value problems of first-order ordinary differential equations*. arXiv preprint arXiv:????.?????, TBA.

[2] K. Jung. *Specular differentiation in normed vector spaces and its applications to nonsmooth convex optimization*. arXiv preprint arXiv:????.?????, TBA. 

[3] K. Jung and J. Oh. [*The specular derivative*](https://arxiv.org/abs/2210.06062). arXiv preprint arXiv:2210.06062, 2022.

[4] K. Jung and J. Oh. [*The wave equation with specular derivatives*](https://arxiv.org/abs/2210.06933). arXiv preprint arXiv:2210.06933, 2022.

[5] K. Jung and J. Oh. [*Nonsmooth convex optimization using the specular gradient method with root-linear convergence*](https://arxiv.org/abs/2412.20747). arXiv preprint arXiv:2210.06933, 2024.