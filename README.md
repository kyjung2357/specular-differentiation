# Specular Differentiation

[![PyPI version](https://badge.fury.io/py/specular-differentiation.svg)](https://badge.fury.io/py/specular-differentiation)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![License](https://img.shields.io/pypi/l/specular-differentiation.svg)](https://pypi.org/project/specular-differentiation/)

This repository contains the Python package `specular` which implements *Specular differentiation*, a generalization of classical differentiation.

A specular derivative can be understood as the average of the inclination angles of the right and left derivatives. 
In contrast, a symmetric derivative is the average of the right and left derivatives.
Their difference is illustrated as in the following figure.

![specular-derivative-animation](docs/figures/specular-derivative-animation.gif)

Also, `specular` includes the following applications:

* [**Nonsmooth convex optimization**](#nonsmooth-convex-optimization)
  * Directory: `examples/optimization/`.
  * Related reference: [[2]](#references), [[5]](#references).

* [**Initial value problems for ordinary differential equations**](#initial-value-problems-for-ordinary-differential-equations)
  * Directory: `examples/ode/`.
  * Related reference: [[1]](#references), [[3]](#references), [[4]](#references).

The package `specular` mainly follows the definitions, notations, and results presented in [[1]](#references) and [[2]](#references).

## Installation

### Dependencies

specular-differentiation requires:

* Python 3.11 or later
* NumPy 2.2.5 or later

### User installation

You can install the released version directly from PyPI:

```bash
pip install specular-differentiation
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

Detailed usage examples can be found in the [tutorial document](/docs/tutorial.md).

## Applications

Specular differentiation is defined in normed vector spaces, allowing for applications in higher-dimensional Euclidean spaces. 
Two applications are provided in this repository.

### Nonsmooth convex optimization

In [[2]](#references), three methods are proposed for for optimizing nonsmooth convex objective functions:

* *the specular gradient (SPEG)* method.
* *the stochastic specular gradient (SSPEG)* method.
* *the hybrid specular gradient (HSPEG)* method.

The following example compares the three proposed methods with the classical methods: [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) (GD), [Adaptive Moment Estimation](https://arxiv.org/abs/1412.6980) (Adam), and [Broyden-Fletcher-Goldfarb-Shanno](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) (BFGS).

![ODE-example](docs/figures/optimization-example.png)

### Initial value problems for ordinary differential equations

In [[1]](#references), *the specular Euler scheme of Type 5* is introduced for solving ODEs numerically, yielding more accurate numerical solutions than classical schemes: the explicit and implicit Euler schemes and the Crank-Nicolson scheme.

![ODE-example](docs/figures/ODE-example.png)


## LaTeX notation

To use the symbol for specular differentiation in your LaTeX document, please refer to the following instructions.

### Setup 

Add the following code to your LaTeX preamble (before `\begin{document}`):

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

[1] K. Jung. *Nonlinear numerical schemes using specular differentiation for initial value problems of first-order ordinary differential equations*. arXiv preprint arXiv:??, 2025.

[2] K. Jung. *Specular differentiation in normed vector spaces and its applications to nonsmooth convex optimization*. arXiv preprint arXiv:??, 2025. 

[3] K. Jung and J. Oh. [*The specular derivative*](https://arxiv.org/abs/2210.06062). arXiv preprint arXiv:2210.06062, 2022.

[4] K. Jung and J. Oh. [*The wave equation with specular derivatives*](https://arxiv.org/abs/2210.06933). arXiv preprint arXiv:2210.06933, 2022.

[5] K. Jung and J. Oh. [*Nonsmooth convex optimization using the specular gradient method with root-linear convergence*](https://arxiv.org/abs/2412.20747). arXiv preprint arXiv:2210.06933, 2024.