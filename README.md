# Specular Differentiation

[![PyPI version](https://badge.fury.io/py/specular-differentiation.svg)](https://badge.fury.io/py/specular-differentiation)
[![License](https://img.shields.io/pypi/l/specular-differentiation.svg)](https://pypi.org/project/specular-differentiation/)

This repository contains the Python package `specular_diff` and codes for applications:

* [**Nonsmooth convex optimization**](#nonsmooth-convex-optimization)
  * Directory: `examples/nonsmooth-convex-opt/`
  * Related reference: [[2]](#references), [[5]](#references)

* [**Initial value problems for ordinary differential equations**](#initial-value-problems-for-ordinary-differential-equations)
  * Directory: `examples/numerical-ODE/`
  * Related reference: [[1]](#references), [[3]](#references), [[4]](#references)


## Introduction

*Specular differentiation* generalizes classical differentiation.
A specular derivative can be understood as the average of the inclination angles of the right and left derivatives. 
In contrast, a symmetric derivative is the average of the right and left derivatives.
Their difference is illustrated as in the following figure.

![specular-derivative-animation](figures/specular_derivative_animation.gif)

## Installation

### Dependencies

specular-differentiation requires:

* Python 3.10 or later
* NumPy 2.2.5 or later

### User installation

You can install the released version directly from PyPI:

```bash
pip install specular-differentiation
```

### Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
>>> import specular_diff as sd
>>> 
>>> ReLU = lambda x: max(x, 0)
>>> sd.specular_derivative(ReLU, x=0)
0.41421356237309515
```

### Tutorial

Detailed usage examples can be found in the [tutorial document](/docs/tutorial.md).

## Applications

Specular differentiation is defined in normed vector spaces, allowing for applications in higher-dimensional Euclidean spaces. 
Two applications are provided in this repository.

### Nonsmooth convex optimization

In [[2]](#references), *the specular gradient method* is introduced for optimizing nonsmooth convex objective functions.

### Initial value problems for ordinary differential equations

In [[1]](#references), *the specular Euler scheme of Type 5* is introduced for solving ODEs numerically, yielding more accurate numerical solutions than classical schemes: the explicit and implicit Euler schemes and the Crank-Nicolson scheme.

![ODE-numerical-example](figures/ODE_Numerical_example.png)


## LaTeX notation

To use the specular differentiation symbol in your LaTeX document, please refer to the following instructions.

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

[3] K. Jung and J. Oh. [*The specular derivative*](https://arxiv.org/abs/2210.06062). *arXiv preprint arXiv:2210.06062*, 2022.

[4] K. Jung and J. Oh. [*The wave equation with specular derivatives*](https://arxiv.org/abs/2210.06933). *arXiv preprint arXiv:2210.06933*, 2022.

[5] K. Jung and J. Oh. [*Nonsmooth convex optimization using the specular gradient method with root-linear convergence*](https://arxiv.org/abs/2412.20747). *arXiv preprint arXiv:2210.06933*, 2024.