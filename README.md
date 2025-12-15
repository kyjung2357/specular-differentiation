# Specular Differentiation

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the Python package `specular_diff` and codes for applications:

* [**Nonsmooth convex optimization**](#nonsmooth-convex-optimization)
  * Directory: `nonsmooth-convex-opt/`
  * Related reference: [[2]](#references), [[5]](#references)

* [**Initial value problems for ordinary differential equations**](#initial-value-problems-for-ordinary-differential-equations)
  * Directory: `numerical-ODE/`
  * Related reference: [[1]](#references), [[3]](#references), [[4]](#references)


## Installation

You can install the released version directly from PyPI:

```bash
pip install specular-differentiation
```

## Introduction

Specular differentiation generalizes classical differentiation.
A specular derivative can be understood as the average of the inclination angles of the right and left derivatives. 
In contrast, a symmetric derivative is the average of the right and left derivatives.
Their difference is illustrated as in the following figure.

![specular-derivative-animation](figures/specular_derivative_animation.gif)


## Applications

Specular differentiation is defined in normed vector spaces, allowing for applications in higher-dimensional Euclidean spaces. 
Two applications are provided in this repository.

### Nonsmooth convex optimization



### Initial value problems for ordinary differential equations

In [[1]](#references), *the specular Euler scheme of Type 5* is introduced, which generates a sequence $\{ u_n \}_{n=0}^{\infty}$ according to the formula

$$
u_{n+1} = u_n + h \, \mathcal{A}(F(t_{n+1}, u_{n+1}), F(t_n, u_n)),
$$

where the initial time $t_0 \geq 0$ and the starting point $u_0 \in \mathbb{R}$ are given, and $t_n := t_0 + nh$ for $n \in \mathbb{N}$ with a step size $h > 0$.

## References

[1] K. Jung. Nonlinear numerical schemes using specular differentiation for initial value problems of first-order ordinary differential equations. arXiv preprint arXiv:??, 2025.

[2] K. Jung. Specular differentiation in normed vector spaces and its applications to nonsmooth convex optimization. arXiv preprint arXiv:??, 2025. 

[3] K. Jung and J. Oh. [The specular derivative](https://arxiv.org/abs/2210.06062). *arXiv preprint arXiv:2210.06062*, 2022.

[4] K. Jung and J. Oh. [The wave equation with specular derivatives](https://arxiv.org/abs/2210.06933). *arXiv preprint arXiv:2210.06933*, 2022.

[5] K. Jung and J. Oh. [Nonsmooth convex optimization using the specular gradient method with root-linear convergence](https://arxiv.org/abs/2412.20747). *arXiv preprint arXiv:2210.06933*, 2024.