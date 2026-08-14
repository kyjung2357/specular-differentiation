# Specular Differentiation

[![PyPI version](https://badge.fury.io/py/specular-differentiation.svg)](https://badge.fury.io/py/specular-differentiation)
![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)
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

## Installation

### Requirements

`specular-differentiation` requires:

* **Python** >= 3.14
* `numpy` >= 2.4

Additional backends are available through optional dependencies:

* `numba`: `numba`
* `jax`: `jax`, `jaxlib`

**Standard Installation**

```bash
pip install specular-differentiation
```

The package is distributed as `specular-differentiation` and imported in
Python as `specular`.

This installs `derivative`, `gradient`, and `jacobian`, using NumPy by default,
as well as the scalar ODE schemes. Backend selection is available through
`set_backend`, `get_backend`, `use_backend`, and `available_backends`.

**Optional features**

```bash
pip install "specular-differentiation[numba]"           # Numba backend
pip install "specular-differentiation[jax]"             # JAX backend
```

**Developer installation**

```bash
pip install -e ".[dev]"
```

Documentation tools can be installed separately with:

```bash
pip install -e ".[docs]"
```

## [Backend support](https://kyjung2357.github.io/specular-differentiation/api/backend/)

The package is organized around a backend system. 
NumPy is the default backend, while accelerated backends are optional and may require extra dependencies.

## Quick start

The following simple example calculates the specular derivative of the [ReLU function](https://en.wikipedia.org/wiki/Rectified_linear_unit) $f(x) = max(0, x)$ at the origin.

```python
import specular

ReLU = lambda x: max(x, 0)
print(specular.derivative(ReLU, x=0))
```

```text
0.41421356237309503
```

## Scalar ODE schemes

The package provides three scalar specular ellipse schemes directly from the
top-level namespace:

```python
import specular

result = specular.ellipse_scheme(
    lambda t, y: -y,
    (0.0, 1.0),
    1.0,
    n_steps=100,
    sigma=1.0,
)
```

`ellipse_scheme`, `ellipse_scheme_3rd_order`, and
`ellipse_scheme_4th_order` all accept either a positive scalar `sigma` or a
callable scale policy. A policy receives the mesh width `h`, so it may depend
on the discretization. The advertised third- and fourth-order convergence is
conditional on the corresponding cancellation assumptions; choosing an
arbitrary or mesh-dependent policy does not by itself provide that order.

See the [scalar ODE API documentation](https://kyjung2357.github.io/specular-differentiation/api/ode/)
for the callback contracts.

## Documentation

- [Calculation API](https://kyjung2357.github.io/specular-differentiation/api/calculation/)
- [Scalar ODE API](https://kyjung2357.github.io/specular-differentiation/api/ode/)
- [Backend API](https://kyjung2357.github.io/specular-differentiation/api/backend/)

## LaTeX Macro

<!-- latex-macro-start -->

To use the specular differentiation symbol in your LaTeX document, add the following code to your preamble (before `\begin{document}`):

```latex
% Required packages
\usepackage{graphicx}
\usepackage{amssymb}

% specular derivative symbol
\newcommand{\sd}{\mathord{\prime\mkern-2.5mu\reflectbox{$\scriptstyle\prime$}}}

% specular Gateaux derivative symbol
\newcommand{\sGd}{\widehat{\mkern-2mu d}\mkern1mu}

% specular gradient symbol
\newcommand{\sg}{%
  \mathchoice
    {\mathord{\raisebox{-0.05ex}{\rule{0pt}{1.3ex}\smash{\scalebox{1.37}[1.22]{\ensuremath{\displaystyle\blacktriangledown}}}}\mkern-1.2mu}}
    {\mathord{\raisebox{-0.05ex}{\rule{0pt}{1.3ex}\smash{\scalebox{1.37}[1.22]{\ensuremath{\textstyle\blacktriangledown}}}}\mkern-1.2mu}}
    {\mathord{\raisebox{-0.03ex}{\rule{0pt}{1.0ex}\smash{\scalebox{1.29}[1.15]{\ensuremath{\scriptstyle\blacktriangledown}}}}\mkern-0.8mu}}
    {\mathord{\raisebox{-0.02ex}{\rule{0pt}{0.8ex}\smash{\scalebox{1.18}[1.05]{\ensuremath{\scriptscriptstyle\blacktriangledown}}}}\mkern-0.5mu}}
}
```

<!-- latex-macro-end -->

## Citing specular-differentiation

To cite this repository:

```bibtex
@software{specular_diff,
  author = {Jung, Kiyuob},
  title = {{specular-differentiation}},
  doi = {10.5281/zenodo.18246734},
  url = {https://github.com/kyjung2357/specular-differentiation},
  version = {1.3.0},
  year = {2026},
}
```

## References

[1] K. Jung. [*Specular differentiation in one dimension: a quasi-mean value theorem, regularity, and discontinuities*](https://arxiv.org/abs/2601.09900). arXiv preprint arXiv:2601.09900, 2026.

[2] K. Jung. [*Specular differentiation in normed vector spaces: Quasi-Mean Value and Quasi-Fermat Theorems*](https://arxiv.org/abs/2601.10950). arXiv preprint arXiv:2601.10950, 2026. 

[3] K. Jung. [*Specular gradient methods for nonsmooth convex optimization in Euclidean spaces: a subgradient selection strategy*](https://arxiv.org/abs/2605.25490). arXiv preprint 	arXiv:2605.25490, 2026.
