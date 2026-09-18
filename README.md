# Specular Differentiation

[![PyPI version](https://badge.fury.io/py/specular-differentiation.svg)](https://badge.fury.io/py/specular-differentiation)
![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB.svg?style=flat&logo=python&logoColor=white)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18246734.svg)](https://doi.org/10.5281/zenodo.18246734)
[![License](https://img.shields.io/pypi/l/specular-differentiation.svg)](https://pypi.org/project/specular-differentiation/)
[![CodeFactor](https://www.codefactor.io/repository/github/kyjung2357/specular-differentiation/badge)](https://www.codefactor.io/repository/github/kyjung2357/specular-differentiation)
[![CodeQL Advanced](https://github.com/kyjung2357/specular-differentiation/actions/workflows/codeql.yml/badge.svg)](https://github.com/kyjung2357/specular-differentiation/actions/workflows/codeql.yml)
[![Docs](https://img.shields.io/github/deployments/kyjung2357/specular-differentiation/github-pages?label=docs&logo=github)](https://kyjung2357.github.io/specular-differentiation)

<p>
  The Python package <code>specular</code> implements <em>specular differentiation</em>, which generalizes classical differentiation.
  See <a href="#references">References</a> for more details.
</p>

The specular derivative is defined by averaging the angles associated with
the forward and backward difference quotients:

$$
f^{\mathord{\prime\mkern-2.5mu{\scriptstyle\backprime}}}(x)
:=
\tan\left(
  \frac{\arctan\bigl(f'_+(x)\bigr)+\arctan\bigl(f'_-(x)\bigr)}{2}
\right),
$$

where $f'_+(x)$ and $f'_-(x)$ are the right and left derivatives of $f$
at $x$, respectively.
In contrast, the symmetric derivative takes the arithmetic mean of these
derivatives. The animation below compares the two: the dashed blue and green
lines have slopes equal to the specular and symmetric derivatives,
respectively.

<div class="home-animation">
  <img
    src="https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/specular-derivative-animation.gif"
    alt="Animation comparing specular and symmetric derivatives"
  >
</div>

## Installation

**Requirements**

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

This installs `scaled_mean`, `derivative`, `gradient`, and `jacobian`, using
NumPy by default, as well as the scalar ODE methods and the SPEG optimizer.
Backend selection is
available through `set_backend`, `get_backend`, `use_backend`, and
`available_backends`.

**Optional features**

```bash
pip install "specular-differentiation[numba]"           # Numba backend
pip install "specular-differentiation[jax]"             # JAX backend
```

**Developer installation**

```bash
pip install -e ".[dev]"
```

## [Backend support](https://kyjung2357.github.io/specular-differentiation/api/backend/)

| Backend | Availability | Minimum version |
| :--- | :--- | :--- |
| NumPy | Default | `numpy >= 2.4` |
| Numba | Optional | `numba >= 0.65` |
| JAX | Optional | `jax >= 0.10`, `jaxlib >= 0.10` |

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

## Documentation

- [User Guide](https://kyjung2357.github.io/specular-differentiation/user-guide/)
- [API Reference](https://kyjung2357.github.io/specular-differentiation/api/)
- [Examples](https://kyjung2357.github.io/specular-differentiation/examples/)
- [Release](https://github.com/kyjung2357/specular-differentiation/releases/latest)

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

For usage examples and Markdown notation, see
[LaTeX Macro in the User Guide](https://kyjung2357.github.io/specular-differentiation/user-guide/latex-macro/).

## Citing specular-differentiation

To cite this repository:

```bibtex
@software{specular_diff,
  author = {Jung, Kiyuob},
  title = {{specular-differentiation}},
  doi = {10.5281/zenodo.18246734},
  url = {https://github.com/kyjung2357/specular-differentiation},
  version = {1.3.2},
  year = {2026},
}
```

## References

<!-- references-start -->

**One dimension**

<!-- --8<-- [start:ref-specular-one-dimension] -->
[1] K. Jung. [*Specular differentiation in one dimension: a quasi-mean value theorem, regularity, and discontinuities*](https://arxiv.org/abs/2601.09900). arXiv preprint arXiv:2601.09900, 2026.
<!-- --8<-- [end:ref-specular-one-dimension] -->

<!-- --8<-- [start:ref-regular-specular-euclidean] -->
[2] K. Jung and J. Oh. [*Regular specular differentiation in Euclidean spaces*](https://arxiv.org/abs/2210.06062v3). arXiv preprint arXiv:2210.06062v3, 2022.
<!-- --8<-- [end:ref-regular-specular-euclidean] -->

**Higher dimensions**

<!-- --8<-- [start:ref-specular-normed-spaces] -->
[3] K. Jung. [*Specular differentiation in normed vector spaces: Quasi-Mean Value and Quasi-Fermat Theorems*](https://arxiv.org/abs/2601.10950). arXiv preprint arXiv:2601.10950, 2026.
<!-- --8<-- [end:ref-specular-normed-spaces] -->

**Applications**

<!-- --8<-- [start:ref-ellipse-ode] -->
[4] K. Jung. [*The specular ellipse method for scalar ordinary differential equations: exactness and accuracy up to fourth order*](https://arxiv.org/abs/2608.30280). arXiv preprint arXiv:2608.30280, 2026.
<!-- --8<-- [end:ref-ellipse-ode] -->

<!-- --8<-- [start:ref-speg-one-dimension] -->
[5] K. Jung and J. Oh. [*Nonsmooth convex optimization using the specular gradient method with root-linear convergence*](https://arxiv.org/abs/2412.20747). arXiv preprint arXiv:2412.20747, 2024.
<!-- --8<-- [end:ref-speg-one-dimension] -->

<!-- --8<-- [start:ref-specular-gradient-convex] -->
[6] K. Jung. [*Specular gradient methods for nonsmooth convex optimization in Euclidean spaces: a subgradient selection strategy*](https://arxiv.org/abs/2605.25490). arXiv preprint arXiv:2605.25490, 2026.
<!-- --8<-- [end:ref-specular-gradient-convex] -->

<!-- references-end -->
