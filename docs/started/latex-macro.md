# LaTeX Macro

- [LaTeX Macro](#latex-macro)
  - [Preamble](#preamble)
  - [Usage examples](#usage-examples)
  - [Markdown usage](#markdown-usage)

## Preamble

{%
    include-markdown "../../README.md"
    start="<!-- latex-macro-start -->"
    end="<!-- latex-macro-end -->"
%}

## Usage examples

Use the symbols in your document after `\begin{document}`.

```tex
% A specular derivative in the one-dimensional Euclidean space
$f^{\sd}(x)$

% A specular directional derivative in normed vector spaces
$\partial^{\sd}_v f(x)$

% A specular Gateaux derivative
$\sGd f(x)$

% A specular gradient
$\sg f(x)$
```

## Markdown usage

The documentation defines `\sd`, `\sGd`, and `\sg` globally, so the same commands work inside Markdown math without adding a preamble to each page.

| Markdown source | Rendered symbol |
| --- | --- |
| `\(f^{\sd}(x)\)` | \(f^{\sd}(x)\) |
| `\(\partial^{\sd}_v f(x)\)` | \(\partial^{\sd}_v f(x)\) |
| `\(\sGd f(x)\)` | \(\sGd f(x)\) |
| `\(\sg f(x)\)` | \(\sg f(x)\) |
| `\(\frac{\sg f(x)}{\|\sg f(x)\|}\)` | \(\frac{\sg f(x)}{\|\sg f(x)\|}\) |

For a display equation, write:

```tex
\[
x_{k+1}=x_k-t_k\frac{f^{\sd}(x_k)}{|f^{\sd}(x_k)|}.
\]
```

\[
x_{k+1}=x_k-t_k\frac{f^{\sd}(x_k)}{|f^{\sd}(x_k)|}.
\]

The shared browser definitions are in `docs/javascripts/mathjax.js`, with the symbol styling in `docs/stylesheets/extra.css`.
The LaTeX preamble above is for `.tex` documents; these browser definitions are loaded on every documentation page.
