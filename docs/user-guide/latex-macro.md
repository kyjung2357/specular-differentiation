# LaTeX Macro

## Preamble

{%
    include-markdown "../../README.md"
    start="<!-- latex-macro-start -->"
    end="<!-- latex-macro-end -->"
%}

The packages above support the following commands:

| Package | Purpose |
| --- | --- |
| `fontenc` with `T1` | Provides the `đ` and `Đ` text symbols used by `\sGd` and `\sFd`. |
| `amsmath` | Provides `\text` for `\sGd` and `\sFd`. |
| `graphicx` | Reflects the prime in `\sd` and scales the triangle in `\sg`. |
| `amssymb` | Provides `\blacktriangledown` for `\sg`. |

If you only use `\sGd` and `\sFd`, you only need `fontenc` with `T1` and `amsmath` from this list. Do not load a package again if your document already loads it.

## Usage examples

Use the symbols in your document after `\begin{document}`.

```tex
% A specular derivative in the one-dimensional Euclidean space
$f^{\sd}(x)$

% A specular directional derivative in normed vector spaces
$\partial^{\sd}_v f(x)$

% A specular Gateaux derivative
$\sGd f(x)$

% A specular Frechet derivative
$\sFd f(x)$

% A specular gradient
$\sg f(x)$
```

## Markdown usage

The documentation defines `\sd`, `\sGd`, `\sFd`, and `\sg` globally, so the same commands work inside Markdown math without adding a preamble to each page. The Gâteaux and Fréchet symbols are the upright letters `đ` and `Đ`, respectively.

| Markdown source | Rendered symbol |
| --- | --- |
| `\(f^{\sd}(x)\)` | \(f^{\sd}(x)\) |
| `\(\partial^{\sd}_v f(x)\)` | \(\partial^{\sd}_v f(x)\) |
| `\(\sGd f(x)\)` | \(\sGd f(x)\) |
| `\(\sFd f(x)\)` | \(\sFd f(x)\) |
| `\(\sg f(x)\)` | \(\sg f(x)\) |
| `\(\frac{\sg f(x)}{\|\sg f(x)\|}\)` | \(\frac{\sg f(x)}{\|\sg f(x)\|}\) |
