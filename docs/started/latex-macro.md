# LaTeX Macro

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
