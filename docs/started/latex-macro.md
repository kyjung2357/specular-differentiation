# 1.3. LaTeX Macro

To use the specular differentiation symbols in your LaTeX document, add the following code to your preamble before `\begin{document}`.

```tex
% Required packages
\usepackage{graphicx}
\usepackage{bm}
\usepackage{amssymb}

% specular derivative symbol
\newcommand\sd[1][.5]{\mathbin{\vcenter{\hbox{\scalebox{#1}{\,$\bm{\wedge}$}}}}}

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
