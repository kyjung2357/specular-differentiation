window.MathJax = {
  loader: {
    load: ["[tex]/html"]
  },
  tex: {
    packages: {"[+]": ["html"]},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    // Browser equivalents of the LaTeX macros in README.md. MathJax does not
    // load graphicx: backprime supplies the reflected prime, and the gradient
    // scaling lives in extra.css. Its extra advance is (scale_x - 1) * 0.722em
    // for MathJax's TeX blacktriangledown, so fractions include the scaled width.
    // https://docs.mathjax.org/en/v3.2/input/tex/extensions/html.html
    macros: {
      sd: "\\mathord{\\prime\\mkern-2.5mu{\\scriptstyle\\backprime}}",
      sGd: "\\widehat{\\mkern-2mu d}\\mkern1mu",
      sg: "\\mathchoice"
        + "{\\mathord{\\rule{0pt}{1.3ex}\\smash{\\class{specular-gradient-main}{\\blacktriangledown}}\\hspace{0.26714em}\\mkern-1.2mu}}"
        + "{\\mathord{\\rule{0pt}{1.3ex}\\smash{\\class{specular-gradient-main}{\\blacktriangledown}}\\hspace{0.26714em}\\mkern-1.2mu}}"
        + "{\\mathord{\\rule{0pt}{1.0ex}\\smash{\\class{specular-gradient-script}{\\blacktriangledown}}\\hspace{0.20938em}\\mkern-0.8mu}}"
        + "{\\mathord{\\rule{0pt}{0.8ex}\\smash{\\class{specular-gradient-scriptscript}{\\blacktriangledown}}\\hspace{0.12996em}\\mkern-0.5mu}}"
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  // The first page event can precede the asynchronously loaded MathJax bundle.
  if (MathJax.startup) {
    MathJax.startup.promise.then(() => MathJax.typesetPromise());
  }
});
