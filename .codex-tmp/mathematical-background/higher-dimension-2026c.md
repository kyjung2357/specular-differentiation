The normed-space theory extends specular differentiation from a scalar variable to directions in a vector space. It distinguishes three levels: a derivative along one direction, a continuous linear operator representing all directional derivatives, and a uniform approximation as the increment approaches zero from arbitrary directions. The discussion below focuses on real-valued functions; the paper also defines specular differentiation for maps between normed vector spaces.

### Specular directional derivatives

Let $X$ be a real normed vector space, let $\Omega\subset X$ be open, and let $f:\Omega\to\mathbb R$. The norm measures the length of a direction and is part of the definition. The corresponding product norm on the graph space is

$$
\|(v,y)\|_{X\times\mathbb R}
=\sqrt{\|v\|^2+|y|^2}.
$$

For $x\in\Omega$, a nonzero direction $v\in X$, and sufficiently small $h>0$, define the slopes per unit distance by

$$
q_{h,v}^{+}(x)=\frac{f(x+hv)-f(x)}{h\|v\|},
\qquad
q_{h,v}^{-}(x)=\frac{f(x)-f(x-hv)}{h\|v\|}.
$$

Using the angular mean $\mathcal C$ from the one-dimensional definition, the **specular directional derivative** is

$$
\partial_v^{\sd}f(x)
=\|v\|\lim_{h\downarrow0}
\mathcal C\left(q_{h,v}^{+}(x),q_{h,v}^{-}(x)\right).
$$

This definition requires the displayed limit to exist as a finite real number. The derivative in the zero direction is defined to be zero. If the classical directional derivative exists, the specular directional derivative agrees with it. As in one dimension, the angular-mean limit can exist even when the individual slopes do not converge.

When the one-sided directional derivatives

$$
\partial_v^+f(x)=\lim_{h\downarrow0}\frac{f(x+hv)-f(x)}{h},
\qquad
\partial_v^-f(x)=\lim_{h\downarrow0}\frac{f(x)-f(x-hv)}{h}
$$

are finite, the formula becomes

$$
\partial_v^{\sd}f(x)
=\|v\|\tan\left[
\frac{1}{2}\arctan\left(\frac{\partial_v^+f(x)}{\|v\|}\right)
+\frac{1}{2}\arctan\left(\frac{\partial_v^-f(x)}{\|v\|}\right)
\right].
$$

The factors $\|v\|$ are essential. For the normalized line restriction

$$
F_v(t)=\frac{f(x+tv)}{\|v\|},
$$

the exact relation is

$$
\partial_v^{\sd}f(x)=\|v\|F_v^{\sd}(0).
$$

For a unit direction, this is simply the specular derivative of $t\mapsto f(x+tv)$. For a general direction, omitting the normalization changes the angular mean. This relation follows directly from the definition and does not require a general chain rule.

### Gâteaux and Fréchet differentiability

Directional derivatives describe one line at a time. To obtain a differential, they must fit together into a continuous linear map. Write $X^*$ for the space of continuous linear functionals on $X$.

The function $f$ is **specularly Gâteaux differentiable** at $x$ if every specular directional derivative exists and there is an $\ell\in X^*$ such that

$$
\partial_v^{\sd}f(x)=\ell(v)
\qquad\text{for every }v\in X.
$$

This operator is unique and is denoted by $\sGd f(x)$. The existence of directional derivatives alone does not establish their linearity, and specular Gâteaux differentiability alone does not imply continuity of $f$.

The stronger **specular Fréchet differentiability** condition requires an $\ell\in X^*$ satisfying

$$
\lim_{\substack{\|w\|\to0\\w\ne0}}
\left|
\mathcal C\left(
\frac{f(x+w)-f(x)}{\|w\|},
\frac{f(x)-f(x-w)}{\|w\|}
\right)
-\frac{\ell(w)}{\|w\|}
\right|=0.
$$

Here the limit covers all small increments $w$, rather than holding a direction fixed first. The resulting operator is also unique and is written $\widehat Df(x)$. It satisfies

$$
\widehat Df(x)(v)=\sGd f(x)(v)=\partial_v^{\sd}f(x).
$$

Classical Fréchet differentiability implies specular Fréchet differentiability with the same differential; classical Gâteaux differentiability likewise implies specular Gâteaux differentiability with the same operator. Specular Fréchet differentiability implies specular Gâteaux differentiability.

For a basic nonsmooth example, consider $f(x)=\|x\|$ on any nonzero real normed space. At the origin, the two normalized slopes in the Fréchet definition are $1$ and $-1$. Since $\mathcal C(1,-1)=0$, the condition holds with

$$
\widehat Df(0)=0.
$$

Thus the norm has a specular Fréchet differential at the origin even though it has no classical Fréchet differential there.

### Quasi-Mean Value and Quasi-Fermat theorems

The **Quasi-Mean Value Theorem** bounds a finite change in the function using specular derivatives along the connecting segment. Assume that $\Omega$ is open and convex, that $f$ is continuous and specularly Gâteaux differentiable throughout $\Omega$, and that both one-sided directional derivatives exist as extended real numbers at every point in every direction. They must not both equal $+\infty$ or both equal $-\infty$.

For distinct $a,b\in\Omega$, put $w=b-a$ and define the line restriction $F(t)=f(a+tw)$ for $t\in[0,1]$. Then

$$
\inf_{0<t<1}F^{\sd}(t)
\le f(b)-f(a)
\le\sup_{0<t<1}F^{\sd}(t),
$$

where the directional form of the line derivative is

$$
F^{\sd}(t)
=\frac{\partial_w^{\sd}\bigl(\|w\|f\bigr)(a+tw)}{\|w\|}.
$$

The derivative acts on the scaled function $\|w\|f$. The two bounds replace the single equality in the classical Mean Value Theorem; they need not be attained at the same point.

The **Quasi-Fermat Theorem** gives a necessary condition at an extremum. If $\Omega$ is open, $f$ is specularly Gâteaux differentiable in $\Omega$, and $x^*\in\Omega$ is a local minimizer or maximizer, then

$$
\left|\partial_v^{\sd}f(x^*)\right|\le\|v\|
\qquad\text{for every }v\in X.
$$

Equivalently, the operator norm of $\sGd f(x^*)$ is at most $1$. A vanishing derivative is therefore not a necessary condition in this general specular setting.

### Gradients and convex functions

In a real Hilbert space $H$, a specular Fréchet differential has a unique representing vector, the **specular gradient** $\sg f(x)$, satisfying

$$
\widehat Df(x)(v)=\langle\sg f(x),v\rangle_H
\qquad\text{for every }v\in H.
$$

In Euclidean space, its components are the specular partial derivatives:

$$
\sg f(x)
=\left(\partial_{e_1}^{\sd}f(x),\ldots,
\partial_{e_n}^{\sd}f(x)\right),
$$

where $e_1,\ldots,e_n$ are the standard basis vectors. This identification assumes the specular Fréchet differential exists. A vector assembled from coordinate derivatives alone need not represent every directional derivative.

For example, $f(x_1,x_2)=\max\{x_1,x_2\}$ has

$$
\partial_{e_1}^{\sd}f(0)=\partial_{e_2}^{\sd}f(0)=\sqrt2-1,
\qquad
\partial_{(1,1)}^{\sd}f(0)=1.
$$

The diagonal value differs from $2(\sqrt2-1)$, so these directional derivatives cannot come from a single linear operator. This function has specular coordinate derivatives at the origin, but no specular Gâteaux or Fréchet differential there.

Finally, suppose $f:X\to\mathbb R\cup\{+\infty\}$ is proper, convex, and lower semicontinuous. If it is specularly Fréchet differentiable at an interior point $x$ of its effective domain, then

$$
\widehat Df(x)\in\widehat\partial f(x),
$$

where $\widehat\partial f(x)$ is the Fréchet subdifferential. The paper also states membership in the convex subdifferential when $X$ is a reflexive Banach space. In particular, in $\mathbb R^n$ the specular gradient then satisfies the supporting-hyperplane inequality

$$
f(y)\ge f(x)+\langle\sg f(x),y-x\rangle
\qquad\text{for every }y\in\mathbb R^n.
$$

This connects specular differentiation with convex optimization while retaining the differentiability assumptions needed for a single gradient to represent all directions.
