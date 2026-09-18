# Mathematical Background

## One dimension

!!! info paper "Paper"
    --8<-- "README.md:ref-specular-one-dimension"
    
    --8<-- "README.md:ref-regular-specular-euclidean"

### Angular averaging

Specular differentiation uses the geometry of one-sided secant lines. Instead of averaging their slopes, it averages the angles they make with the horizontal axis and converts the resulting angle back into a slope. This construction extends classical differentiation to corners, certain vertical tangents, and some discontinuities.

Let $f:I\to\mathbb{R}$, where $I$ is an open interval, and let $x\in I$. For sufficiently small $h>0$, define the forward and backward difference quotients by

$$
q_h^+f(x)=\frac{f(x+h)-f(x)}{h},
\qquad
q_h^-f(x)=\frac{f(x)-f(x-h)}{h}.
$$

The angular mean of two finite slopes is

$$
\mathcal B(\alpha,\beta)
=\tan\left(\frac{\arctan\alpha+\arctan\beta}{2}\right).
$$

An equivalent algebraic expression is

$$
\mathcal C(\alpha,\beta)
=\frac{
\displaystyle\frac{\alpha}{\sqrt{1+\alpha^2}}
+\displaystyle\frac{\beta}{\sqrt{1+\beta^2}}
}{
\displaystyle\frac{1}{\sqrt{1+\alpha^2}}
+\displaystyle\frac{1}{\sqrt{1+\beta^2}}
}.
$$

The **specular derivative** is defined by

$$
f^{\sd}(x)
:=\lim_{h\searrow0}\mathcal C\bigl(q_h^+f(x),q_h^-f(x)\bigr),
$$

provided this limit exists as a finite real number. The limit is taken after combining the quotients. The two quotients do not need to converge separately.

Write $\partial^+f(x)=f'_+(x)$ and $\partial^-f(x)=f'_-(x)$ for the right and left derivatives, respectively. When both are finite, the definition gives

$$
f^{\sd}(x)
=\mathcal C\bigl(\partial^+f(x),\partial^-f(x)\bigr)
=\tan\left(
\frac{\arctan\bigl(\partial^+f(x)\bigr)
+\arctan\bigl(\partial^-f(x)\bigr)}{2}
\right).
$$

The identities $\mathcal C(\alpha,\alpha)=\alpha$ and $\mathcal C(\alpha,-\alpha)=0$ explain two basic features: the specular derivative agrees with the classical derivative wherever the latter exists, and opposite slopes give a horizontal specular tangent. Moreover, $\mathcal C(\alpha,\beta)$ lies between its two arguments.

### The regular case and its geometry

The regular theory of Jung and Oh calls $f$ **regularly specularly differentiable** at $x$ when both one-sided derivatives are finite. This condition implies continuity at $x$. The general difference-quotient definition above agrees with the regular derivative at every such point.

The two one-sided tangent rays form the **phototangent**:

$$
\operatorname{pht}f(y)=
\begin{cases}
f(x)+f'_+(x)(y-x), & y\geq x,\\
f(x)+f'_-(x)(y-x), & y<x.
\end{cases}
$$

A circle centered at $(x,f(x))$ intersects these rays at two points. The chord joining them has a slope independent of the circle's radius. Its parallel translate through $(x,f(x))$ is the **specular tangent line**. This line acts as a mirror: the incident and reflected rays make equal angles with it.

### Infinite slopes and oscillation

The angular formula extends $\mathcal B$ and $\mathcal C$ to infinite one-sided slopes using $\arctan(\pm\infty)=\pm\frac{\pi}{2}$. For example,

$$
\mathcal C(+\infty,\beta)=\beta+\sqrt{1+\beta^2},
\qquad \beta\in\mathbb R.
$$

If a finite specular derivative exists and either one-sided derivative exists in the extended real line, then the other one-sided derivative exists as well, and their angular mean equals the specular derivative. Two derivatives both equal to $+\infty$, or both equal to $-\infty$, cannot produce a finite specular derivative.

### Classical and symmetric differentiation

The symmetric derivative averages the same difference quotients arithmetically:

$$
f^\ast(x)
=\lim_{h\searrow0}\frac{q_h^+f(x)+q_h^-f(x)}{2}
=\lim_{h\searrow0}\frac{f(x+h)-f(x-h)}{2h}.
$$

When both one-sided derivatives are finite, this is their arithmetic mean. At a corner, that mean generally differs from the angular mean. For the function $f(x)=\max\{x,0\}$,

$$
\partial^+f(0)=1,\qquad \partial^-f(0)=0,
\qquad f^{\sd}(0)=\sqrt2-1,
\qquad f^\ast(0)=\frac12.
$$

For $f(x)=|x|$, both generalized derivatives are zero at the origin, although the classical derivative does not exist there.

A vertical tangent can also have a finite angular average. Consider

$$
f(x)=
\begin{cases}
\sqrt{x},&x\geq0,\\
-x,&x<0.
\end{cases}
$$

Here $\partial^+f(0)=+\infty$ and $\partial^-f(0)=-1$, so $f^{\sd}(0)=\sqrt2-1$. The symmetric derivative does not exist as a finite number. More generally, neither specular nor symmetric differentiability implies the other: angular cancellation and arithmetic cancellation can behave differently when the quotients oscillate.

The base-point value also matters. It cancels from the symmetric quotient, but generally remains relevant to the angular mean. For example, the step function with $f(x)=0$ for $x\leq0$ and $f(x)=1$ for $x>0$ has $f^{\sd}(0)=1$. Changing only $f(0)$ to $\frac12$ makes both quotients tend to $+\infty$, so no finite specular derivative remains at zero. Thus the definition applies directly at discontinuities and can depend on the assigned value of the function there.

Specular differentiation is nonlinear. The usual linearity and chain rules cannot be assumed, even though its value agrees with the classical derivative at every classically differentiable point.

Even multiplication by a constant need not commute with the derivative. For $r(x)=\max\{x,0\}$,

$$
(2r)^{\sd}(0)=\frac{\sqrt5-1}{2}
\ne 2(\sqrt2-1)=2r^{\sd}(0).
$$

### The Quasi-Mean Value Theorem

Suppose $a<b$, and let $f:[a,b]\to\mathbb R$ be continuous on $[a,b]$ and specularly differentiable on $(a,b)$. Then there exist $c_1,c_2\in(a,b)$ such that

$$
f^{\sd}(c_1)
\leq\frac{f(b)-f(a)}{b-a}
\leq f^{\sd}(c_2).
$$

This theorem retains the order information of the classical Mean Value Theorem: values of the specular derivative bound every secant slope. It does not require the secant slope to equal the specular derivative at one point.

Under the same continuity and specular differentiability assumptions, $f$ is nondecreasing exactly when $f^{\sd}\geq0$ throughout the interior, and nonincreasing exactly when $f^{\sd}\leq0$. In particular, $f^{\sd}\equiv0$ on $(a,b)$ if and only if $f$ is constant on $[a,b]$.

For convex functions, the derivative also has a direct interpretation in convex analysis. Every real-valued convex function on an open interval is continuous and specularly differentiable, and

$$
\partial^-f(x)\leq f^{\sd}(x)\leq\partial^+f(x).
$$

Consequently, $f^{\sd}(x)$ belongs to the convex subdifferential $\partial f(x)$ and satisfies

$$
f(y)\geq f(x)+f^{\sd}(x)(y-x)
\qquad\text{for all }x,y\in I.
$$

This gives a particular subgradient selected by angular averaging.

### Continuity and classical regularity

Specular differentiability alone does not imply continuity or even Lebesgue measurability. Without either assumption, however, an everywhere specularly differentiable function has at most countably many discontinuities at which its specular derivative is nonzero. The restriction to nonzero derivatives is essential: there are nowhere-continuous, nonmeasurable functions with identically zero specular derivative.

For a **continuous** function $f$ that is specularly differentiable throughout an open interval $I$, stronger conclusions hold:

- $f^{\sd}$ is a pointwise limit of continuous functions, hence is of Baire class $1$. Its continuity points form a dense $G_\delta$ subset of $I$.
- At every continuity point of $f^{\sd}$, the classical derivative exists and $f'=f^{\sd}$.
- If $|f^{\sd}|\leq M$ throughout $I$, then $|f(y)-f(x)|\leq M|y-x|$ for all $x,y\in I$. Thus $f$ is Lipschitz continuous and $f'=f^{\sd}$ almost everywhere.
- If $f^{\sd}$ is continuous throughout $I$, then $f\in C^1(I)$ and $f'=f^{\sd}$ everywhere.

These results make continuity a substantive hypothesis, rather than an automatic consequence of the generalized derivative.

The second-order derivative is defined by $f^{\sd\sd}=(f^{\sd})^{\sd}$. Let $S^1(I)$ consist of continuous specularly differentiable functions, and let $S^2(I)$ consist of those $f\in S^1(I)$ for which $f^{\sd}\in S^1(I)$. Then

$$
C^2(I)\subsetneq S^2(I)\subsetneq C^1(I)
\subsetneq S^1(I)\subsetneq C^0(I).
$$

The definition of $S^2$ requires continuity of both $f$ and $f^{\sd}$. Merely taking the specular derivative twice does not ensure classical differentiability, even for continuous functions.

For a simple example in $S^2$ but outside $C^2$, take $f(x)=\frac12x|x|$. Then $f^{\sd}(x)=|x|$ and $f^{\sd\sd}(x)=\operatorname{sgn}(x)$, with $\operatorname{sgn}(0)=0$. The second specular derivative exists at the origin, where the classical second derivative does not.

The regular theory gives a complementary result: if both $f$ and $f^{\sd}$ are **regularly** specularly differentiable throughout $I$, then $f\in C^1(I)$ and $f'=f^{\sd}$. In this case, the regular differentiability of $f^{\sd}$ supplies the continuity required above.

### Integration

The regular theory also relates differentiation to integration. Here piecewise continuity means continuity except at finitely many interior points, with finite one-sided limits at each of those points and one-sided continuity at the endpoints. For such a function $g:[a,b]\to\mathbb{R}$, define

$$
F(x)=\int_a^x g(t)\,dt.
$$

At every interior point, its regular specular derivative is the angular mean of the one-sided limits of $g$:

$$
F^{\sd}(x)=\tan\left(
\frac{\arctan g(x+)+\arctan g(x-)}{2}
\right).
$$

This recovers $g(x)$ at every continuity point. At a jump, the assigned value $g(x)$ must equal this angular mean for the same identity to hold.

Conversely, if $F$ is regularly specularly differentiable on $[a,b]$ and $F^{\sd}$ is Riemann integrable, then

$$
\int_a^b F^{\sd}(x)\,dx=F(b)-F(a).
$$

Here regular differentiability on the closed interval includes finite inward one-sided derivatives at the endpoints, which define $F^{\sd}(a)$ and $F^{\sd}(b)$.

## Higher dimension

!!! info paper "Paper"
    --8<-- "README.md:ref-specular-normed-spaces"

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

Equivalently, the operator norm of $\sGd f(x^*)$ is at most $1$. The derivative can be nonzero at an extremum: already in one dimension, $f(t)=t-2|t|$ has a maximum at $0$ but

$$
f^{\sd}(0)=\sqrt5-2\ne0.
$$

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

For a function that may take the value $+\infty$, its effective domain is $\operatorname{dom}f=\{x:f(x)<+\infty\}$. Such a function is **proper** if this domain is nonempty. At a point where $f(x)$ is finite, the **Fréchet subdifferential** $\widehat\partial f(x)$ consists of the functionals $\ell\in X^*$ satisfying

$$
\liminf_{\substack{w\to0\\w\ne0}}
\frac{f(x+w)-f(x)-\ell(w)}{\|w\|}\geq0.
$$

Now suppose $f:X\to\mathbb R\cup\{+\infty\}$ is proper, convex, and lower semicontinuous. If it is specularly Fréchet differentiable at an interior point $x$ of its effective domain, then

$$
\widehat Df(x)\in\widehat\partial f(x).
$$

On a reflexive Banach space, the differential also belongs to the convex subdifferential $\partial f(x)$. In particular, in $\mathbb R^n$ the specular gradient then satisfies the supporting-hyperplane inequality

$$
f(y)\ge f(x)+\langle\sg f(x),y-x\rangle
\qquad\text{for every }y\in\mathbb R^n.
$$

This connects specular differentiation with convex optimization while retaining the differentiability assumptions needed for a single gradient to represent all directions.

### Relation to the package

The package evaluates finite-difference approximations to these quantities. For scalar-valued functions, `specular.derivative` combines the two sampled quotients through $\mathcal C$, while `specular.gradient` assembles the corresponding coordinate values. `specular.jacobian` applies this coordinate construction to each scalar output component. These computations do not by themselves establish the existence of a Gâteaux or Fréchet differential.

The function `specular.scaled_mean` also exposes the rescaled angular mean

$$
\mathcal C_\sigma(\alpha,\beta)
=\sigma\mathcal C\left(\frac{\alpha}{\sigma},\frac{\beta}{\sigma}\right),
\qquad \sigma>0.
$$

The unscaled mean corresponds to $\sigma=1$. See [Calculation](calculation.md) for numerical evaluation and [Optimization](optimization.md) for the SPEG iteration.

## References

{%
    include-markdown "../../README.md"
    start="<!-- references-start -->"
    end="<!-- references-end -->"
%}
