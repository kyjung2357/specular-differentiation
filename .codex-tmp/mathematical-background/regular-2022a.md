### The regular case and its geometry

The regular theory studies points where both ordinary one-sided derivatives exist as finite real numbers. Let $I\subset\mathbb{R}$ be an open interval, let $f:I\to\mathbb{R}$, and fix $x\in I$. We say that $f$ is **regularly specularly differentiable** at $x$ when $f'_+(x)$ and $f'_-(x)$ are finite. This condition implies continuity at $x$.

The two one-sided tangent rays form the **phototangent**:

$$
\operatorname{pht}f(y)=
\begin{cases}
f(x)+f'_+(x)(y-x), & y\geq x,\\
f(x)+f'_-(x)(y-x), & y<x.
\end{cases}
$$

A circle centered at $(x,f(x))$ intersects these rays at two points. The chord joining them has a slope independent of the circle's radius. Its parallel translate through $(x,f(x))$ is the **specular tangent line**. Geometrically, this line acts as a mirror: the incident and reflected rays make equal angles with it.

Writing $\alpha=f'_+(x)$ and $\beta=f'_-(x)$, the resulting slope is

$$
f^{\sd}(x)
=\tan\left(\frac{\arctan\alpha+\arctan\beta}{2}\right)
=\frac{\alpha\sqrt{1+\beta^2}+\beta\sqrt{1+\alpha^2}}
{\sqrt{1+\alpha^2}+\sqrt{1+\beta^2}}.
$$

This geometric definition agrees with the difference-quotient limit defining the general specular derivative. The regular theory is therefore a subclass of the general one-dimensional theory, rather than a different value assigned to the same finite one-sided slopes. General specular differentiability does not require those two ordinary one-sided derivatives to exist as finite numbers.

### Corners and nonlinear behavior

At a classically differentiable point, $\alpha=\beta=f'(x)$, so $f^{\sd}(x)=f'(x)$. At a corner, angular averaging gives a single slope even when the classical derivative is unavailable. For example,

$$
f(x)=|x|
\qquad\text{gives}\qquad
f^{\sd}(x)=\operatorname{sgn}(x),
$$

with $f^{\sd}(0)=0$. Thus a regular specular derivative need not be continuous. For $r(x)=\max\{x,0\}$, the one-sided slopes at zero are $1$ and $0$, and

$$
r^{\sd}(0)=\tan\frac{\pi}{8}=\sqrt{2}-1.
$$

The symmetric derivative instead averages the slopes arithmetically and equals $1/2$ at this point. Specular differentiation is nonlinear: for the same function,

$$
(2r)^{\sd}(0)=\frac{\sqrt{5}-1}{2}
\ne 2(\sqrt{2}-1)=2r^{\sd}(0).
$$

The classical chain rule also need not hold at a nonsmooth point. For a composite function with finite one-sided derivatives, angular averaging can be applied to those slopes directly.

### Regularity from the specular derivative

The regular derivative still controls classical regularity. Suppose $f$ is continuous on $I$ and regularly specularly differentiable on $I\setminus\{x_0\}$. If

$$
\lim_{x\to x_0,\;x\ne x_0}f^{\sd}(x)=L\in\mathbb{R},
$$

then the classical derivative exists at $x_0$ and satisfies

$$
f'(x_0)=f^{\sd}(x_0)=L.
$$

Consequently, if both $f$ and $f^{\sd}$ are regularly specularly differentiable throughout $I$, then $f\in C^1(I)$ and $f'=f^{\sd}$. Here **regularly** is essential: two successive specular derivatives under the broader limit definition do not, by themselves, imply classical differentiability.

### Integration

For a piecewise continuous function $g:[a,b]\to\mathbb{R}$, define

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
