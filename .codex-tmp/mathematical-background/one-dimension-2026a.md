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

Write $\partial^+f(x)$ and $\partial^-f(x)$ for the right and left derivatives, respectively. When both are finite, the definition gives

$$
f^{\sd}(x)
=\mathcal C\bigl(\partial^+f(x),\partial^-f(x)\bigr)
=\tan\left(
\frac{\arctan\bigl(\partial^+f(x)\bigr)
+\arctan\bigl(\partial^-f(x)\bigr)}{2}
\right).
$$

The identities $\mathcal C(\alpha,\alpha)=\alpha$ and $\mathcal C(\alpha,-\alpha)=0$ explain two basic features: the specular derivative agrees with the classical derivative wherever the latter exists, and opposite slopes give a horizontal specular tangent. Moreover, $\mathcal C(\alpha,\beta)$ lies between its two arguments.

The angular formula also accommodates infinite one-sided derivatives using $\arctan(\pm\infty)=\pm\frac{\pi}{2}$. For example,

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
