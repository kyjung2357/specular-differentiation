from ._wrapper import _wrapper

def gradient_descent(objective_function, initial_point, step_size="constant", max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Classical Gradient Descent."""
    return _wrapper(objective_function, initial_point, method="gradient_descent", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

def specular_gradient(objective_function, initial_point, step_size="constant", max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Specular Gradient Descent."""
    return _wrapper(objective_function, initial_point, method="specular_gradient", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

def adam(objective_function, initial_point, step_size="constant", max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Adam optimization."""
    return _wrapper(objective_function, initial_point, method="Adam", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

def bfgs(objective_function, initial_point, step_size="Wolfe", max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Classical BFGS optimization."""
    return _wrapper(objective_function, initial_point, method="BFGS", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

def specular_bfgs(objective_function, initial_point, step_size="Wolfe", max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """BFGS optimization using Specular gradients."""
    return _wrapper(objective_function, initial_point, method="specular_BFGS", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

def stochastic_specular_gradient(objective_function, initial_point, step_size="constant", f_j=None, m=1, max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Stochastic Specular Gradient Descent."""
    kwargs['f_j'] = f_j
    kwargs['m'] = m
    return _wrapper(objective_function, initial_point, method="stochastic_specular_gradient", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

def hybrid_specular_gradient(objective_function, initial_point, step_size="constant", f_j=None, m=1, switch_iter=10, max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Hybrid Specular Gradient Descent."""
    kwargs['f_j'] = f_j
    kwargs['m'] = m
    kwargs['switch_iter'] = switch_iter
    return _wrapper(objective_function, initial_point, method="hybrid_specular_gradient", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)
# ---------------------------------------------------------
# Aliases for backwards compatibility with old `optimization`
# ---------------------------------------------------------
def gradient_method(
    f,
    x_0,
    step_size,
    h=1e-6,
    form="specular gradient",
    tol=1e-6,
    zero_tol=1e-8,
    max_iter=1000,
    f_j=None,
    m=1,
    switch_iter=2,
    record_history=True,
    print_bar=True,
):
    """Backward-compatible interface for the 1.2 specular-gradient methods."""
    del record_history  # The unified solver always records the available history.

    step_name = getattr(step_size, "step_size", step_size)
    step_options = {}
    for name in ("a", "b", "r", "user_defined_rule"):
        value = getattr(step_size, name, None)
        if value is not None:
            step_options[name] = value

    common = dict(
        objective_function=f,
        initial_point=x_0,
        step_size=step_name,
        h=h,
        zero_tol=zero_tol,
        tol=tol,
        max_iter=max_iter,
        print_bar=print_bar,
        **step_options,
    )

    if form == "specular gradient":
        return specular_gradient(**common)
    if form == "stochastic":
        return stochastic_specular_gradient(**common, f_j=f_j, m=m)
    if form == "hybrid":
        return hybrid_specular_gradient(
            **common,
            f_j=f_j,
            m=m,
            switch_iter=2 if switch_iter is None else switch_iter,
        )
    if form == "implicit":
        raise NotImplementedError(
            "The legacy implicit specular-gradient form is not available in 1.2.2."
        )
    raise TypeError(
        "Unknown form "
        f"'{form}'. Supported forms: ['specular gradient', 'stochastic', 'hybrid']"
    )

def BFGS_method(*args, **kwargs):
    """Alias for bfgs for backward compatibility."""
    return bfgs(*args, **kwargs)
