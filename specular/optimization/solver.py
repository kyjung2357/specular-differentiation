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

def specular_modified_bfgs(objective_function, initial_point, step_size="Wolfe", max_iter=1000, tol=1e-6, print_bar=True, **kwargs):
    """Specular Modified BFGS optimization (with custom y_k logic)."""
    return _wrapper(objective_function, initial_point, method="specular_modified_BFGS", step_size=step_size, max_iter=max_iter, tol=tol, print_bar=print_bar, options=kwargs)

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
def gradient_method(*args, **kwargs):
    """Alias for gradient_descent for backward compatibility."""
    return gradient_descent(*args, **kwargs)

def BFGS_method(*args, **kwargs):
    """Alias for bfgs for backward compatibility."""
    return bfgs(*args, **kwargs)
