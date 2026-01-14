import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Callable, Any
from specular.optimization.result import OptimizationResult
from specular.optimization.step_size import StepSize
import specular.jax as sjax

SUPPORTED_METHODS = ['specular gradient', 'stochastic', 'hybrid']


def _create_step_fn(step_size: StepSize) -> Callable[[Any], Any]:
    """
    Creates a JAX-compatible step size function based on the StepSize object.
    """
    rule_name = step_size.step_size
    params = step_size.parameters

    if rule_name == 'user_defined':
        if not callable(params):
            raise TypeError("For 'user_defined' step size, parameters must be a callable function.")
        
        return params

    if isinstance(params, (float, int)):
        p = jnp.array([params], dtype=float)
    else:
        p = jnp.array(params, dtype=float)

    # h_k = a
    if rule_name == 'constant':
        return lambda k: p[0]
    
    # h_k = a / sqrt(k)
    elif rule_name == 'not_summable':
        return lambda k: p[0] / jnp.sqrt(k.astype(float))

    # params: [a, b]
    # h_k = a / (b + k)
    elif rule_name == 'square_summable_not_summable':
        return lambda k: p[0] / (p[1] + k.astype(float))

    # params: [a, r]
    # h_k = a * r^k  
    elif rule_name == 'geometric_series':
        return lambda k: p[0] * jnp.power(p[1], k.astype(float))

    else:
        raise ValueError(f"JAX solver does not support step size rule: '{rule_name}'")


def gradient_method(
    f: Callable[[Any], Any],
    x_0: Any,
    step_size: StepSize,
    h: float = 1e-6,
    form: str = 'specular gradient',
    tol: float = 1e-6,
    zero_tol: float = 1e-8,
    max_iter: int = 1000,
    f_j: Callable[[Any, int], Any] | None = None,
    m: int = 1,
    seed: int = 0,
    switch_iter: int | None = 2,
    record_history: bool = True
) -> OptimizationResult:
    """
    JAX implementation of ``specular.gradient_method``.
    """
    if h <= 0:
        raise ValueError(f"Mesh size 'h' must be positive. Got {h}")

    x_0_jax = jnp.array(x_0, dtype=float)
    
    step_func = _create_step_fn(step_size)

    all_history = {}

    start_time = time.time()

    if form == 'specular gradient':
        res_x, x_history, res_f, f_history, res_k = _vector_jax(
            f, x_0_jax, step_func, h, tol, zero_tol, max_iter, record_history, k_start=1
        )

    elif form == 'stochastic':
        if f_j is None:
            raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")

        form = 'stochastic specular gradient'
        res_x, x_history, res_f, f_history, res_k = _vector_stochastic_jax(
            f, x_0_jax, step_func, h, tol, zero_tol, f_j, m, max_iter, record_history, seed, k_start=1 # type: ignore
        )

    elif form == 'hybrid':
        if f_j is None:
            raise ValueError("Component functions 'f_j' must be provided for the stochastic form.")
        
        form = 'hybrid specular gradient'
        switch_iter = switch_iter if switch_iter is not None else max_iter
        remaining_iter = max_iter - switch_iter

        # Phase 1: Deterministic
        x_1, hist_x1, _, hist_f1, k1 = _vector_jax(
            f, x_0_jax, step_func, h, tol, zero_tol, switch_iter, record_history, k_start=1
        )
        
        # Phase 2: Stochastic
        res_x, hist_x2, res_f, hist_f2, k2 = _vector_stochastic_jax(
            f, x_1, step_func, h, tol, zero_tol, f_j, m, remaining_iter, record_history, seed, k_start=k1 + 1
        )
        
        if record_history and hist_x1 is not None:
             x_history = (hist_x1, hist_x2)
             f_history = (hist_f1, hist_f2)
        else:
             x_history, f_history = None, None
        
        res_k = k1 + k2

    else:
        raise TypeError(f"Unknown form for JAX backend: '{form}'. Supported forms: {SUPPORTED_METHODS}")

    res_x.block_until_ready()
    runtime = time.time() - start_time

    final_x_hist = None
    final_f_hist = None

    if record_history:
        if isinstance(x_history, tuple):
            final_x_hist = np.concatenate([np.array(x_history[0]), np.array(x_history[1])], axis=0)
            final_f_hist = np.concatenate([np.array(f_history[0]), np.array(f_history[1])], axis=0) # type: ignore

        elif x_history is not None:
            final_x_hist = np.array(x_history)
            final_f_hist = np.array(f_history)
        
        if final_x_hist is not None:
             final_x_hist = np.insert(final_x_hist, 0, np.array(x_0_jax), axis=0)
             final_f_hist = np.insert(final_f_hist, 0, float(f(x_0_jax)), axis=0) # type: ignore
        
        all_history["variables"] = final_x_hist
        all_history["values"] = final_f_hist

    return OptimizationResult(
        method=f"JAX {form}",
        solution=np.array(res_x),
        func_val=float(res_f),
        iteration=int(res_k),
        runtime=runtime,
        all_history=all_history
    )


def _vector_jax(
    f: Callable[[Any], Any],
    x_0_jax: Any,
    step_func: Callable[[Any], Any],
    h: float,
    tol: float,
    zero_tol: float,
    max_iter: int,
    record_history: bool,
    k_start: int = 1
) -> tuple:
    """
    Vector implementation of ``specular.jax.gradient_method``
    The specular gradient method in the n-dimensional case.
    """
    def body_fun(carry, _):
        x, k, active = carry
        
        specular_gradient = sjax.gradient(f, x, h=h, zero_tol=zero_tol)
        norm = jnp.linalg.norm(specular_gradient)
        
        is_converged = norm < tol
        new_active = active & (~is_converged)
        
        safe_norm = jnp.where(norm > 0, norm, 1.0)
        
        update = step_func(k) * (specular_gradient / safe_norm) # type: ignore
        x_new = jnp.where(new_active, x - update, x)
        
        return (x_new, k + 1, new_active), (x_new, f(x_new))

    init_val = (x_0_jax, k_start, jnp.array(True))
    final_val, history = jax.lax.scan(body_fun, init_val, None, length=max_iter)
    
    res_x, res_k_final, _ = final_val
    
    iter_count = res_k_final - k_start

    if record_history:
        hist_x, hist_f = history

        return res_x, hist_x, f(res_x), hist_f, iter_count
    else:
        return res_x, None, f(res_x), None, iter_count


def _vector_stochastic_jax(
    f: Callable[[Any], Any],
    x_0_jax: list | np.ndarray,
    step_func: Callable[[Any], Any],
    h: float,
    tol: float,
    zero_tol: float,
    f_j: Callable[[Any, int], Any],
    m: int,
    max_iter: int, 
    record_history: bool,
    seed: int,
    k_start: int = 1
) -> tuple:
    """
    Vector implementation of ``specular.jax.gradient_method``.
    The stochastic specular gradient method in the n-dimensional case.
    """
    def body_fun(carry, _):
        x, k, active, key = carry
        
        key, subkey = jax.random.split(key)

        # A random index j is selected at each iteration
        j = jax.random.randint(subkey, (), 0, m)
        
        def f_comp(val): return f_j(val, j) # type: ignore
        
        specular_gradient = sjax.gradient(f_comp, x, h=h, zero_tol=zero_tol)
        norm = jnp.linalg.norm(specular_gradient)
        
        is_converged = norm < tol
        new_active = active & (~is_converged)
        
        safe_norm = jnp.where(norm > 0, norm, 1.0)
        
        update = step_func(k) * (specular_gradient / safe_norm) # type: ignore
        x_new = jnp.where(new_active, x - update, x)
        
        return (x_new, k + 1, new_active, key), (x_new, f(x_new))

    rng_key = jax.random.PRNGKey(seed)
    init_val = (x_0_jax, k_start, jnp.array(True), rng_key)
    
    final_val, history = jax.lax.scan(body_fun, init_val, None, length=max_iter)
    
    res_x, res_k_final, _, _ = final_val
    
    iter_count = res_k_final - k_start

    if record_history:
        hist_x, hist_f = history
        return res_x, hist_x, f(res_x), hist_f, iter_count
    else:
        return res_x, None, f(res_x), None, iter_count