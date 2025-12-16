import inspect
from functools import wraps
import numpy as np

INTEGER_ONLY_TYPE = (int,)
ARRAY_ONLY_TYPES = (list, np.ndarray)
ARRAY_LIKE_TYPES = (int, float, np.floating, list, np.ndarray)

def check_integer_index_i(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults() 
        args_dict = bound_args.arguments
        
        if 'i' in args_dict:
            i = args_dict['i']
            
            if not isinstance(i, INTEGER_ONLY_TYPE):
                raise TypeError(
                    f"Index 'i' must be a pure integer. Got type {type(i).__name__}"
                )

            if i <= 0:
                raise ValueError(
                    f"Index 'i' must be a positive integer greater than zero (i >= 1). Got {i}"
                )
        
        return func(*args, **kwargs)
    return wrapper

def check_positive_h(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults() 
        h = bound_args.arguments.get('h')

        if h is not None and h <= 0:
            raise ValueError(f"Input 'h' must be positive. Got {h}.")
            
        return func(*args, **kwargs)
    return wrapper

def check_types_array_like_x_v(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults() 
        
        args_dict = bound_args.arguments

        if 'x' in args_dict:
            x = args_dict['x']
            if not isinstance(x, ARRAY_LIKE_TYPES):
                raise TypeError(
                    f"Input 'x' must be a scalar, list, or numpy array. Got type {type(x).__name__}"
                )

        if 'v' in args_dict:
            v = args_dict['v']
            if not isinstance(v, ARRAY_LIKE_TYPES):
                raise TypeError(
                    f"Input 'v' must be a scalar, list, or numpy array. Got type {type(v).__name__}"
                )
                
        return func(*args, **kwargs)
    return wrapper

def check_types_array_only_x(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults() 
        args_dict = bound_args.arguments

        if 'x' in args_dict:
            x = args_dict['x']
            
            if isinstance(x, (int, float, np.number)):
                raise TypeError(
                    f"Input 'x' for this function must be array-like (list or np.ndarray), not a scalar. Got {type(x).__name__}"
                )
            
            if not isinstance(x, ARRAY_ONLY_TYPES):
                raise TypeError(
                    f"Input 'x' must be a list or numpy array. Got type {type(x).__name__}"
                )
            
            try:
                if len(x) < 2:
                    raise ValueError(
                        f"Input 'x' must have length 2 or more for this function. Got length {len(x)}. "
                        "Use `specular_derivative` for 1-dimensional inputs."
                    )
            except TypeError:
                pass 

        return func(*args, **kwargs)
    return wrapper