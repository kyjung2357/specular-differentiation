import math
import numpy as np
from typing import Callable

class StepSchedule:
    """
    Step size rules for optimization methods.
    """
    _SUPPORTED_OPTIONS = {
        'constant': ['a'],
        'not_summable': ['a'],
        'square_summable_not_summable': ['a', 'b'],
        'geometric_series': ['a', 'r'],
        'user_defined': []
    }

    def __init__(
        self,
        name: str,
        *,
        a: float | int | None = None,
        b: float | int | None = None,
        r: float | int | None = None,
        user_defined_rule: Callable | None = None
    ):
        """
        The step size rules for optimization methods $x_{k+1} = x_k - h_k s_k$, where $s_k$ is the search direction and $h_k > 0$ is the step size at iteration $k >= 1$.

        Parameters:
            name (str):
                Options: ['constant', 'not_summable', 'square_summable_not_summable', 'geometric_series', 'user_defined']
            a (float | int, optional):
                Parameter `a` used in various rules.
            b (float | int, optional):
                Parameter `b` used in 'square_summable_not_summable'.
            r (float | int, optional):
                Parameter `r` used in 'geometric_series'.
            user_defined_rule (Callable, optional):
                A function that takes the current iteration `k` as input and returns the step size (float).
        
        Examples:
            >>> from specular.optimization.step_schedule import StepSchedule
            >>> 
            >>> # 'constant': h_k = a
            >>> step = StepSchedule(name='constant', a=0.5)
            >>> 
            >>> # 'not_summable' rule: h_k = a / sqrt(k)
            >>> # a = 2.0
            >>> step = StepSchedule(name='not_summable', a=2.0)
            >>> 
            >>> # 'square_summable_not_summable' rule: h_k = a / (b + k)
            >>> # a = 10, b = 2
            >>> step = StepSchedule(name='square_summable_not_summable', a=10.0, b=2.0)
            >>> 
            >>> # 'geometric_series' rule: h_k = a * r^k
            >>> # a = 1.0, r = 0.5
            >>> step = StepSchedule(name='geometric_series', a=1.0, r=0.5)
            >>> 
            >>> # 'user_defined' callable.
            >>> # Custom rule: h_k = 1 / k^2
            >>> custom_rule = lambda k: 1.0 / (k**2)
            >>> step = StepSchedule(name='user_defined', user_defined_rule=custom_rule)
        """
        self.step_size = name
        
        self.a = float(a) if a is not None else None
        self.b = float(b) if b is not None else None
        self.r = float(r) if r is not None else None
        self.user_defined_rule = user_defined_rule

        init_methods = {
            'constant': self._init_constant,
            'not_summable': self._init_not_summable,
            'square_summable_not_summable': self._init_square_summable,
            'geometric_series': self._init_geometric,
            'user_defined': self._init_user_defined
        }

        if name not in init_methods:
             raise ValueError(f"Invalid step size '{name}'. Options: {list(self._SUPPORTED_OPTIONS.keys())}")
        
        init_methods[name]()

    def __call__(self, k: int, **kwargs) -> float:
        """
        Returns the step size at iteration k.

        Accepts **kwargs for interface compatibility with LineSearch.
        """
        return self._rule(k)

    # ==== Initialization Methods ====
    def _init_constant(self):
        if self.a is None:
            raise ValueError("Parameter 'a' is required for 'constant' step schedule.")
        if self.a <= 0:
            raise ValueError(f"Invalid value: positive number required for 'a'. Got {self.a}")
        self._rule = self._calc_constant

    def _init_not_summable(self):
        if self.a is None:
            raise ValueError("Parameter 'a' is required for 'not_summable' step schedule.")
        if self.a <= 0:
            raise ValueError(f"Invalid value: positive number required for 'a'. Got {self.a}")
        self._rule = self._calc_not_summable

    def _init_square_summable(self):
        if self.a is None or self.b is None:
            raise ValueError("Parameters 'a' and 'b' are required for 'square_summable_not_summable' step schedule.")
        if self.a <= 0 or self.b < 0:
            raise ValueError(f"Invalid parameters: a > 0 and b >= 0 required. Got a={self.a}, b={self.b}")
        self._rule = self._calc_square_summable_not_summable

    def _init_geometric(self):
        if self.a is None or self.r is None:
            raise ValueError("Parameters 'a' and 'r' are required for 'geometric_series' step schedule.")
        if self.a <= 0 or not (0.0 < self.r < 1.0):
            raise ValueError(f"Invalid parameters: a > 0 and 0 < r < 1 required. Got a={self.a}, r={self.r}")
        self._rule = self._calc_geometric_series

    def _init_user_defined(self):
        if not callable(self.user_defined_rule):
            raise TypeError("Invalid type: callable function required for 'user_defined_rule'.")
        self._rule = self.user_defined_rule

    # ==== Calculation Methods ====
    def _calc_constant(self, k: int) -> float:
        """
        h_k = a 
        """
        return self.a

    def _calc_not_summable(self, k: int) -> float:
        """
        h_k = a / sqrt{k}
        """
        return self.a / math.sqrt(k)

    def _calc_square_summable_not_summable(self, k: int) -> float:
        """
        h_k = a / (b + k)
        """
        return self.a / (self.b + k)

    def _calc_geometric_series(self, k: int) -> float:
        """
        h_k = a * r**k
        """
        return self.a * (self.r ** k)
