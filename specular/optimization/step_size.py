import math
import numpy as np
from typing import Callable, Tuple

class StepSize:
    """
    Step size rules for optimization methods.
    """
    __options__ = [
        'constant',
        'not_summable',
        'square_summable_not_summable',
        'geometric_series',
        'user_defined'
    ]

    def __init__(
        self,
        name: str,
        parameters: float | np.floating | int | Tuple | list | np.ndarray | Callable
    ):
        """
        The step size rules for optimization methods:

        :math:`x_{k+1} = x_k - h_k s_k`,
        
        where :math:`s_k` is the search direction and :math:`h_k > 0` is the step size at iteration `k >= 1`.

        Parameters
        ----------
        name : str
            Options: 'constant', 'not_summable', 'square_summable_not_summable', 'geometric_series', 'user_defined'
        parameters : float | int | tuple | list | np.ndarray | Callable
            The parameters required for the selected step size rule:

            * 'constant': float or int

                A number `a > 0` for the rule :math:`h_k = a` for each `k`.
            
            * 'not_summable': float or int

                A number `a > 0` for the rule :math:`h_k = a / sqrt{k}` for each `k`.
            
            * 'square_summable_not_summable': list or tuple

                A pair of numbers `[a, b]`, where `a > 0` and `b >= 0`, for the rule :math:`h_k = a / (b + k)` for each `k`.
            
            * 'geometric_series': list or tuple

                A pair of numbers `[a, r]`, where `a > 0` and `0 < r < 1`, for the rule :math:`h_k = a * r^k` for each `k`.
            
            * 'user_defined': Callable

                A function that takes the current iteration `k` as input and returns the step size (float).
        
        Examples
        --------
        >>> from specular.optimization.step_size import StepSize
        >>> 
        >>> # 'constant': h_k = a
        >>> step = StepSize(name='constant', parameters=0.5)
        >>> 
        >>> # 'not_summable' rule: h_k = a / sqrt(k)
        >>> # a = 2.0
        >>> step = StepSize(name='not_summable', parameters=2.0)
        >>> 
        >>> # 'square_summable_not_summable' rule: h_k = a / (b + k
        >>> # a = 10, b = 2
        >>> step = StepSize(name='square_summable_not_summable', parameters=[10.0, 2.0])
        >>> 
        >>> # 'geometric_series' rule: h_k = a * r^k
        >>> # a = 1.0, r = 0.5
        >>> step = StepSize(name='geometric_series', parameters=[1.0, 0.5])
        >>> 
        >>> # 'user_defined' callable.
        >>> # Custom rule: h_k = 1 / k^2
        >>> custom_rule = lambda k: 1.0 / (k**2)
        >>> step = StepSize(name='user_defined', parameters=custom_rule)
        """
        self.step_size = name
        self.parameters = parameters

        init_methods = {
            'constant': self._init_constant,
            'not_summable': self._init_not_summable,
            'square_summable_not_summable': self._init_square_summable,
            'geometric_series': self._init_geometric,
            'user_defined': self._init_user_defined
        }

        if name not in init_methods:
             raise ValueError(f"Invalid step size '{name}'. Options: {self.__options__}")
        
        init_methods[name]()

    def __call__(self, k: int) -> float:
        """
        Returns the step size at iteration k.
        """
        return self._rule(k)

    # ==== Initialization Methods ====
    def _init_constant(self):
        if not isinstance(self.parameters, (float, int, np.floating)):
            raise TypeError(f"Invalid type: number required. Got {type(self.parameters)}")

        if self.parameters <= 0:
            raise ValueError(f"Invalid value: positive number required. Got {self.parameters}")

        self.a = float(self.parameters)
        self._rule = self._calc_constant

    def _init_not_summable(self):
        if not isinstance(self.parameters, (float, int, np.floating)):
            raise TypeError(f"Invalid type: number required. Got {type(self.parameters)}")

        if self.parameters <= 0:
            raise ValueError(f"Invalid value: positive number required. Got {self.parameters}")

        self.a = float(self.parameters)
        self._rule = self._calc_not_summable

    def _init_square_summable(self):
        if not isinstance(self.parameters, (tuple, list, np.ndarray)):
            raise TypeError(f"Invalid type: list/tuple required. Got {type(self.parameters)}")

        if len(self.parameters) != 2:
            raise ValueError(f"Invalid length: 2 parameters [a, b] required. Got {len(self.parameters)}")

        self.a, self.b = self.parameters[0], self.parameters[1]

        if self.a <= 0 or self.b < 0:
            raise ValueError(f"Invalid parameters: a > 0 and b >= 0 required. Got a={self.a}, b={self.b}")

        self._rule = self._calc_square_summable_not_summable

    def _init_geometric(self):
        if not isinstance(self.parameters, (tuple, list, np.ndarray)):
            raise TypeError(f"Invalid type: list/tuple required. Got {type(self.parameters)}")
        
        if len(self.parameters) != 2:
            raise ValueError(f"Invalid length: 2 parameters [a, r] required. Got {len(self.parameters)}")

        self.a, self.r = self.parameters[0], self.parameters[1]

        if self.a <= 0 or not (0.0 < self.r < 1.0):
            raise ValueError(f"Invalid parameters: a > 0 and 0 < r < 1 required. Got a={self.a}, r={self.r}")

        self._rule = self._calc_geometric_series

    def _init_user_defined(self):
        if not callable(self.parameters):
            raise TypeError("Invalid type: callable function required.")

        self._rule = self.parameters

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