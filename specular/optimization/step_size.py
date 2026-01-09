import math
import numpy as np
from typing import Callable, Tuple, List

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

                A number `a > 0` for the rule `h_k = a` for each `k`.
            
            * 'not_summable': float or int

                A number `a > 0` for the rule `h_k = a / sqrt{k}` for each `k`.
            
            * 'square_summable_not_summable': list or tuple

                A pair of numbers `[a, b]`, where `a > 0` and `b >= 0`, for the rule `a / (b + k)` for each `k`.
            
            * 'geometric_series': list or tuple

                A pair of numbers `[a, r]`, where `a > 0` and `0 < r < 1`, for the rule :math:`a * r^k` for each `k`.
            
            * 'user_defined': Callable

                A function that takes the current iteration `k` as input and returns the step size (float).
        """
        self.step_size, self.parameters = name, parameters

        if self.step_size == 'constant':
            if not isinstance(self.parameters, (float, int, np.floating)):
                raise TypeError(f"Invalid type: number required. Got {type(self.parameters)}")
            
            if self.parameters <= 0:
                raise ValueError(f"Invalid value: positive number required. Got {self.parameters}")
            
            self.a, self._rule = float(self.parameters), self._constant 

        elif self.step_size == 'not_summable':
            if not isinstance(self.parameters, (float, int, np.floating)):
                raise TypeError(f"Invalid type: number required. Got {type(self.parameters)}")

            if self.parameters <= 0:
                raise ValueError(f"Invalid value: positive number required. Got {self.parameters}")
            
            self.a, self._rule = float(self.parameters), self._not_summable

        elif self.step_size == 'square_summable_not_summable':
            if not isinstance(self.parameters, (tuple, list, np.ndarray)):
                raise TypeError(f"Invalid type: list/tuple required. Got {type(self.parameters)}")
            
            if len(self.parameters) != 2:
                raise ValueError(f"Invalid length: 2 parameters [a, b] required. Got {len(self.parameters)}")
            
            self.a, self.b = self.parameters[0], self.parameters[1]

            if self.a <= 0 or self.b < 0:
                raise ValueError(f"Invalid parameters: a > 0 and b >= 0 required. Got a={self.a}, b={self.b}")

            self._rule = self._square_summable_not_summable

        elif self.step_size == 'geometric_series':
            if not isinstance(self.parameters, (tuple, list, np.ndarray)):
                raise TypeError(f"Invalid type: list/tuple required. Got {type(self.parameters)}")
            
            if len(self.parameters) != 2:
                raise ValueError(f"Invalid length: 2 parameters [a, r] required. Got {len(self.parameters)}")
            
            self.a, self.r = self.parameters[0], self.parameters[1]

            if self.a <= 0 or not (0.0 < self.r < 1.0):
                raise ValueError(f"Invalid parameters: a > 0 and 0 < r < 1 required. Got a={self.a}, r={self.r}")

            self._rule = self._geometric_series

        elif self.step_size == 'user_defined':
            if not callable(self.parameters):
                raise TypeError("Invalid type: callable function required.")
            
            self._rule = self.parameters 
        
        else:
            raise ValueError(f"Invalid step size '{self.step_size}'. Options: {self.__options__}")
    
    def __call__(self, k: int) -> float:
        """
        k = 1, 2, ...
        """
        return self._rule(k)
    
    def _constant(self, k: int) -> float:
        """
        h_k = a 
        """
        return self.a
    
    def _not_summable(self, k: int) -> float:
        """
        h_k = a / sqrt{k}
        """
        return self.a / math.sqrt(k)
    
    def _square_summable_not_summable(self, k: int) -> float:
        """
        h_k = a / (b + k)
        """
        return self.a / (self.b + k)
    
    def _geometric_series(self, k: int) -> float:
        """
        h_k = a * r**k
        """
        return self.a * (self.r ** k)
    