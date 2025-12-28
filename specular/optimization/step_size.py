import math
import numpy as np
from typing import Callable, Tuple, List

SUPPORTED_STEP_SIZES = ['constant', 'not_summable', 'square_summable_not_summable', 'geometric_series', 'user_defined']

class StepSize:
    def __init__(
        self, 
        step_size: str, 
        parameters: float | np.floating | int | Tuple | list | np.ndarray | Callable
    ):
        self.step_size = step_size 
        self.parameters = parameters

        if self.step_size == 'constant':
            if not isinstance(self.parameters, (float, int, np.floating)):
                raise TypeError(f"For 'constant' step size, parameters must be a number. Got {type(self.parameters)}")
            
            if self.parameters <= 0:
                raise ValueError(f"Step size must be positive.")
            
            self.a = float(self.parameters)
            self._rule = self._constant 

        elif self.step_size == 'not_summable':
            if not isinstance(self.parameters, (float, int, np.floating)):
                raise TypeError(f"For 'not_summable' step size, parameters must be a number. Got {type(self.parameters)}")

            if self.parameters <= 0:
                raise ValueError(f"Step size must be positive.")
            
            self.a = float(self.parameters)
            self._rule = self._not_summable

        elif self.step_size == 'square_summable_not_summable':
            if not isinstance(self.parameters, (tuple, list, np.ndarray)):
                raise TypeError(f"For 'square_summable_not_summable' step size, parameters must be a list. Got {type(self.parameters)}")
            
            if len(self.parameters) != 2:
                raise ValueError(f"For 'square_summable_not_summable' step size, only two parameters [a, b] are required for a/(b + k). Got {self.parameters}")
            
            self.a = self.parameters[0]
            self.b = self.parameters[1]

            if self.a <= 0 or self.b <= 0:
                raise ValueError(f"For 'square_summable_not_summable' step size, a and b are must be positive for a/(b + k). Got a: {self.a} and b: {self.b}")

            self._rule = self._square_summable_not_summable

        elif self.step_size == 'geometric_series':
            if not isinstance(self.parameters, (tuple, list, np.ndarray)):
                raise TypeError(f"For 'geometric_series' step size, parameters must be a list. Got {type(self.parameters)}")
            
            if len(self.parameters) != 2:
                raise ValueError(f"For 'geometric_series' step size, only two parameters [a, r] are required for a * r^k. Got {self.parameters}")
            
            self.a = self.parameters[0]
            self.r = self.parameters[1]

            if self.a <= 0:
                raise ValueError(f"For 'geometric_series' step size, the initial step size 'a' must be positive. Got {self.a}")

            if not (0.0 < self.r < 1.0):
                raise ValueError(f"For 'geometric_series' to be always positive and converging, the ratio 'r' must be in (0, 1). Got {self.r}")

            self._rule = self._geometric_series

        elif self.step_size == 'user_defined':
            if not callable(self.parameters):
                raise TypeError("For 'user_defined' mode, parameters must be a function.")
            
            self._rule = self.parameters 
        
        else:
            raise ValueError(f"Unknown step size '{self.step_size}'. Supported forms: {SUPPORTED_STEP_SIZES}")
    
    def __call__(self, k: int) -> float:
        """
        k = 0, 1, 2, ...
        """
        return self._rule(k)
    
    def _constant(self, k: int) -> float:
        return self.a
    
    def _not_summable(self, k: int) -> float:
        return self.a / math.sqrt(k)
    
    def _square_summable_not_summable(self, k: int) -> float:
        return self.a / (self.b + k)
    
    def _geometric_series(self, k: int) -> float:
        return self.a * (self.r ** k)
    