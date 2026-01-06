import os
import matplotlib.pyplot as plt

class OptimizationResult:
    def __init__(self, solution, func_val, iteration, scheme, history=None):
        self.x = solution 
        self.f = func_val
        self.k = iteration
        self.scheme = scheme
        self.history = history
    
    def __repr__(self):
        return (
            f"[{self.scheme}]\n"
            f" solution: {self.x}\n"
            f" func val: {self.f}\n"
            f"iteration: {self.k}"
        )