import os
import matplotlib.pyplot as plt

class OptimizationResult:
    def __init__(self, solution, objective_func_val, iteration, history=None):
        self.x = solution 
        self.f = objective_func_val
        self.k = iteration
        self.history = history
    
    def __repr__(self):
        return (
            f" solution: {self.x}\n"
            f" func val: {self.f}\n"
            f"iteration: {self.k}"
        )