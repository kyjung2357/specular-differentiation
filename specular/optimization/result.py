
class OptimizationResult:
    def __init__(self, solution, objective_function_value, iteration, history=None):
        self.x = solution 
        self.fun_val = objective_function_value
        self.k = iteration
        self.history = history