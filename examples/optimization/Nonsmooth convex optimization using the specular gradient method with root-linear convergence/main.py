import torch
import numpy as np
from tools import repeat_experiment 

# First example
def f_1(x): 
    return np.sum([np.abs(x - i/100) + np.abs(x + i/100) - 2*i/100 for i in range(99)])

def f_1_torch(x): 
    i = torch.arange(99, device=x.device) / 100.0
    term = torch.abs(x - i) + torch.abs(x + i) - 2*i
    return torch.sum(term)

# Second example
p = 1.3
q = 1.2

def f_2(x):
    x = np.asarray(x)
    
    res = np.zeros_like(x, dtype=float)
    
    mask_high = (x >= 0.5)              
    mask_mid = (x >= 0) & (x < 0.5)    
    mask_low = (x < 0)                 
    
    res[mask_high] = 3 * (x[mask_high] - 0.5) + (0.5**q) / q
    res[mask_mid] = (np.abs(x[mask_mid])**q) / q
    res[mask_low] = (np.abs(x[mask_low])**p) / p
    
    return np.sum(res)

def f_2_torch(x):
    val_high = 3 * (x - 0.5) + (0.5**q) / q
    val_mid = (torch.abs(x)**q) / q
    val_low = (torch.abs(x)**p) / p
    
    out = torch.where(x >= 0.5, val_high, torch.where(x >= 0, val_mid, val_low))
    
    return torch.sum(out)

# Third example (Huber Loss)
delta = 0.5

def f_3(x):
    x = np.asarray(x)
    abs_x = np.abs(x)
    
    res = np.zeros_like(x, dtype=float)
    
    mask_in = (abs_x <= delta)  
    mask_out = (abs_x > delta)   
    
    res[mask_in] = 0.5 * (x[mask_in]**2)
    res[mask_out] = delta * (abs_x[mask_out] - 0.5 * delta)
    
    return np.sum(res)

def f_3_torch(x): 
    abs_x = torch.abs(x)
    
    out = torch.where(abs_x <= delta, 0.5 * (x**2), delta * (abs_x - 0.5 * delta))
    
    return torch.sum(out)

# Fourth example (j(x) = (|x|^p) / p)
p = 1.3

def f_4(x):
    x = np.asarray(x)
    return np.sum((np.abs(x)**p) / p)

def f_4_torch(x):
    return torch.sum((torch.abs(x)**p) / p)

num_runs = 100
if __name__ == '__main__':
    repeat_experiment(f_1, f_1_torch, num_runs, max_iter=1000, latex_code=True, save_name='1') # type: ignore
    repeat_experiment(f_2, f_2_torch, num_runs, max_iter=2000, latex_code=True, save_name='2') # type: ignore
    repeat_experiment(f_3, f_3_torch, num_runs, max_iter=500, latex_code=True, save_name='3') # type: ignore
    repeat_experiment(f_4, f_4_torch, num_runs, max_iter=2000, latex_code=True, save_name='4') # type: ignore
