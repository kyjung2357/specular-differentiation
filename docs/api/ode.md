
# 2.2. Ordinary differential equations

### 2.2.1 Specular Euler scheme

All functions return an instance of the `ODEResult` class that encapsulates the numerical results.

```python
import specular

def F(t, u):
    return -2*u 

specular.Euler_scheme(of_Type='1', F=F, t_0=0.0, u_0=1.0, T=2.5, h=0.1)
# Output: Running the specular Euler scheme of Type 1: 100%|██████████| 24/24 [00:00<?, ?it/s]
# Output: <specular.ode.result.ODEResult at 0x1765982d8d0>
```

To access the numerical results, call `.history()`.
It returns a tuple containing the time grid and the numerical solution.

```python
import specular

def F(t, u):
    return -2*u 

specular.Euler_scheme(of_Type=1, F=F, t_0=0.0, u_0=1.0, T=2.5, h=0.1).history()
# Output: Running the specular Euler scheme of Type 1: 100%|██████████| 24/24 [00:00<?, ?it/s]
# Output: (array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
#        1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5]),
# Output:  array([1.        , 0.8       , 0.62169432, 0.48101574, 0.37172557,
#        0.2870388 , 0.22149069, 0.17081087, 0.13166787, 0.1014624 ,
#        0.07816953, 0.06021577, 0.04638162, 0.0357239 , 0.02751427,
#        0.02119088, 0.01632056, 0.0125695 , 0.00968054, 0.00745555,
#        0.00574195, 0.00442221, 0.00340579, 0.00262299, 0.00202011,
#        0.0015558 ]))
```

To visualize the numerical results, call `.visualization()`.

```python
import specular
import numpy as np

def F(t, u):
   return -2*u 

def exact_sol(t):
    return np.exp(-2*t)

def u_0(t_0):
    return exact_sol(t_0)

specular.Euler_scheme(of_Type='1', F=F, t_0=0.0, u_0=u_0, T=2.5, h=0.1).visualization(exact_sol=exact_sol, save_path="specular-Euler-scheme-of-Type-1")
# Output: Running the specular Euler scheme of Type 1: 100%|██████████| 24/24 [00:00<?, ?it/s]
# Output: Figure saved: figures\specular-Euler-scheme-of-Type-1
```

![specular-Euler-scheme-of-Type-1](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/specular-Euler-scheme-of-Type-1.png)

To obtain the table of the numerical results, call `.table()`. 

```python
import specular
import numpy as np

def F(t, u):
    return -2*u

def exact_sol(t):
    return np.exp(-2*t)
    
def u_0(t_0):
    return exact_sol(t_0)

specular.Euler_scheme(of_Type=4, F=F, t_0=0.0, u_0=u_0, T=2.5, h=0.1).table(exact_sol=exact_sol, save_path="specular-Euler-scheme-of-type-4")
# Output: Running the specular Euler scheme of Type 4: 100%|██████████| 25/25 [00:00<?, ?it/s]
# Output: Table saved: tables\specular-Euler-scheme-of-type-4.csv
```

`.visualization()` and `.table()` are are chainable.

```python
import specular
import numpy as np

def F(t, u):
    return -2*u

def exact_sol(t):
return np.exp(-2*t)
    
def u_0(t_0):
    return exact_sol(t_0)

specular.Euler_scheme(of_Type=4, F=F, t_0=0.0, u_0=u_0, T=2.5, h=0.1).visualization(exact_sol=exact_sol).table(exact_sol=exact_sol)
# Output: Running the specular Euler scheme of Type 4: 100%|██████████| 25/25 [00:00<?, ?it/s]
```

To compute the total error of the numerical results, call `.total_error()`.
The exact solution is required.
The norm can be `max`, `l1`, or `l2`.

```python
def F(t, u):
    return -2*u 

def exact_sol(t):
    return np.exp(-2*t)

def u_0(t_0):
    return exact_sol(t_0)

specular.Euler_scheme(of_Type=5, F=F, t_0=0.0, u_0=u_0, T=10.0, h=0.1).total_error(exact_sol=exact_sol, norm='max')
# Output: Running the specular Euler scheme of Type 5: 100%|██████████| 100/100 [00:00<00:00, 300882.64it/s]
# Output: 0.0011409613137273178
```

### 2.2.2 Specular trigonometric scheme

```python
import specular

def F(t, u):
    return -2*u 

def exact_sol(t):
    return np.exp(-2*t)

def u_0(t_0):
    return exact_sol(t_0)

u_1 = exact_sol(t_0 + h)

specular.trigonometric_scheme(F=F, t_0=0.0, u_0=u_0, u_1=u_1, T=2.5, h=0.1).visualization(exact_sol=exact_sol, save_path="specular-trigonometric")
# Output: Running specular trigonometric scheme: 100%|██████████| 24/24 [00:00<?, ?it/s]
# Output: Figure saved: figures\specular-trigonometric
```

![specular-trigonometric-scheme](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/specular-trigonometric.png)

### 2.2.3 Classical schemes

The three classical schemes are available: the explicit Euler, the implicit Euler, and the Crank-Nicolson schemes.

```python
import specular
import numpy as np
import matplotlib.pyplot as plt

def F(t, u):
    return -(t*u)/(1-t**2)
def exact_sol(t):
    return np.sqrt(1 - t**2)
def u_0(t_0):
    return exact_sol(t_0)
t_0 = 0.0
T = 0.9
h = 0.05

result_EE = specular.ode.solver.classical_scheme(F=F, t_0=t_0, u_0=u_0, T=T, h=h, scheme="explicit Euler").history()
result_IE = specular.ode.solver.classical_scheme(F=F, t_0=t_0, u_0=u_0, T=T, h=h, scheme="implicit Euler").history()
result_CN = specular.ode.solver.classical_scheme(F=F, t_0=t_0, u_0=u_0, T=T, h=h, scheme="Crank-Nicolson").history()
exact_values = np.array([exact_sol(t) for t in result_EE[0]])

plt.figure(figsize=(5.5, 2.5))

plt.plot(result_EE[0], exact_values, color='black', label='Exact solution')
plt.plot(result_EE[0], result_EE[1],  marker='x', linestyle='None', markerfacecolor='none', markeredgecolor='red', label='Explicit Euler') 
plt.plot(result_IE[0], result_IE[1],  marker='x', linestyle='None', markerfacecolor='none', markeredgecolor='blue', label='Implicit Euler') 
plt.plot(result_CN[0], result_CN[1],  marker='x', linestyle='None', markerfacecolor='none', markeredgecolor='purple', label='Crank-Nicolson')

plt.xlabel(r"Time", fontsize=10)
plt.ylabel(r"Solution", fontsize=10)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0., fontsize=10)
plt.savefig('figures/classical-schemes.png', dpi=1000, bbox_inches='tight')
plt.show()
# Output: Running the explicit Euler scheme: 100%|██████████| 18/18 [00:00<?, ?it/s]
# Output: Running the implicit Euler scheme: 100%|██████████| 18/18 [00:00<?, ?it/s]
# Output: Running Crank-Nicolson scheme: 100%|██████████| 18/18 [00:00<00:00, 17988.44it/s]
```

![classical-schemes](https://raw.githubusercontent.com/kyjung2357/specular-differentiation/main/docs/figures/classical-schemes.png)

