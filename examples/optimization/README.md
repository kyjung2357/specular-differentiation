# Optimization examples

`one_dimension.py` implements the scalar Elastic Net and sum-of-absolute-values
experiments from K. Jung and J. Oh, *Nonsmooth convex optimization using the
specular gradient method with root-linear convergence*,
[arXiv:2412.20747](https://arxiv.org/abs/2412.20747).
The objectives, methods, settings, plotting, and exports are in this one file.
The Elastic Net default is `lambda_1 = 1.0`.

## Run

From the repository root, install the local package and optional comparison
dependencies:

```bash
pip install -e .
pip install matplotlib torch
python examples/optimization/one_dimension.py
```

The examples require `specular-differentiation` 1.3.1 or newer. To run SPEG
without PyTorch:

```bash
pip install matplotlib
python examples/optimization/one_dimension.py --speg-only
```

```bash
python examples/optimization/one_dimension.py --example elastic-net
python examples/optimization/one_dimension.py --example absolute-sum
python examples/optimization/one_dimension.py --example elastic-net --lambda1 100 --lambda2 1
python examples/optimization/one_dimension.py --backend numba
python examples/optimization/one_dimension.py --output-dir my-results
```

## Settings

Edit the settings at the top of `one_dimension.py`, then run it again:

```python
lambda_1 = 1.0
lambda_2 = 0.5
m = 100
num_runs = 100
max_iter = 1000
absolute_sum_max_iter = 50

example = "both"
tune_learning_rates = True
backend = "numpy"
```

`num_runs` is the number of independent initial points for each objective.
`max_iter` sets the Elastic Net update count, and `absolute_sum_max_iter`
sets the sum-of-absolute-values update count. Both use 100 initial points by
default: Elastic Net stops at `k = 1000`, and the absolute sum stops at
`k = 50`. Including the initial point, their saved trajectories have 1001
and 51 values per run, respectively. The random seeds are also settings in
the same file.
Optional command-line arguments override those settings for one execution:

| Option | Default | Meaning |
| --- | --- | --- |
| `--example` | `both` | `elastic-net`, `absolute-sum`, or `both` |
| `--trials` | `100` | Shared test starts for each method |
| `--updates` | per-objective settings | Positive number of updates; when supplied, overrides the budget for every selected objective |
| `--seed` | `20260905` | Initial-point seed |
| `--data-seed` | `20260907` | Elastic Net design seed |
| `--m` | `100` | Elastic Net observations; the variable remains scalar |
| `--lambda1` | `1` | Elastic Net absolute-value penalty |
| `--lambda2` | `0.5` | Elastic Net quadratic penalty |
| `--backend` | `numpy` | `numpy`, `numba`, or `jax` for `specular.scaled_mean` |
| `--speg-only` | off | Skip GD, Adam, and tuning; no PyTorch dependency |
| `--tune` / `--no-tune` | on | Search baseline learning rates / use fixed reference rates |
| `--output-dir` | script directory | Destination for `figures/` and `results/` |

The Elastic Net uses independent standard normal observations and an all-ones
response. Its default minimizer is zero and its minimum is 0.5. Test starts
are uniform on `[-4, 4]`. The absolute sum is
`sum(abs(x-i/100) + abs(x+i/100) for i in range(100))`, with minimizer zero,
minimum 99, and test starts uniform on `[-1, 1]`.

SPEG combines analytic one-sided derivatives through `specular.scaled_mean`
and uses the package's `specular_gradient` iteration with geometric step lengths
`t0 * 2**(-k)`, where the default is `t0=3` for Elastic Net and `t0=0.5` for
the absolute sum. If changed Elastic Net settings move the minimizer, the
runner enlarges `t0` when needed to cover every possible start in `[-4, 4]`.
The package numbers steps from 1, so its arguments are `a=2*t0` and `r=0.5`.
The zero gradient tolerance is for the fixed-update comparison. After a zero
derivative or a step that cannot change the point at working precision, the
last point is retained for all remaining indices. The same applies when a
long run reaches the float64 range limit for the geometric step. Metadata records the actual
update count and stop reason for every run.
The convergence bound assumes a convex scalar objective attaining its minimum
and `dist(x0, minimizers) <= 2*t0`. The runner checks this initialization
condition and records the chosen `t0` and minimizer in metadata.

`backend` selects the NumPy, Numba, or JAX calculation kernel for
`specular.scaled_mean`. Install the selected optional backend before using it.
From the repository root, use `pip install -e ".[numba]"` or
`pip install -e ".[jax]"`.
The iteration and trajectory storage use Python and NumPy; changing this
setting does not compile the entire optimizer. SPEG uses double precision,
with JAX's 64-bit mode enabled when selected. The GD and Adam comparisons use
PyTorch on the CPU in double precision for every backend setting.

GD is PyTorch SGD without momentum. Adam uses its standard betas and epsilon.
By default, both methods select a learning rate for the configured objective
and update count. The search uses 32 independent starts, the paper's candidate
grids, and schedules `alpha0 * (k + 1)**(-p)` with `p` equal to `0`, `0.5`, or
`1`. Among candidates with no nonfinite runs, the smallest mean final objective
gap determines the selection, with deterministic tie-breaking for equal or
very small gaps. This is a finite search, not a guarantee of the best possible
learning rate. The chosen schedule is then fixed for the test runs.

The default tuning seed is `20260906`. If the test seed is `20260906`, the
tuning seed becomes `20260908`; the chosen seed is recorded in metadata.
Test starts are not used for selection.

For the default configuration, `--no-tune` reuses the following rates selected
with `lambda_1 = 1.0` and the indicated update budgets:

| Objective | Updates | GD `alpha0` | GD `p` | Adam `alpha0` | Adam `p` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Elastic Net | 1000 | `0.3` | `1` | `2/3` | `1` |
| Sum of absolute values | 50 | `1/200` | `1` | `0.3` | `1` |

These fixed rates are stored in `REFERENCE_RATES` in the script. Leave tuning
enabled when changing the objective or update count; `--no-tune` does not
adapt the rates to new settings:

```bash
python examples/optimization/one_dimension.py --no-tune
```

## Outputs

- `figures/elastic_net.pdf` and `.png`: Elastic Net comparison.
- `figures/absolute_sum.pdf` and `.png`: absolute-sum comparison.
- `results/summary.csv`: raw median errors and each objective's update budget.
- `results/summary.tex`: LaTeX table of the same errors, rounded to three significant digits.
- `results/trajectories.npz`: full numerical paths and starts; by default, each method has arrays of shape `(100, 1001)` for Elastic Net and `(100, 51)` for the absolute sum.
- `results/elastic_data.npz`: the fixed Elastic Net design and response.
- `results/metadata.json`: experiment settings and runtime information.
- `results/selected_parameters.csv`: baseline rates and update budgets used in the run.
- `results/tuning.csv`: all tuning candidates, when tuning is enabled.

Each figure has current-iterate distance on the left and best objective gap
on the right. Lines show medians; shaded bands show interquartile ranges.
The plotting floor of `1e-16` does not change the saved raw results. Stable
gap formulas avoid cancellation near the known minimum.

The LaTeX table contains only the objectives and methods run, with a separate
`Updates k` column so the two experiments' budgets are explicit.
Load `booktabs` in the document preamble and include the generated table:

```latex
% Preamble
\usepackage{booktabs}

% Document body; adjust the path relative to your main .tex file.
\begin{table}[tbp]
  \centering
  \input{examples/optimization/results/summary.tex}
\end{table}
```

Use a separate `--output-dir` for each configuration you want to keep. Each
successful run replaces its results and removes figures and Elastic Net data
from objectives excluded from that run. Baseline parameter and tuning CSVs
are omitted when those stages are skipped. Each output folder therefore
contains the selected experiment's generated files.
