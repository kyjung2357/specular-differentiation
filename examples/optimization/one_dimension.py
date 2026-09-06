"""Run the one-dimensional SPEG examples of Jung and Oh.

Run from a repository checkout:
    python examples/optimization/one_dimension.py
    python examples/optimization/one_dimension.py --speg-only
    python examples/optimization/one_dimension.py --lambda1 100

Edit the settings below, then run this file. GD and Adam learning rates
are selected on independent training starts before the comparison.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import cached_property
import importlib.metadata
import json
import operator
from pathlib import Path
import platform

import numpy as np
import specular
from specular.optimization import specular_gradient


# Settings: edit this block, or override individual values on the command line.
lambda_1 = 1.0       # Elastic Net absolute-value penalty
lambda_2 = 0.5       # Elastic Net quadratic penalty
m = 100              # Number of Elastic Net observations
num_runs = 100       # Number of independent test initial points
max_iter = 1000      # Elastic Net updates per run
absolute_sum_max_iter = 50  # Sum of absolute values updates per run

example = "both"     # "both", "elastic-net", or "absolute-sum"
tune_learning_rates = True
speg_only = False     # True skips GD/Adam and needs no PyTorch
backend = "numpy"    # "numpy", "numba", or "jax" for specular calculations
seed = 20260905       # Test initial points
data_seed = 20260907  # Fixed Elastic Net data
output_dir = Path(__file__).resolve().parent

PLOT_FLOOR = 1e-16
REFERENCE_RATES = {
    "elastic_net": {"GD": 0.3, "Adam": 2 / 3},
    "absolute_sum": {"GD": 1 / 200, "Adam": 0.3},
}


@dataclass(frozen=True)
class ElasticNet:
    """E(x) = mean((a*x-b)**2)/2 + lambda2*x**2/2 + lambda1*abs(x).

    ``a`` and ``b`` are finite vectors of equal positive length; the two
    regularization parameters are nonnegative. The arrays are copied so a
    caller cannot accidentally change a problem while an experiment runs.
    """

    symbol = "E"

    a: np.ndarray
    b: np.ndarray
    lambda1: float
    lambda2: float

    def __post_init__(self):
        for name in ("a", "b"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a nonempty finite vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.a.shape != self.b.shape:
            raise ValueError("a and b must have the same number of observations")
        if (not np.isfinite([self.lambda1, self.lambda2]).all()
                or min(self.lambda1, self.lambda2) < 0):
            raise ValueError("lambda1 and lambda2 must be finite and nonnegative")

    @property
    def m(self):
        return self.a.size

    @cached_property
    def curvature(self):
        return float(np.mean(self.a ** 2) + self.lambda2)

    @cached_property
    def correlation(self):
        return float(np.mean(self.a * self.b))

    @cached_property
    def constant(self):
        return float(np.mean(self.b ** 2) / 2)

    @cached_property
    def minimizer(self):
        """Return a minimizer (zero also covers a constant objective)."""
        if self.curvature == 0 or abs(self.correlation) <= self.lambda1:
            return 0.0
        return float(np.sign(self.correlation)
                     * (abs(self.correlation) - self.lambda1) / self.curvature)

    @property
    def minimum(self):
        return self.constant - self.curvature * self.minimizer ** 2 / 2

    def objective(self, x):
        x = np.asarray(x, dtype=np.float64)
        residual = x[..., None] * self.a - self.b
        return (np.mean(residual ** 2, axis=-1) / 2
                + self.lambda2 * x ** 2 / 2 + self.lambda1 * np.abs(x))

    def gap(self, x):
        """Evaluate E(x)-E* without subtracting nearly equal objective values."""
        x = np.asarray(x, dtype=np.float64)
        c = self.correlation
        if abs(c) <= self.lambda1:
            return (self.curvature * x ** 2 / 2
                    + (self.lambda1 - c) * np.maximum(x, 0)
                    + (self.lambda1 + c) * np.maximum(-x, 0))
        return (self.curvature * (x - self.minimizer) ** 2 / 2
                + 2 * self.lambda1 * np.maximum(-np.sign(c) * x, 0))

    def slopes(self, x):
        """Return the left and right derivatives, including the kink at zero."""
        x = np.asarray(x, dtype=np.float64)
        base = self.curvature * x - self.correlation
        return (base + np.where(x <= 0, -self.lambda1, self.lambda1),
                base + np.where(x < 0, -self.lambda1, self.lambda1))

    def analytic_gradient(self, x):
        """Raw specular derivative callback for ``specular_gradient``."""
        return specular.scaled_mean(*self.slopes(x))

    @cached_property
    def torch_data(self):
        import torch

        return (torch.tensor(self.a, dtype=torch.float64),
                torch.tensor(self.b, dtype=torch.float64))

    def torch_objective(self, x):
        """Differentiable PyTorch form for the optional CPU comparisons."""
        a, b = self.torch_data
        residual = x[..., None] * a - b
        return (residual.square().mean(dim=-1) / 2
                + self.lambda2 * x.square() / 2 + self.lambda1 * x.abs())


def random_problem(m=None, seed=None, lambda1=None, lambda2=None):
    """Draw one fixed standard-normal design, with b equal to an all-one vector."""
    # Omitted arguments follow the editable settings at the top of this file.
    m = globals()["m"] if m is None else m
    seed = data_seed if seed is None else seed
    lambda1 = lambda_1 if lambda1 is None else lambda1
    lambda2 = lambda_2 if lambda2 is None else lambda2
    if isinstance(m, (bool, np.bool_)):
        raise ValueError("m must be a positive integer")
    try:
        m = operator.index(m)
    except TypeError as exc:
        raise ValueError("m must be a positive integer") from exc
    if m < 1:
        raise ValueError("m must be a positive integer")
    a = np.random.default_rng(seed).standard_normal(m)
    return ElasticNet(a, np.ones(m), lambda1, lambda2)


A = np.arange(100, dtype=np.float64) / 100
KNOTS = np.sort(np.r_[A, -A])
A.setflags(write=False)
KNOTS.setflags(write=False)


class AbsoluteSum:
    """F(x) = sum(|x-i/100| + |x+i/100| for i in range(100))."""

    symbol = "F"
    minimizer = 0.0
    minimum = 99.0

    @staticmethod
    def objective(x):
        return np.abs(np.asarray(x, dtype=np.float64)[..., None] - KNOTS).sum(axis=-1)

    @staticmethod
    def gap(x):
        """Evaluate F(x)-99 from nonnegative terms, preserving tiny errors."""
        radius = np.abs(np.asarray(x, dtype=np.float64))
        value = np.zeros_like(radius)
        for a in A:
            value += np.maximum(radius - a, 0.0)
        return 2 * value

    @staticmethod
    def slopes(x):
        left = 2 * np.searchsorted(KNOTS, x, side="left") - len(KNOTS)
        right = 2 * np.searchsorted(KNOTS, x, side="right") - len(KNOTS)
        return left.astype(np.float64), right.astype(np.float64)

    def analytic_gradient(self, x):
        """Raw specular derivative callback for ``specular_gradient``."""
        return specular.scaled_mean(*self.slopes(x))

    @cached_property
    def torch_data(self):
        import torch

        return torch.tensor(KNOTS, dtype=torch.float64)

    def torch_objective(self, x):
        """Differentiable PyTorch form for the optional CPU comparisons."""
        return (x[..., None] - self.torch_data).abs().sum(dim=-1)


def run_speg(problem, starts, updates, t0):
    """Return package trajectories, retaining the final point after a stop."""
    points = np.empty((len(starts), updates + 1), dtype=np.float64)
    stops = []
    # The package's 0.5**n is zero beyond the float64 subnormal range.
    precision_limit = np.finfo(np.float64).nmant - np.finfo(np.float64).minexp
    for row, x0 in enumerate(starts):
        # The paper starts at k=0; the package's a*r**n starts at n=1.
        result = specular_gradient(
            problem.objective,
            initial_point=float(x0),
            gradient=problem.analytic_gradient,
            step_size="geometric_series",
            a=2 * t0,
            r=0.5,
            max_iter=min(updates, precision_limit),
            tol=0.0,
        )
        if result.stop_reason not in {
            "max_iter reached", "gradient norm below tolerance",
            "step does not change the point at working precision",
        }:
            raise RuntimeError(f"SPEG run {row} failed: {result.stop_reason}")
        history = result.history["variables"]
        points[row, :len(history)] = history
        points[row, len(history):] = result.solution
        reason = result.stop_reason
        if reason == "max_iter reached" and updates > precision_limit:
            reason = "geometric step underflows at working precision"
        stops.append({"run": row, "updates": result.iteration, "reason": reason})
    return points, stops


def run_baseline(problem, method, starts, updates, learning_rate, decay_power=1.0):
    """Apply PyTorch SGD or Adam to independent scalar starts in float64."""
    import torch

    x = torch.tensor(starts, dtype=torch.float64, requires_grad=True)
    if method == "GD":
        optimizer = torch.optim.SGD(
            [x], lr=learning_rate, momentum=0, weight_decay=0, foreach=False,
        )
    elif method == "Adam":
        optimizer = torch.optim.Adam(
            [x], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, foreach=False,
        )
    else:
        raise ValueError(f"unknown baseline: {method}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda k: (k + 1) ** (-decay_power),
    )
    points = np.empty((len(starts), updates + 1), dtype=np.float64)
    points[:, 0] = starts
    for k in range(updates):
        optimizer.zero_grad(set_to_none=True)
        # Sum, not mean: each scalar start keeps the stated learning rate.
        problem.torch_objective(x).sum().backward()
        optimizer.step()
        scheduler.step()
        points[:, k + 1] = x.detach().numpy()
    return points


def rate_grid(name, problem):
    if name == "absolute_sum":
        return (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3,
                3e-3, 0.005, 0.01, 0.03, 0.1, 0.3, 1.0)
    base = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 2/3, 1., 3., 10.)
    return sorted(set(base) | {rate / max(1., problem.lambda1) for rate in base})


def tune_baseline(problem, method, starts, updates, rates):
    """Use the manuscript's rate grid and deterministic tie-breaking rule."""
    records = []
    for rate in rates:
        for power in (0.0, 0.5, 1.0):
            points = run_baseline(problem, method, starts, updates, rate, power)
            with np.errstate(over="ignore", invalid="ignore"):
                gaps = problem.gap(points)
                gaps = np.where(np.isfinite(gaps), gaps, np.inf)
                records.append({
                    "method": method, "updates": updates,
                    "learning_rate": rate, "decay_power": power,
                    "nonfinite_runs": int(np.sum(~np.isfinite(points).all(axis=1))),
                    "mean_current_gap": float(np.mean(gaps[:, -1])),
                    "mean_path_gap": float(np.mean(gaps[:, 1:])),
                    "mean_best_gap": float(np.mean(np.min(gaps, axis=1))),
                })
    valid = [row for row in records if row["nonfinite_runs"] == 0
             and np.isfinite(row["mean_current_gap"])
             and np.isfinite(row["mean_path_gap"])]
    if not valid:
        raise RuntimeError(f"No finite learning-rate candidate for {method}.")
    chosen = min(valid, key=lambda row: (
        row["nonfinite_runs"], max(row["mean_current_gap"], 1e-24),
        row["mean_path_gap"], row["mean_best_gap"],
        row["learning_rate"], row["decay_power"],
    ))
    return {key: chosen[key] for key in ("learning_rate", "decay_power")}, records


def error_labels(symbol):
    """Use the manuscript's objective and minimizer notation in figures and tables."""
    return (rf"$|x_k-x_{{{symbol}}}^\ast|$",
            rf"${symbol}_k^{{\mathrm{{best}}}}-{symbol}^\ast$")


def plot(problem, paths, destination):
    """Match the typography and 5.125-inch width of the scalar ODE examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    plt.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.axisbelow": True,
        "lines.dashed_pattern": (3.7, 1.6),
        "lines.dash_capstyle": "butt",
        "lines.scale_dashes": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    })
    figure, axes = plt.subplots(1, 2, figsize=(5.125, 1.76))
    styles = {"SPEG": ("#ef3b2c", "-"),
              "GD": ("#238b45", "--"),
              "Adam": ("#08519c", "-.")}
    updates = next(iter(paths.values())).shape[1] - 1
    indices = np.arange(updates + 1)
    for method, points in paths.items():
        series = (np.abs(points - problem.minimizer),
                  np.minimum.accumulate(problem.gap(points), axis=1))
        color, line_style = styles[method]
        for ax, data in zip(axes, series, strict=True):
            q25, median, q75 = np.quantile(data, [0.25, 0.5, 0.75], axis=0)
            # Apply the display floor after computing statistics; raw data stay intact.
            ax.plot(indices, np.maximum(median, PLOT_FLOOR), color=color,
                    linestyle=line_style, linewidth=1.1, label=method)
            ax.fill_between(indices, np.maximum(q25, PLOT_FLOOR),
                            np.maximum(q75, PLOT_FLOOR), color=color, alpha=0.12)
    for ax, ylabel in zip(axes, error_labels(problem.symbol), strict=True):
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Updates $k$")
        ax.set_xlim(0, updates)
        ax.set_xscale("symlog", linthresh=1)
        powers = range(len(str(updates)))
        ticks = [0] + [10**power for power in powers]
        labels = [r"$0$", r"$1$"] + [
            r"$10$" if power == 1 else rf"$10^{{{power}}}$"
            for power in range(1, len(str(updates)))
        ]
        if updates not in ticks:
            if ticks[-1] > 1 and updates / ticks[-1] < 1.5:
                ticks.pop()
                labels.pop()
            ticks.append(updates)
            labels.append(rf"${updates}$")
        ax.set_xticks(ticks, labels)
        ax.set_yscale("log")
        ax.set_ylim(bottom=PLOT_FLOOR / 3)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.grid(color="0.85", linewidth=0.4, which="major")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles, labels, loc="center right", bbox_to_anchor=(0.995, 0.6),
        ncol=1, handlelength=1.7, handletextpad=0.6, borderpad=0.4,
        frameon=True, facecolor="white", framealpha=1.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_linewidth(0.6)
    figure.subplots_adjust(left=0.11, right=0.79, bottom=0.25, top=0.95, wspace=0.56)
    figure.savefig(destination.with_suffix(".pdf"), dpi=300)
    figure.savefig(destination.with_suffix(".png"), dpi=300)
    plt.close(figure)


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_tex(path, rows):
    """Write a booktabs table for one objective; its caption supplies the budget."""
    if not rows:
        path.unlink(missing_ok=True)
        return
    if len({row["example"] for row in rows}) != 1:
        raise ValueError("Each LaTeX table must contain exactly one objective.")

    def number(value):
        if value == 0:
            return "$0$"
        mantissa, exponent = f"{value:.2e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"

    symbol = {"elastic_net": ElasticNet.symbol,
              "absolute_sum": AbsoluteSum.symbol}[rows[0]["example"]]
    distance_label, gap_label = error_labels(symbol)
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        f"Method & Median {distance_label} & Median {gap_label}" + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['method']} & "
            f"{number(row['median_distance'])} & "
            f"{number(row['median_best_gap'])}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("both", "elastic-net", "absolute-sum"), default=example)
    parser.add_argument("--trials", type=int, default=num_runs)
    parser.add_argument("--updates", type=int, default=None,
                        help="override the update count for all selected objectives")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--data-seed", type=int, default=data_seed)
    parser.add_argument("--m", type=int, default=m)
    parser.add_argument("--lambda1", type=float, default=lambda_1)
    parser.add_argument("--lambda2", type=float, default=lambda_2)
    parser.add_argument("--speg-only", action="store_true", default=speg_only)
    parser.add_argument("--tune", action=argparse.BooleanOptionalAction, default=None,
                        help="select rates on independent starts (default: settings block)")
    parser.add_argument("--backend", choices=("numpy", "numba", "jax"), default=backend)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    args = parser.parse_args(argv)
    if args.updates is None:
        args.updates = max_iter
        args.absolute_sum_updates = absolute_sum_max_iter
    else:
        args.absolute_sum_updates = args.updates
    for name in ("trials", "updates", "absolute_sum_updates", "m", "seed", "data_seed"):
        value = getattr(args, name)
        minimum = 0 if name in {"seed", "data_seed"} else 1
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            parser.error(f"{name} must be an integer >= {minimum}")
    if not np.isfinite([args.lambda1, args.lambda2]).all() or min(args.lambda1, args.lambda2) < 0:
        parser.error("lambda1 and lambda2 must be finite and nonnegative")
    if args.example not in {"both", "elastic-net", "absolute-sum"}:
        parser.error("example must be both, elastic-net, or absolute-sum")
    if args.backend not in {"numpy", "numba", "jax"}:
        parser.error("backend must be numpy, numba, or jax")
    if args.speg_only and args.tune:
        parser.error("--tune is for GD/Adam and cannot be combined with --speg-only")
    if args.tune is None:
        args.tune = tune_learning_rates and not args.speg_only
    args.output_dir = Path(args.output_dir)
    return args


def tuning_seed(test_seed):
    """Keep the paper's default training seed distinct from the test seed."""
    return 20260906 if test_seed != 20260906 else 20260908


def main(argv=None):
    args = parse_args(argv)
    if args.backend == "jax":
        import jax
        jax.config.update("jax_enable_x64", True)
    specular.set_backend(args.backend)
    if not args.speg_only:
        try:
            import torch
        except ImportError:
            raise SystemExit("Install torch for GD/Adam, or run with --speg-only.") from None
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
    experiments = []
    if args.example in ("both", "elastic-net"):
        problem = random_problem(args.m, args.data_seed, args.lambda1, args.lambda2)
        # Retain the paper's t0=3 when valid; enlarge it for shifted minimizers.
        t0 = max(3., (4. + abs(problem.minimizer)) / 2)
        experiments.append(("elastic_net", problem, 4., t0, args.updates))
    if args.example in ("both", "absolute-sum"):
        experiments.append(("absolute_sum", AbsoluteSum(), 1., 0.5, args.absolute_sum_updates))
    output = args.output_dir.resolve()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    summaries, parameters, candidates = [], [], []
    trajectories, stopping, problem_settings = {}, {}, {}
    for name, problem, radius, t0, updates in experiments:
        print(f"Running {name}: {args.trials} starts, {updates} updates", flush=True)
        starts = np.random.default_rng(args.seed).uniform(-radius, radius, args.trials)
        problem_settings[name] = {"minimizer": problem.minimizer,
                                  "minimum": problem.minimum,
                                  "initial_radius": radius, "t0": t0,
                                  "updates": updates}
        if np.any(np.abs(starts - problem.minimizer) > 2 * t0):
            raise ValueError("The initial points exceed the SPEG distance budget.")
        trajectories[f"{name}_starts"] = starts
        if name == "elastic_net":
            np.savez_compressed(results / "elastic_data.npz", A=problem.a[:, None], b=problem.b)
        speg, stopping[name] = run_speg(problem, starts, updates, t0)
        paths = {"SPEG": speg}
        for method in (() if args.speg_only else ("GD", "Adam")):
            setting = {"learning_rate": REFERENCE_RATES[name][method], "decay_power": 1.0}
            if args.tune:
                print(f"  Tuning {method} on 100 independent starts", flush=True)
                training = np.random.default_rng(tuning_seed(args.seed)).uniform(-radius, radius, 100)
                trajectories[f"{name}_tuning_starts"] = training
                setting, rows = tune_baseline(problem, method, training, updates, rate_grid(name, problem))
                candidates.extend({"example": name, **row} for row in rows)
            parameters.append({"example": name, "method": method, "updates": updates, **setting})
            paths[method] = run_baseline(problem, method, starts, updates, **setting)
        for method, points in paths.items():
            if not np.isfinite(points).all():
                raise RuntimeError(f"{name}/{method} has nonfinite iterates; try --tune.")
            gap = problem.gap(points)
            if not np.isfinite(gap).all():
                raise RuntimeError(f"{name}/{method} has nonfinite objective gaps.")
            row = {"example": name, "method": method, "updates": updates,
                   "median_distance": float(np.median(np.abs(points[:, -1] - problem.minimizer))),
                   "median_best_gap": float(np.median(np.min(gap, axis=1)))}
            summaries.append(row)
            trajectories[f"{name}_{method}"] = points
            print(f"  {method:4s}  distance={row['median_distance']:.8e}  best gap={row['median_best_gap']:.8e}", flush=True)
        plot(problem, paths, output / "figures" / name)
    write_csv(results / "summary.csv", summaries)
    for name in ("elastic_net", "absolute_sum"):
        write_summary_tex(results / f"{name}_summary.tex",
                          [row for row in summaries if row["example"] == name])
    (results / "summary.tex").unlink(missing_ok=True)
    write_csv(results / "selected_parameters.csv", parameters)
    write_csv(results / "tuning.csv", candidates)
    np.savez_compressed(results / "trajectories.npz", **trajectories)
    versions = {name: importlib.metadata.version(name) for name in
                ("specular-differentiation", "numpy", "matplotlib")}
    if not args.speg_only:
        versions["torch"] = torch.__version__
    if args.backend != "numpy":
        versions[args.backend] = importlib.metadata.version(args.backend)
    metadata = {"arguments": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                "python": platform.python_version(), "versions": versions,
                "tuning_seed": tuning_seed(args.seed) if args.tune else None,
                "baseline_rates": ("not run" if args.speg_only else
                                   "retuned on independent starts" if args.tune else "fixed reference rates"),
                "plot_floor": PLOT_FLOOR, "stopping": stopping,
                "problems": problem_settings}
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Remove only known outputs from objectives excluded from this successful run.
    for omitted in {"elastic_net", "absolute_sum"} - problem_settings.keys():
        for suffix in (".pdf", ".png"):
            (output / "figures" / f"{omitted}{suffix}").unlink(missing_ok=True)
        if omitted == "elastic_net":
            (results / "elastic_data.npz").unlink(missing_ok=True)
    print(f"Saved figures, LaTeX table, and raw results under {output}", flush=True)


if __name__ == "__main__":
    main()
