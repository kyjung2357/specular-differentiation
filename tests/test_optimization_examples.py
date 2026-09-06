"""Numerical contracts for the manuscript's reusable scalar objectives."""

from contextlib import nullcontext
import csv
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import specular
from examples.optimization import one_dimension
from examples.optimization.one_dimension import (
    AbsoluteSum, ElasticNet, random_problem,
)
from specular.optimization import specular_gradient


def test_numpy_examples_do_not_import_torch():
    subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys; from examples.optimization.one_dimension import "
         "AbsoluteSum, random_problem; "
         "random_problem().analytic_gradient(1.); AbsoluteSum().objective(0.); "
         "assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1], check=True,
    )


def test_default_elastic_net_uses_current_research_settings():
    problem = random_problem()
    np.testing.assert_array_equal(
        problem.a, np.random.default_rng(one_dimension.data_seed).standard_normal(one_dimension.m)
    )
    np.testing.assert_array_equal(problem.b, np.ones(one_dimension.m))
    assert problem.m == one_dimension.m
    assert problem.lambda1 == one_dimension.lambda_1
    assert problem.lambda2 == one_dimension.lambda_2
    assert problem.gap(problem.minimizer) == 0.0
    assert problem.objective(problem.minimizer) == pytest.approx(problem.minimum)


def test_large_penalty_is_available_as_an_explicit_setting():
    problem = random_problem(m=100, seed=20260907, lambda1=1000., lambda2=.5)
    assert problem.lambda1 == 1000.
    assert problem.lambda2 == .5
    assert problem.minimizer == 0.
    assert problem.minimum == .5
    assert problem.gap(1.) == pytest.approx(
        problem.curvature / 2 + 1000 - problem.correlation
    )


@pytest.mark.parametrize("correlation,lambda1,lambda2", product(
    [-2., 0., 2.], [0., .5, 2., 1000.], [0., .5, 2.]
))
def test_elastic_net_parameter_variants(correlation, lambda1, lambda2):
    problem = ElasticNet([1., 2., -1.], [correlation, correlation, 0.],
                         lambda1, lambda2)
    grid = np.r_[np.linspace(-3, 3, 81), problem.minimizer]
    direct = (np.mean((grid[:, None] * problem.a - problem.b) ** 2, axis=1) / 2
              + lambda2 * grid ** 2 / 2 + lambda1 * np.abs(grid))
    np.testing.assert_allclose(problem.objective(grid), direct)
    np.testing.assert_allclose(problem.gap(grid), direct - problem.minimum,
                               atol=1e-12, rtol=2e-14)
    assert np.all(problem.gap(grid) >= 0)
    left, right = problem.slopes(problem.minimizer)
    assert left <= 1e-13
    assert right >= -1e-13
    assert problem.gap(problem.minimizer) == 0


def test_elastic_net_slopes_match_one_sided_difference_quotients():
    problem = ElasticNet([1., 2.], [2., -1.], lambda1=3., lambda2=.5)
    x = np.array([-2., 0., 2.])
    h = 1e-6
    left, right = problem.slopes(x)
    np.testing.assert_allclose(left, (problem.objective(x) - problem.objective(x-h))/h,
                               atol=2e-6, rtol=1e-7)
    np.testing.assert_allclose(right, (problem.objective(x+h) - problem.objective(x))/h,
                               atol=2e-6, rtol=1e-7)


def test_zero_curvature_and_copied_data():
    source = np.array([0., 0.])
    problem = ElasticNet(source, [1., 3.], lambda1=0., lambda2=0.)
    source[0] = 2.
    assert problem.a[0] == 0
    assert not problem.a.flags.writeable
    np.testing.assert_array_equal(problem.objective([-3., 4.]), [2.5, 2.5])
    np.testing.assert_array_equal(problem.gap([-3., 4.]), [0., 0.])
    assert problem.minimizer == 0


@pytest.mark.parametrize("m", [0, -1, 1.5, True])
def test_random_problem_rejects_invalid_observation_count(m):
    with pytest.raises(ValueError, match="positive integer"):
        random_problem(m=m)


@pytest.mark.parametrize("a,b,lambda1,lambda2", [
    ([], [], 1., 1.), ([1.], [1., 2.], 1., 1.),
    ([np.nan], [1.], 1., 1.), ([1.], [1.], -1., 0.),
    ([1.], [1.], 0., np.inf),
])
def test_elastic_net_rejects_invalid_parameters(a, b, lambda1, lambda2):
    with pytest.raises(ValueError):
        ElasticNet(a, b, lambda1, lambda2)


def test_absolute_sum_matches_direct_formula_and_knot_slopes():
    problem = AbsoluteSum()
    x = np.array([-2., -.01, 0., .005, .01, 2.])
    direct = sum(np.abs(x-i/100) + np.abs(x+i/100) for i in range(100))
    np.testing.assert_allclose(problem.objective(x), direct)
    np.testing.assert_allclose(problem.gap(x), direct - 99., atol=1e-12)
    left, right = problem.slopes(x)
    np.testing.assert_array_equal(left, [-200, -4, -2, 2, 2, 200])
    np.testing.assert_array_equal(right, [-200, -2, 2, 2, 4, 200])
    assert problem.objective(0.) == pytest.approx(problem.minimum)
    assert problem.analytic_gradient(0.) == 0.


def test_tiny_objective_gaps_survive_cancellation():
    elastic = random_problem()
    absolute = AbsoluteSum()
    x = np.array([-1e-30, 1e-30])
    np.testing.assert_array_equal(elastic.objective(x), [elastic.minimum]*2)
    np.testing.assert_array_equal(absolute.objective(x), [absolute.minimum]*2)
    assert np.all(elastic.gap(x) > 0)
    np.testing.assert_array_equal(absolute.gap(x), [2e-30, 2e-30])
    shifted = ElasticNet([1.], [2.], lambda1=1., lambda2=0.)
    nearby = np.nextafter(shifted.minimizer, np.inf)
    assert shifted.gap(nearby) == (nearby - shifted.minimizer)**2 / 2
    assert shifted.gap(nearby) > 0


@pytest.mark.parametrize("problem", [random_problem(), AbsoluteSum()])
def test_analytic_gradient_dispatches_through_public_package(problem, monkeypatch):
    points = np.array([-1., 0., .01])
    seen = []
    expected = np.array([7., 8., 9.])

    def selected_mean(left, right):
        seen.append((left, right))
        return expected

    monkeypatch.setattr(specular, "scaled_mean", selected_mean)
    assert problem.analytic_gradient(points) is expected
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], problem.slopes(points))


@pytest.mark.parametrize("backend", ["numpy", "numba", "jax"])
def test_analytic_gradient_and_speg_use_selected_backend(backend):
    if backend != "numpy":
        pytest.importorskip(backend)
    precision = (sys.modules["jax"].enable_x64(True)
                 if backend == "jax" else nullcontext())
    with precision, specular.use_backend(backend):
        for problem in (random_problem(), AbsoluteSum()):
            points = np.array([-1., 0., .01])
            left, right = problem.slopes(points)
            expected = np.tan((np.arctan(left) + np.arctan(right)) / 2)
            values = np.asarray(problem.analytic_gradient(points))
            np.testing.assert_allclose(values, expected, rtol=5e-14, atol=1e-14)
            assert values.dtype == np.float64
            assert np.ndim(problem.analytic_gradient(0.)) == 0
        smooth = ElasticNet([1.], [0.], lambda1=0., lambda2=0.)
        assert float(smooth.analytic_gradient(1e-30)) == 1e-30
        paths, _ = one_dimension.run_speg(smooth, [0.7], 12, 0.5)
        bound = np.ldexp(1., -np.arange(13))
        assert np.all(np.abs(paths[0]) <= bound + 1e-15)


@pytest.mark.parametrize("problem", [random_problem(), AbsoluteSum()])
def test_analytic_gradient_supports_scalar_and_array_inputs(problem):
    points = np.array([[-1., -.01], [0., 2.]])
    values = problem.analytic_gradient(points)
    assert values.shape == points.shape
    for point, value in zip(points.flat, values.flat):
        assert problem.analytic_gradient(float(point)) == value


@pytest.mark.parametrize("problem", [
    random_problem(), AbsoluteSum(), ElasticNet([1., 2.], [2., -1.], .5, 2.),
])
def test_optional_torch_objectives_and_autograd_match_analytic_slopes(problem):
    torch = pytest.importorskip("torch")
    points = np.array([[-2., -.01, 0.], [.005, .01, 2.]])
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    value = problem.torch_objective(x)
    value.sum().backward()
    np.testing.assert_allclose(value.detach().numpy(), problem.objective(points),
                               atol=1e-12, rtol=2e-14)
    left, right = problem.slopes(points)
    # PyTorch selects the zero subgradient of abs at its kink.
    np.testing.assert_allclose(x.grad.numpy(), (left+right)/2,
                               atol=2e-13, rtol=1e-14)


@pytest.mark.parametrize("problem", [
    random_problem(), AbsoluteSum(), ElasticNet([1.], [3.], lambda1=.5, lambda2=0.),
])
@pytest.mark.parametrize("offset", [-10., -3.14159, 0., 7.12345, 10.])
def test_package_speg_geometric_steps_obey_manuscript_distance_bound(problem, offset):
    # Package indices start at n=1: a=2*t0, r=1/2 gives t_k=t0*2**(-k).
    t0 = 5.
    result = specular_gradient(
        problem.objective, problem.minimizer + offset,
        gradient=problem.analytic_gradient, step_size="geometric_series",
        a=2*t0, r=.5, max_iter=40, tol=0,
    )
    history = result.history["variables"]
    np.testing.assert_array_equal(result.history["step_sizes"],
                                  np.ldexp(t0, -np.arange(result.iteration)))
    bound = np.ldexp(2*t0, -np.arange(len(history)))
    error = np.abs(history - problem.minimizer)
    assert np.all(error <= bound + 4*np.finfo(float).eps*max(1., abs(problem.minimizer)))


@pytest.fixture
def example_runner():
    return one_dimension


def test_editable_settings_control_cli_and_problem_defaults(example_runner, monkeypatch):
    settings = {
        "lambda_1": 7., "lambda_2": .25, "m": 7, "num_runs": 4,
        "max_iter": 1300, "absolute_sum_max_iter": 37,
        "seed": 18, "data_seed": 19, "backend": "numba",
    }
    for name, value in settings.items():
        monkeypatch.setattr(example_runner, name, value)
    args = example_runner.parse_args([])
    assert (args.lambda1, args.lambda2, args.m, args.trials, args.updates,
            args.absolute_sum_updates, args.seed, args.data_seed, args.backend) == (
        7., .25, 7, 4, 1300, 37, 18, 19, "numba",
    )
    problem = example_runner.random_problem()
    assert (problem.lambda1, problem.lambda2, problem.m) == (7., .25, 7)
    np.testing.assert_array_equal(problem.a, np.random.default_rng(19).standard_normal(7))
    overridden = example_runner.random_problem(m=2, seed=20, lambda1=0., lambda2=0.)
    assert (overridden.lambda1, overridden.lambda2, overridden.m) == (0., 0., 2)
    np.testing.assert_array_equal(overridden.a, np.random.default_rng(20).standard_normal(2))


def test_example_cli_accepts_custom_configuration(example_runner):
    customized = example_runner.parse_args([
        "--example", "elastic-net", "--trials", "2", "--updates", "12",
        "--seed", "8", "--data-seed", "9", "--m", "20",
        "--lambda1", "100", "--lambda2", "1", "--speg-only", "--backend", "jax",
    ])
    assert (customized.m, customized.lambda1, customized.updates) == (20, 100., 12)
    assert (customized.example, customized.trials, customized.seed,
            customized.data_seed, customized.lambda2, customized.speg_only) == (
        "elastic-net", 2, 8, 9, 1., True,
    )
    assert customized.backend == "jax"
    assert not customized.tune


def test_default_budgets_keep_independent_runs_separate_from_updates(example_runner):
    args = example_runner.parse_args([])
    assert args.trials == 100
    assert args.updates == 1000
    assert args.absolute_sum_updates == 50


@pytest.mark.parametrize("example", ["both", "elastic-net", "absolute-sum"])
def test_explicit_updates_override_both_objective_budgets(example_runner, example):
    args = example_runner.parse_args(["--example", example, "--updates", "12"])
    assert (args.updates, args.absolute_sum_updates) == (12, 12)


@pytest.mark.parametrize("updates", [0, -1, 1.5, True])
def test_absolute_sum_setting_must_be_a_positive_integer(example_runner, monkeypatch, updates):
    monkeypatch.setattr(example_runner, "absolute_sum_max_iter", updates)
    with pytest.raises(SystemExit):
        example_runner.parse_args([])


@pytest.mark.parametrize("flags,tune,speg_only", [
    ([], True, False), (["--tune"], True, False),
    (["--no-tune"], False, False), (["--speg-only"], False, True),
])
def test_tuning_is_default_and_can_be_disabled(example_runner, flags, tune, speg_only):
    args = example_runner.parse_args(flags)
    assert (args.tune, args.speg_only) == (tune, speg_only)


def test_explicit_tuning_cannot_be_combined_with_speg_only(example_runner):
    with pytest.raises(SystemExit):
        example_runner.parse_args(["--speg-only", "--tune"])


@pytest.mark.parametrize("updates", [0, -1])
def test_iteration_count_must_be_positive(example_runner, updates):
    with pytest.raises(SystemExit):
        example_runner.parse_args(["--updates", str(updates)])


def test_iteration_count_can_exceed_manuscript_budget(example_runner):
    assert example_runner.parse_args(["--updates", "1500"]).updates == 1500


def test_training_seed_stays_independent_when_test_seed_changes(example_runner):
    assert example_runner.tuning_seed(20260905) == 20260906
    for test_seed in (0, 20260905, 20260906, 20260908):
        training_seed = example_runner.tuning_seed(test_seed)
        assert training_seed != test_seed
        training = np.random.default_rng(training_seed).uniform(-1, 1, 32)
        testing = np.random.default_rng(test_seed).uniform(-1, 1, 32)
        assert not np.array_equal(training, testing)


def test_empty_csv_output_removes_stale_results(example_runner, tmp_path):
    target = tmp_path / "selected_parameters.csv"
    example_runner.write_csv(target, [{"method": "GD", "learning_rate": 0.1}])
    assert "GD,0.1" in target.read_text(encoding="utf-8")
    example_runner.write_csv(target, [])
    assert not target.exists()
    example_runner.write_csv(target, [])  # An absent file is also valid.


def test_latex_summary_uses_raw_errors_and_each_objective_budget(example_runner, tmp_path):
    target = tmp_path / "summary.tex"
    rows = [
        {"example": "elastic_net", "method": "SPEG", "updates": 1000,
         "median_distance": 1.234e-30, "median_best_gap": 0.},
        {"example": "absolute_sum", "method": "SPEG", "updates": 50,
         "median_distance": 2.345e-16, "median_best_gap": 4.69e-16},
    ]
    example_runner.write_summary_tex(target, rows)
    table = target.read_text(encoding="utf-8")
    assert r"\begin{tabular}{llrcc}" in table
    assert "Updates $k$" in table
    assert r"x_k" in table
    assert r"f_k" in table
    assert r"1.23\times10^{-30}" in table
    assert "Elastic Net & SPEG & 1000 &" in table
    assert "Sum of absolute values & SPEG & 50 &" in table
    assert r"2.35\times10^{-16}" in table
    assert "$0$" in table
    assert "GD" not in table and "Adam" not in table


def test_exported_histories_and_metadata_use_each_objective_budget(
        example_runner, monkeypatch, tmp_path):
    monkeypatch.setattr(example_runner, "max_iter", 5)
    monkeypatch.setattr(example_runner, "absolute_sum_max_iter", 2)
    monkeypatch.setattr(example_runner, "plot", lambda *args: None)
    example_runner.main([
        "--example", "both", "--trials", "3", "--speg-only", "--backend", "numpy",
        "--output-dir", str(tmp_path),
    ])
    results = tmp_path / "results"
    with np.load(results / "trajectories.npz") as trajectories:
        assert trajectories["elastic_net_SPEG"].shape == (3, 6)
        assert trajectories["absolute_sum_SPEG"].shape == (3, 3)
        assert trajectories["elastic_net_starts"].shape == (3,)
        assert trajectories["absolute_sum_starts"].shape == (3,)
    with (results / "summary.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert {row["example"]: int(row["updates"]) for row in summary} == {
        "elastic_net": 5, "absolute_sum": 2,
    }
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["problems"]["elastic_net"]["updates"] == 5
    assert metadata["problems"]["absolute_sum"]["updates"] == 2


@pytest.mark.parametrize("method", ["GD", "Adam"])
def test_batched_baselines_match_individual_scalar_runs(example_runner, method):
    pytest.importorskip("torch")
    starts = np.array([-1.231, 0.017, 1.725])
    for problem in (random_problem(), AbsoluteSum()):
        batch = example_runner.run_baseline(problem, method, starts, 12, 0.002)
        single = np.vstack([
            example_runner.run_baseline(problem, method, [x], 12, 0.002)[0]
            for x in starts
        ])
        np.testing.assert_allclose(batch, single, rtol=2e-14, atol=1e-15)


def test_precision_stop_pads_shifted_minimizer_history(example_runner):
    # Floating-point slopes cannot be exactly zero at this nonzero minimizer.
    problem = ElasticNet([1.], [2.], lambda1=.3, lambda2=0.)
    paths, stops = example_runner.run_speg(problem, [1.9], 100, 3.)
    stop = stops[0]
    assert stop["reason"] == "step does not change the point at working precision"
    assert 0 < stop["updates"] < 100
    np.testing.assert_array_equal(paths[0, stop["updates"]:], paths[0, -1])
    assert abs(paths[0, -1] - problem.minimizer) <= np.spacing(problem.minimizer)


def test_long_run_preserves_final_iterate_after_geometric_underflow(example_runner):
    paths, stops = example_runner.run_speg(random_problem(), [1.231], 1500, 3.)
    assert paths.shape == (1, 1501)
    assert np.isfinite(paths).all()
    stop = stops[0]
    assert stop["reason"] == "geometric step underflows at working precision"
    assert 1000 < stop["updates"] < 1500
    np.testing.assert_array_equal(paths[0, stop["updates"]:], paths[0, -1])
    assert abs(paths[0, -1]) < np.finfo(float).tiny


def test_failed_speg_stop_is_not_padded_as_success(example_runner, monkeypatch):
    monkeypatch.setattr(example_runner, "specular_gradient", lambda *a, **k:
                        SimpleNamespace(stop_reason="non-finite objective at proposed point"))
    with pytest.raises(RuntimeError, match="SPEG run 0 failed: non-finite objective"):
        example_runner.run_speg(AbsoluteSum(), [0.3], 12, 0.5)
