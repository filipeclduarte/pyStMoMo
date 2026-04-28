"""Unit tests for ExternalKtForecaster."""
import numpy as np
import pytest

import pystmomo as ps
from pystmomo.forecast.external import ExternalKtForecaster


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_forecast_fn(N: int, slope: float = -1.0, sigma: float = 0.5):
    """Return a simple random-walk forecast function for N indexes."""
    def fn(h, *, level=0.95):
        from scipy.stats import norm
        z = norm.ppf((1 + level) / 2)
        base = slope * np.arange(1, h + 1)
        mean = np.tile(base, (N, 1))          # (N, h)
        se = sigma * np.sqrt(np.arange(1, h + 1))
        return mean, mean - z * se, mean + z * se
    return fn


def _make_simulate_fn(N: int, slope: float = -1.0, sigma: float = 0.5):
    """Return a simple random-walk simulate function for N indexes."""
    def fn(h, nsim, *, rng):
        paths = np.zeros((N, h, nsim))
        for i in range(N):
            innov = rng.normal(slope, sigma, (h, nsim))
            paths[i] = np.cumsum(innov, axis=0)
        return paths
    return fn


# ── Construction ───────────────────────────────────────────────────────────────

def test_construction_forecast_only():
    ext = ExternalKtForecaster(_make_forecast_fn(1))
    assert ext._simulate_fn is None


def test_construction_with_simulate():
    ext = ExternalKtForecaster(_make_forecast_fn(1), _make_simulate_fn(1))
    assert ext._simulate_fn is not None


def test_repr_no_simulate():
    ext = ExternalKtForecaster(_make_forecast_fn(1))
    assert "deterministic fallback" in repr(ext)


def test_repr_with_simulate():
    ext = ExternalKtForecaster(_make_forecast_fn(1), _make_simulate_fn(1))
    assert "yes" in repr(ext)


# ── forecast() ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [1, 2, 3])
def test_forecast_shapes(N):
    ext = ExternalKtForecaster(_make_forecast_fn(N))
    mean, lo, hi = ext.forecast(h=10)
    assert mean.shape == (N, 10)
    assert lo.shape == (N, 10)
    assert hi.shape == (N, 10)


def test_forecast_ci_ordering():
    ext = ExternalKtForecaster(_make_forecast_fn(1))
    mean, lo, hi = ext.forecast(h=10)
    assert np.all(lo <= mean + 1e-10)
    assert np.all(hi >= mean - 1e-10)


def test_forecast_fn_returns_array_only():
    """If forecast_fn returns a plain array, lo==hi==mean."""
    def fn(h, *, level=0.95):
        return np.ones((1, h))

    ext = ExternalKtForecaster(fn)
    mean, lo, hi = ext.forecast(h=5)
    np.testing.assert_array_equal(lo, mean)
    np.testing.assert_array_equal(hi, mean)


def test_forecast_fn_returns_none_bounds():
    """If lower/upper are None, they default to mean."""
    def fn(h, *, level=0.95):
        mean = np.zeros((1, h))
        return mean, None, None

    ext = ExternalKtForecaster(fn)
    mean, lo, hi = ext.forecast(h=5)
    np.testing.assert_array_equal(lo, mean)
    np.testing.assert_array_equal(hi, mean)


# ── simulate() ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("N", [1, 2, 3])
def test_simulate_shape_with_fn(N):
    ext = ExternalKtForecaster(_make_forecast_fn(N), _make_simulate_fn(N))
    rng = np.random.default_rng(0)
    paths = ext.simulate(h=10, nsim=50, rng=rng)
    assert paths.shape == (N, 10, 50)


def test_simulate_deterministic_fallback():
    """Without simulate_fn, all nsim paths are identical (the forecast mean)."""
    ext = ExternalKtForecaster(_make_forecast_fn(1))
    rng = np.random.default_rng(0)
    paths = ext.simulate(h=10, nsim=20, rng=rng)
    assert paths.shape == (1, 10, 20)
    # all simulations identical
    assert np.all(paths[:, :, 0:1] == paths)


def test_simulate_2d_output_promoted():
    """simulate_fn returning (h, nsim) is promoted to (1, h, nsim)."""
    def fn(h, nsim, *, rng):
        return rng.standard_normal((h, nsim))   # 2-D

    ext = ExternalKtForecaster(_make_forecast_fn(1), fn)
    rng = np.random.default_rng(0)
    paths = ext.simulate(h=8, nsim=15, rng=rng)
    assert paths.shape == (1, 8, 15)


# ── Integration with forecast() and simulate() ────────────────────────────────

def test_lc_forecast_with_external(lc_fit, ew_data):
    ext = ExternalKtForecaster(
        _make_forecast_fn(1),
        _make_simulate_fn(1),
    )
    fc = ps.forecast(lc_fit, h=10, kt_method=ext)
    assert fc.rates.shape == (len(ew_data.ages), 10)
    assert np.all(fc.rates > 0)


def test_lc_simulate_with_external(lc_fit, ew_data):
    ext = ExternalKtForecaster(
        _make_forecast_fn(1),
        _make_simulate_fn(1),
    )
    sim = ps.simulate(lc_fit, nsim=30, h=10, kt_method=ext, seed=0)
    assert sim.rates.shape == (len(ew_data.ages), 10, 30)
    assert np.all(sim.rates > 0)


def test_cbd_forecast_with_external_n2(cbd_fit, ew_data):
    """CBD has N=2 — forecaster must return shape (2, h)."""
    ext = ExternalKtForecaster(
        _make_forecast_fn(N=2),
        _make_simulate_fn(N=2),
    )
    fc = ps.forecast(cbd_fit, h=10, kt_method=ext)
    assert fc.rates.shape == (len(ew_data.ages), 10)
    assert np.all(fc.rates > 0)
    assert np.all(fc.rates < 1)   # CBD gives probabilities


def test_apc_forecast_with_external_kt_and_gc(apc_fit, ew_data):
    """APC: separate ExternalKtForecaster for kt and gc."""
    ext_kt = ExternalKtForecaster(_make_forecast_fn(1), _make_simulate_fn(1))
    ext_gc = ExternalKtForecaster(_make_forecast_fn(1), _make_simulate_fn(1))
    fc = ps.forecast(apc_fit, h=10, kt_method=ext_kt, gc_method=ext_gc)
    assert fc.rates.shape == (len(ew_data.ages), 10)


def test_set_style_does_not_fire_on_import():
    """Importing pystmomo must not mutate rcParams."""
    import matplotlib as mpl
    # Default figsize before any style call
    default_figsize = mpl.rcParamsDefault["figure.figsize"]
    # Re-import should be a no-op for rcParams
    import importlib
    import pystmomo  # noqa: F401
    importlib.reload(pystmomo)
    assert list(mpl.rcParams["figure.figsize"]) == list(default_figsize), \
        "import pystmomo must not call set_style() automatically"
