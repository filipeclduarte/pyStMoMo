"""Unit tests for MultivariateRandomWalkDrift."""
import numpy as np
import pytest

from pystmomo.forecast.mrwd import MultivariateRandomWalkDrift


def _make_kt(N=2, T=30, seed=0):
    rng = np.random.default_rng(seed)
    drifts = np.array([-1.0, 0.5])[:N]
    kt = np.zeros((N, T))
    kt[:, 0] = rng.standard_normal(N)
    for t in range(1, T):
        kt[:, t] = kt[:, t - 1] + drifts + rng.standard_normal(N) * 0.1
    return kt


def test_fit_drift_shape():
    kt = _make_kt()
    mrwd = MultivariateRandomWalkDrift.fit(kt)
    assert mrwd.drift.shape == (2,)
    assert mrwd.sigma.shape == (2, 2)


def test_forecast_shape():
    kt = _make_kt()
    mrwd = MultivariateRandomWalkDrift.fit(kt)
    mean, lower, upper = mrwd.forecast(h=10)
    assert mean.shape == (2, 10)
    assert lower.shape == upper.shape == mean.shape


def test_forecast_monotone_ci():
    kt = _make_kt()
    mrwd = MultivariateRandomWalkDrift.fit(kt)
    mean, lo, hi = mrwd.forecast(h=10)
    assert np.all(lo <= mean + 1e-10)
    assert np.all(hi >= mean - 1e-10)


def test_simulate_shape():
    kt = _make_kt()
    mrwd = MultivariateRandomWalkDrift.fit(kt)
    rng = np.random.default_rng(0)
    sims = mrwd.simulate(h=15, nsim=200, rng=rng)
    assert sims.shape == (2, 15, 200)


def test_univariate():
    kt = _make_kt(N=1)
    mrwd = MultivariateRandomWalkDrift.fit(kt)
    mean, lo, hi = mrwd.forecast(h=5)
    assert mean.shape == (1, 5)
