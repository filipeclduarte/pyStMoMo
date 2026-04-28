"""Integration tests for the Lee-Carter model."""
import numpy as np
import pytest

import pystmomo as ps


def test_lc_fit_converges(lc_fit):
    assert lc_fit.converged


def test_lc_parameters_shapes(lc_fit, ew_data):
    n_ages = len(ew_data.ages)
    n_years = len(ew_data.years)
    assert lc_fit.ax.shape == (n_ages,)
    assert lc_fit.bx.shape == (n_ages, 1)
    assert lc_fit.kt.shape == (1, n_years)


def test_lc_constraint_sum_bx(lc_fit):
    np.testing.assert_allclose(lc_fit.bx[:, 0].sum(), 1.0, atol=1e-6)


def test_lc_fitted_rates_positive(lc_fit):
    assert np.all(lc_fit.fitted_rates[lc_fit.wxt > 0] > 0)


def test_lc_aic_bic(lc_fit):
    assert lc_fit.aic() < lc_fit.bic()


def test_lc_forecast_shape(lc_fit, ew_data):
    fc = ps.forecast(lc_fit, h=10)
    assert fc.rates.shape == (len(ew_data.ages), 10)


def test_lc_forecast_rates_in_range(lc_fit):
    fc = ps.forecast(lc_fit, h=10)
    assert np.all(fc.rates > 0)
    assert np.all(fc.rates < 1)


def test_lc_simulate_shape(lc_fit, ew_data):
    sim = ps.simulate(lc_fit, nsim=50, h=10, seed=0)
    assert sim.rates.shape == (len(ew_data.ages), 10, 50)


def test_lc_simulate_rates_positive(lc_fit):
    sim = ps.simulate(lc_fit, nsim=50, h=10, seed=0)
    assert np.all(sim.rates > 0)


def test_lc_boot_semipar(lc_fit):
    boot = ps.semiparametric_bootstrap(lc_fit, nboot=20, seed=0)
    assert len(boot.fits) == 20
    lo, hi = boot.parameter_ci("kt", level=0.9)
    assert lo.shape == lc_fit.kt.shape
    assert np.all(lo <= hi + 1e-10)


def test_lc_boot_residual(lc_fit):
    boot = ps.residual_bootstrap(lc_fit, nboot=20, seed=1)
    assert len(boot.fits) == 20


def test_lc_deviance_residuals(lc_fit):
    res = ps.deviance_residuals(lc_fit)
    mask = lc_fit.wxt > 0
    np.testing.assert_allclose(np.sum(res[mask] ** 2), lc_fit.deviance, rtol=1e-4)
