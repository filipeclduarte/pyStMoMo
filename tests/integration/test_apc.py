"""Integration tests for the APC model."""
import numpy as np
import pytest

import pystmomo as ps


def test_apc_fit_shapes(apc_fit, ew_data):
    n_ages = len(ew_data.ages)
    n_years = len(ew_data.years)
    assert apc_fit.ax.shape == (n_ages,)
    assert apc_fit.kt.shape == (1, n_years)
    assert apc_fit.gc is not None


def test_apc_kt_zero_mean(apc_fit):
    np.testing.assert_allclose(apc_fit.kt[0].mean(), 0.0, atol=1e-6)


def test_apc_fitted_positive(apc_fit):
    mask = apc_fit.wxt > 0
    assert np.all(apc_fit.fitted_deaths[mask] > 0)


def test_apc_forecast_shape(apc_fit, ew_data):
    fc = ps.forecast(apc_fit, h=10)
    assert fc.rates.shape == (len(ew_data.ages), 10)
