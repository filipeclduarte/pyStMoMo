"""Integration tests for the CBD model."""
import numpy as np
import pytest

import pystmomo as ps


def test_cbd_fit_shapes(cbd_fit, ew_data):
    n_ages = len(ew_data.ages)
    n_years = len(ew_data.years)
    assert cbd_fit.ax is None
    assert cbd_fit.bx.shape == (n_ages, 2)
    assert cbd_fit.kt.shape == (2, n_years)


def test_cbd_fitted_rates_range(cbd_fit):
    mask = cbd_fit.wxt > 0
    rates = cbd_fit.fitted_rates[mask]
    assert np.all(rates > 0) and np.all(rates < 1)


def test_cbd_forecast_shape(cbd_fit, ew_data):
    fc = ps.forecast(cbd_fit, h=10)
    assert fc.rates.shape == (len(ew_data.ages), 10)


def test_cbd_forecast_rates_range(cbd_fit):
    fc = ps.forecast(cbd_fit, h=10)
    assert np.all(fc.rates > 0) and np.all(fc.rates < 1)
