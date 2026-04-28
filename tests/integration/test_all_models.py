"""Smoke-test all 7 pre-built models: fit, forecast, simulate."""
import numpy as np
import pytest

import pystmomo as ps


MODELS = [
    ("lc", ps.lc()),
    ("cbd", ps.cbd()),
    ("apc", ps.apc()),
    ("rh", ps.rh()),
    ("m6", ps.m6()),
    ("m7", ps.m7()),
    ("m8", ps.m8()),
]


@pytest.mark.parametrize("name,model", MODELS)
def test_fit_and_forecast(name, model, ew_data):
    fit = model.fit(
        ew_data.deaths, ew_data.exposures,
        ages=ew_data.ages, years=ew_data.years,
    )
    assert fit.fitted_rates is not None
    mask = fit.wxt > 0
    rates = fit.fitted_rates[mask]
    assert np.all(rates > 0), f"{name}: fitted rates not positive"
    assert np.all(rates < 1), f"{name}: fitted rates not < 1"

    fc = ps.forecast(fit, h=10)
    assert fc.rates.shape == (len(ew_data.ages), 10)
    assert np.all(fc.rates > 0), f"{name}: forecast rates not positive"
    assert np.all(fc.rates < 1), f"{name}: forecast rates not < 1"


@pytest.mark.parametrize("name,model", MODELS)
def test_simulate(name, model, ew_data):
    fit = model.fit(
        ew_data.deaths, ew_data.exposures,
        ages=ew_data.ages, years=ew_data.years,
    )
    sim = ps.simulate(fit, nsim=20, h=5, seed=0)
    assert sim.rates.shape == (len(ew_data.ages), 5, 20)
    assert np.all(sim.rates > 0), f"{name}: simulated rates not positive"
    assert np.all(sim.rates < 1), f"{name}: simulated rates not < 1"
