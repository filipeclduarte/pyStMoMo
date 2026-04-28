"""pyStMoMo — Stochastic Mortality Modelling in Python.

A Python port of the StMoMo R library for Generalised Age-Period-Cohort (GAPC)
mortality models.  Provides fitting, forecasting, simulation, bootstrap
uncertainty quantification, diagnostics, and visualisation.

Quick start
-----------
>>> import pystmomo as ps
>>> data = ps.load_ew_male()
>>> fit  = ps.lc().fit(data.deaths, data.exposures,
...                    ages=data.ages, years=data.years)
>>> fc   = ps.forecast(fit, h=20)
>>> sim  = ps.simulate(fit, nsim=500, h=20, seed=42)
>>> boot = ps.semiparametric_bootstrap(fit, nboot=200)
>>> ps.plot_parameters(fit)
"""
from __future__ import annotations

# ── Version ───────────────────────────────────────────────────────────────────
from ._version import __version__
from .bootstrap.boot_result import BootStMoMo
from .bootstrap.residual_boot import residual_bootstrap
from .bootstrap.semipar_boot import semiparametric_bootstrap

# ── Core age-functions ────────────────────────────────────────────────────────
from .core.age_functions import (
    CallableAgeFun,
    CenteredCohortAgeFun,
    ConstantAgeFun,
    LinearAgeFun,
    NonParametricAgeFun,
    QuadraticAgeFun,
)

# ── Model specification ───────────────────────────────────────────────────────
from .core.stmomo import StMoMo

# ── Data ──────────────────────────────────────────────────────────────────────
from .data._loader import StMoMoData, load_ew_male, load_hmd_csv
from .diagnostics.crossval import cv_stmomo

# ── Diagnostics ───────────────────────────────────────────────────────────────
from .diagnostics.residuals import (
    deviance_residuals,
    pearson_residuals,
    response_residuals,
)

# ── Results ───────────────────────────────────────────────────────────────────
from .fit.fit_result import FitStMoMo
from .forecast.external import ExternalKtForecaster

# ── High-level operations ─────────────────────────────────────────────────────
from .forecast.forecast import forecast
from .forecast.forecast_result import ForStMoMo
from .models.apc import apc
from .models.cbd import cbd

# ── Pre-built models ──────────────────────────────────────────────────────────
from .models.lc import lc
from .models.m6 import m6
from .models.m7 import m7
from .models.m8 import m8
from .models.rh import rh
from .plot.forecast_plot import plot_fan, plot_forecast

# ── Plotting ──────────────────────────────────────────────────────────────────
from .plot.parameters import plot_parameters
from .plot.residual_plot import plot_residual_heatmap, plot_residual_scatter
from .simulate.sim_result import SimStMoMo
from .simulate.simulate import simulate

__all__ = [
    # data
    "StMoMoData",
    "load_ew_male",
    "load_hmd_csv",
    # age functions
    "NonParametricAgeFun",
    "ConstantAgeFun",
    "LinearAgeFun",
    "QuadraticAgeFun",
    "CenteredCohortAgeFun",
    "CallableAgeFun",
    # model spec
    "StMoMo",
    # pre-built models
    "lc",
    "cbd",
    "apc",
    "rh",
    "m6",
    "m7",
    "m8",
    # results
    "FitStMoMo",
    "ForStMoMo",
    "SimStMoMo",
    "BootStMoMo",
    # operations
    "forecast",
    "simulate",
    "ExternalKtForecaster",
    "semiparametric_bootstrap",
    "residual_bootstrap",
    # diagnostics
    "deviance_residuals",
    "pearson_residuals",
    "response_residuals",
    "cv_stmomo",
    # plotting
    "plot_parameters",
    "plot_forecast",
    "plot_fan",
    "plot_residual_heatmap",
    "plot_residual_scatter",
    # version
    "__version__",
]
