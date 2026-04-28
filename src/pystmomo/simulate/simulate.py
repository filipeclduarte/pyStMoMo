"""Monte Carlo simulation of future mortality trajectories."""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..core.predictor import invlogit
from ..fit.fit_result import FitStMoMo
from ..forecast.external import ExternalKtForecaster
from .sim_result import SimStMoMo

_CLIP = 30.0
_EPS = 1e-15


def simulate(
    fit: FitStMoMo,
    nsim: int = 1000,
    h: int = 50,
    *,
    kt_method: Literal['mrwd', 'arima'] | ExternalKtForecaster = "mrwd",
    kt_arima_order: tuple[int, int, int] = (0, 1, 0),
    gc_method: Literal['arima', 'mrwd'] | ExternalKtForecaster = "arima",
    gc_arima_order: tuple[int, int, int] = (1, 1, 0),
    jump_choice: Literal["fit", "actual"] = "fit",
    seed: int | None = None,
) -> SimStMoMo:
    """Simulate future mortality rate sample paths.

    Parameters
    ----------
    fit:
        Fitted model from which to simulate.
    nsim:
        Number of sample paths to generate.
    h:
        Forecast horizon (years).
    kt_method:
        Method for simulating period indexes: ``"mrwd"`` (Multivariate Random
        Walk with Drift, default) or ``"arima"`` (independent ARIMA models).
    kt_arima_order:
        ARIMA order for period indexes when ``kt_method="arima"``.
    gc_method:
        Method for simulating cohort effects (``"arima"`` default).
    gc_arima_order:
        ARIMA order for cohort effects.
    jump_choice:
        ``"fit"`` (default) uses fitted rates as the jump-off; ``"actual"``
        uses observed rates.
    seed:
        Integer random seed for reproducibility.

    Returns
    -------
    SimStMoMo
        Simulation result with ``rates`` array of shape (n_ages, h, nsim).

    Examples
    --------
    >>> from pystmomo import lc, load_ew_male, simulate
    >>> data = load_ew_male()
    >>> fit = lc().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    >>> sim = simulate(fit, nsim=500, h=30, seed=42)
    >>> sim.rates.shape
    (35, 30, 500)
    """
    rng = np.random.default_rng(seed)
    ages = fit.ages
    years = fit.years
    cohorts = fit.cohorts
    link = fit.model.link
    n_ages = len(ages)

    years_f = np.arange(int(years[-1]) + 1, int(years[-1]) + h + 1)

    # ------------------------------------------------------------------ #
    # Simulate period indexes kt_s: (N, h, nsim)                          #
    # ------------------------------------------------------------------ #
    if fit.model.N == 0:
        kt_s = np.zeros((0, h, nsim))
    elif isinstance(kt_method, ExternalKtForecaster):
        kt_s = kt_method.simulate(h, nsim, rng)
    elif kt_method == "mrwd":
        from ..forecast.mrwd import MultivariateRandomWalkDrift
        kt_model = MultivariateRandomWalkDrift.fit(fit.kt)
        kt_s = kt_model.simulate(h, nsim, rng)
    else:
        from ..forecast.arima_fc import IndependentArima
        kt_model = IndependentArima.fit(fit.kt, order=kt_arima_order)
        kt_s = kt_model.simulate(h, nsim, rng)

    # ------------------------------------------------------------------ #
    # Simulate cohort effects gc_s for new cohorts                        #
    # ------------------------------------------------------------------ #
    gc_s: np.ndarray | None = None
    new_cohort_values: list[int] = []

    if fit.model.has_cohort and fit.gc is not None:
        max_fitted_cohort = int(cohorts[-1])
        # Cohorts needed for future years and observed ages
        for yr in years_f:
            for age in ages:
                c = int(yr) - int(age)
                if c > max_fitted_cohort and c not in new_cohort_values:
                    new_cohort_values.append(c)
        new_cohort_values.sort()

        if new_cohort_values:
            n_new = len(new_cohort_values)
            if isinstance(gc_method, ExternalKtForecaster):
                gc_s_raw = gc_method.simulate(n_new, nsim, rng)  # (1, n_new, nsim)
                gc_s = gc_s_raw[0]
            elif gc_method == "arima":
                from ..forecast.arima_fc import IndependentArima
                gc_2d = fit.gc[np.newaxis, :]
                gc_model_obj = IndependentArima.fit(gc_2d, order=gc_arima_order)
                gc_sim_raw = gc_model_obj.simulate(n_new, nsim, rng)
                gc_s = gc_sim_raw[0]
            else:
                from ..forecast.mrwd import MultivariateRandomWalkDrift
                gc_2d = fit.gc[np.newaxis, :]
                gc_model_obj = MultivariateRandomWalkDrift.fit(gc_2d)
                gc_s_raw = gc_model_obj.simulate(n_new, nsim, rng)
                gc_s = gc_s_raw[0]

    # ------------------------------------------------------------------ #
    # Compute rates for each simulation path                              #
    # ------------------------------------------------------------------ #
    # Build gc lookup: {cohort: index_in_gc_s} for new cohorts
    new_cohort_map = {c: j for j, c in enumerate(new_cohort_values)}
    fitted_cohort_map = {int(c): j for j, c in enumerate(cohorts)}

    # rates: (n_ages, h, nsim)
    rates = np.zeros((n_ages, h, nsim))

    ax = fit.ax    # (n_ages,) or None
    bx = fit.bx    # (n_ages, N)
    b0x = fit.b0x  # (n_ages,) or None
    gc_fit = fit.gc  # (n_cohorts,)

    for s_idx in range(nsim):
        kt_path = kt_s[:, :, s_idx]   # (N, h)
        # Assemble eta for this simulation: (n_ages, h)
        eta = np.zeros((n_ages, h))
        if ax is not None:
            eta += ax[:, None]
        if bx.size > 0:
            eta += bx @ kt_path          # (n_ages, h)

        if fit.model.has_cohort and b0x is not None:
            for j_f, yr in enumerate(years_f):
                for i, age in enumerate(ages):
                    c = int(yr) - int(age)
                    if c in fitted_cohort_map:
                        gc_val = gc_fit[fitted_cohort_map[c]]
                    elif c in new_cohort_map and gc_s is not None:
                        gc_val = gc_s[new_cohort_map[c], s_idx]
                    else:
                        gc_val = 0.0
                    eta[i, j_f] += b0x[i] * gc_val

        if link == "log":
            rates[:, :, s_idx] = np.exp(np.clip(eta, -_CLIP, _CLIP))
        else:
            rates[:, :, s_idx] = invlogit(eta)

    return SimStMoMo(
        fit=fit,
        h=h,
        nsim=nsim,
        rates=rates,
        years_f=years_f,
        ages=ages,
        kt_s=kt_s,
        gc_s=gc_s,
        seed=seed,
    )
