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
    new_cohort_values: np.ndarray = np.array([], dtype=int)

    if fit.model.has_cohort and fit.gc is not None:
        max_fitted_cohort = int(cohorts[-1])
        # Vectorised: find all unique cohorts needed in the future grid
        cohort_grid_f = years_f[None, :] - ages[:, None]   # (n_ages, h)
        all_future_cohorts = np.unique(cohort_grid_f)
        new_cohort_values = all_future_cohorts[all_future_cohorts > max_fitted_cohort]

        if len(new_cohort_values) > 0:
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
    # Compute rates for all simulation paths (fully vectorised)           #
    # ------------------------------------------------------------------ #
    ax = fit.ax    # (n_ages,) or None
    bx = fit.bx    # (n_ages, N)
    b0x = fit.b0x  # (n_ages,) or None
    gc_fit = fit.gc  # (n_cohorts,)

    # eta_base: (n_ages, h) — parts that don't vary across simulations
    eta_base = np.zeros((n_ages, h))
    if ax is not None:
        eta_base += ax[:, None]

    # kt contribution: bx @ kt_s → (n_ages, h, nsim) via einsum
    # bx: (n_ages, N), kt_s: (N, h, nsim)
    if bx.size > 0 and kt_s.size > 0:
        kt_contrib = np.einsum('aN,Nhs->ahs', bx, kt_s)  # (n_ages, h, nsim)
    else:
        kt_contrib = np.zeros((n_ages, h, nsim))

    # Cohort contribution: precompute index grid once
    coh_contrib = np.zeros((n_ages, h, nsim))
    if fit.model.has_cohort and b0x is not None:
        cohort_grid_f = years_f[None, :] - ages[:, None]  # (n_ages, h)

        # Build combined lookup: fitted cohorts first, then new cohorts
        n_new = len(new_cohort_values)
        if n_new > 0:
            all_c = np.concatenate([cohorts.astype(int), new_cohort_values.astype(int)])
        else:
            all_c = cohorts.astype(int)
        c_min = int(all_c.min())
        c_max = int(all_c.max())
        n_total = c_max - c_min + 1

        # Index grid: maps each (age, year) cell to position in all_c
        raw_idx = (cohort_grid_f - c_min).astype(int)
        # Sentinel for out-of-range: we'll handle separately
        in_range = (raw_idx >= 0) & (raw_idx < n_total)

        # Fitted cohort values: constant across simulations
        # Build a (n_total,) array mapping cohort offset → gc value for fitted
        gc_fitted_lookup = np.zeros(n_total + 1)  # +1 sentinel
        fitted_idx = (cohorts.astype(int) - c_min).astype(int)
        valid_fitted = (fitted_idx >= 0) & (fitted_idx < n_total)
        gc_fitted_lookup[fitted_idx[valid_fitted]] = gc_fit[valid_fitted]

        sentinel_idx = n_total
        safe_raw_idx = np.where(in_range, raw_idx, sentinel_idx)
        fitted_contribution = gc_fitted_lookup[safe_raw_idx]  # (n_ages, h)
        coh_contrib += b0x[:, None, None] * fitted_contribution[:, :, None]

        # New cohort values: vary across simulations
        if n_new > 0 and gc_s is not None:
            # gc_s: (n_new, nsim), new_cohort_values: (n_new,)
            new_c_offsets = (new_cohort_values - c_min).astype(int)
            # Build index: for each cell, which new cohort index (if any)?
            # Map raw_idx → index in new_cohort_values, or -1
            new_c_lookup = np.full(n_total + 1, -1, dtype=int)
            for k, offset in enumerate(new_c_offsets):
                if 0 <= offset < n_total:
                    new_c_lookup[offset] = k
            new_k = new_c_lookup[safe_raw_idx]  # (n_ages, h), values -1..n_new-1
            has_new = new_k >= 0  # (n_ages, h) mask

            if has_new.any():
                # Extract (cell_i, cell_j) where new cohorts apply
                ai, aj = np.where(has_new)
                k_vals = new_k[ai, aj]  # which new cohort index
                # gc_s[k_vals, :] → (n_cells, nsim)
                coh_contrib[ai, aj, :] += b0x[ai, None] * gc_s[k_vals, :]

    # Assemble: eta_base + kt_contrib + coh_contrib → rates
    eta_all = eta_base[:, :, None] + kt_contrib + coh_contrib  # (n_ages, h, nsim)

    rates = np.exp(np.clip(eta_all, -_CLIP, _CLIP)) if link == "log" else invlogit(eta_all)

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
