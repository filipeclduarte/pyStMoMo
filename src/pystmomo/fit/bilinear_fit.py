"""Bilinear IRLS fitting for LC and RH models (Path B).

Block-coordinate IRLS: alternately update α_x, β_x, κ_t, (β_x^(0), γ_c).
Convergence is measured on relative deviance change.

For Poisson log link:
    η_xt  = log(E_xt) + α_x + β_x κ_t          [offset included in η]
    μ̂_xt  = exp(η_xt) = E_xt * exp(α_x + β_x κ_t)  [expected deaths]

Score for θ: Σ f(x,t) (D_xt - μ̂_xt)
Info for θ:  Σ f(x,t)² μ̂_xt

For Binomial logit link:
    η_xt  = α_x + β_x κ_t                        [no offset in η]
    q̂_xt  = invlogit(η_xt)                        [probability]
    Fitted deaths: E_xt * q̂_xt

Score for θ: Σ f(x,t) (D_xt - E_xt q̂_xt)
Info for θ:  Σ f(x,t)² E_xt q̂_xt (1 - q̂_xt)
"""
from __future__ import annotations

import numpy as np

from ..core.age_functions import NonParametricAgeFun
from ..core.predictor import invlogit
from ..core.stmomo import StMoMo
from .families import (
    binomial_deviance,
    binomial_loglik,
    poisson_deviance,
    poisson_loglik,
)
from .fit_result import FitStMoMo
from .starting_values import svd_starting_values

_CLIP = 30.0
_EPS = 1e-15


def _expected_deaths(eta: np.ndarray, link: str, Ext: np.ndarray) -> np.ndarray:
    """Compute expected deaths from linear predictor.

    Poisson log: eta includes log(E), so exp(eta) = E*μ = expected deaths.
    Binomial logit: eta excludes offset, q = invlogit(eta), deaths = E*q.
    """
    if link == "log":
        return np.exp(np.clip(eta, -_CLIP, _CLIP))
    return Ext * invlogit(eta)


def _rates(eta: np.ndarray, link: str, Ext: np.ndarray) -> np.ndarray:
    """Compute mortality rates from linear predictor."""
    if link == "log":
        return np.exp(np.clip(eta - np.log(np.maximum(Ext, _EPS)), -_CLIP, _CLIP))
    return invlogit(eta)


def _newton_step(
    obs: np.ndarray,       # deaths for this slice
    edeath: np.ndarray,    # expected deaths for this slice
    covariate: np.ndarray, # age function values for this slice
    weight: np.ndarray,    # wxt for this slice
    link: str,
    Ext_slice: np.ndarray, # exposures for this slice
) -> float:
    """Single Newton-Raphson step for a scalar GLM coefficient.

    Score: Σ_i w_i · f_i · (D_i - D̂_i)
    Info:  Σ_i w_i · f_i² · V_i
    where V_i = D̂_i  (Poisson) or  E_i · q̂_i · (1-q̂_i)  (Binomial).
    """
    num = np.sum(weight * covariate * (obs - edeath))
    if link == "log":
        denom = np.sum(weight * covariate ** 2 * edeath)
    else:
        q = edeath / np.maximum(Ext_slice, _EPS)
        q = np.clip(q, _EPS, 1 - _EPS)
        denom = np.sum(weight * covariate ** 2 * Ext_slice * q * (1 - q))
    if abs(denom) < _EPS:
        return 0.0
    return float(num / denom)


def fit_bilinear(
    model: StMoMo,
    Dxt: np.ndarray,
    Ext: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    cohorts: np.ndarray,
    wxt: np.ndarray,
    oxt: np.ndarray,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    verbose: bool = False,
) -> FitStMoMo:
    """Fit a bilinear GAPC model via block-coordinate IRLS."""
    n_ages, n_years = len(ages), len(years)
    n_cohorts = len(cohorts)
    link = model.link

    has_cohort = model.has_cohort
    cohort_np = has_cohort and isinstance(model.cohort_age_fun, NonParametricAgeFun)
    cohort_map = {int(c): j for j, c in enumerate(cohorts)}
    cohort_grid = years[None, :] - ages[:, None]  # (n_ages, n_years)

    # --- SVD initialisation ---
    n_components = 1
    ax, bx, kt = svd_starting_values(Dxt, Ext, ages, years, wxt, link, n_components)

    b0x: np.ndarray | None = None
    gc: np.ndarray | None = None
    if has_cohort:
        b0x = np.ones(n_ages) * 0.01
        gc = np.zeros(n_cohorts)

    # --- Assemble initial η ---
    def _build_eta() -> np.ndarray:
        eta = ax[:, None] + bx @ kt                    # (n_ages, n_years)
        if b0x is not None and gc is not None:
            # Vectorised cohort contribution
            c_min = int(cohorts[0])
            gc_pad = np.append(gc, 0.0)  # sentinel for out-of-range
            idx = (cohort_grid - c_min).astype(int)
            sentinel = len(gc)
            idx = np.where((idx >= 0) & (idx < len(gc)), idx, sentinel)
            eta += b0x[:, None] * gc_pad[idx]
        if link == "log":
            eta += oxt                                  # add log(E) for Poisson
        return eta

    eta = _build_eta()
    edeath = _expected_deaths(eta, link, Ext)

    def _deviance() -> float:
        if link == "log":
            return poisson_deviance(Dxt, edeath, wxt)
        return binomial_deviance(Dxt, edeath / np.maximum(Ext, _EPS), Ext, wxt)

    prev_dev = _deviance()
    converged = False
    n_iter = 0

    for iteration in range(max_iter):
        # --- Update α_x ---
        for i in range(n_ages):
            if wxt[i].sum() == 0:
                continue
            # Remove α_x contribution from η for this row
            eta_no_ax = eta[i] - ax[i]
            ed_row = _expected_deaths(eta_no_ax + ax[i], link, Ext[i])
            step = _newton_step(
                Dxt[i], ed_row, np.ones(n_years), wxt[i], link, Ext[i]
            )
            ax[i] += step
            eta[i] = eta_no_ax + ax[i]
            edeath[i] = _expected_deaths(eta[i], link, Ext[i])

        # --- Update κ_t ---
        for j in range(n_years):
            if wxt[:, j].sum() == 0:
                continue
            b = bx[:, 0]                              # β_x vector
            eta_no_kt = eta[:, j] - b * kt[0, j]
            ed_col = _expected_deaths(eta_no_kt + b * kt[0, j], link, Ext[:, j])
            step = _newton_step(
                Dxt[:, j], ed_col, b, wxt[:, j], link, Ext[:, j]
            )
            kt[0, j] += step
            eta[:, j] = eta_no_kt + b * kt[0, j]
            edeath[:, j] = _expected_deaths(eta[:, j], link, Ext[:, j])

        # --- Update β_x ---
        for i in range(n_ages):
            if wxt[i].sum() == 0:
                continue
            k = kt[0]                                 # κ_t vector
            eta_no_bx = eta[i] - bx[i, 0] * k
            ed_row = _expected_deaths(eta_no_bx + bx[i, 0] * k, link, Ext[i])
            step = _newton_step(
                Dxt[i], ed_row, k, wxt[i], link, Ext[i]
            )
            bx[i, 0] += step
            eta[i] = eta_no_bx + bx[i, 0] * k
            edeath[i] = _expected_deaths(eta[i], link, Ext[i])

        # --- Update cohort parameters (RH) ---
        if has_cohort and b0x is not None and gc is not None:
            # Update γ_c
            for cj, c_val in enumerate(cohorts):
                mask = cohort_grid == int(c_val)
                rows_in_c, cols_in_c = np.where(mask)
                if len(rows_in_c) == 0 or wxt[mask].sum() == 0:
                    continue
                b0x_c = b0x[rows_in_c]  # β_x^(0) for cells in this cohort
                eta_c = eta[mask]
                eta_no_gc_c = eta_c - b0x_c * gc[cj]
                ed_c = _expected_deaths(eta_no_gc_c + b0x_c * gc[cj], link, Ext[mask])
                step = _newton_step(
                    Dxt[mask], ed_c, b0x_c, wxt[mask], link, Ext[mask]
                )
                gc[cj] += step
                # Update η for cells in this cohort
                for ii, jj in zip(rows_in_c, cols_in_c, strict=False):
                    eta[ii, jj] += b0x[ii] * step
                    edeath[ii, jj] = _expected_deaths(
                        np.array([eta[ii, jj]]), link, np.array([Ext[ii, jj]])
                    )[0]

            # Update β_x^(0) (non-parametric)
            if cohort_np:
                for i in range(n_ages):
                    if wxt[i].sum() == 0:
                        continue
                    gc_row = np.array([
                        gc[cohort_map[int(cohort_grid[i, j])]]
                        if int(cohort_grid[i, j]) in cohort_map else 0.0
                        for j in range(n_years)
                    ])
                    eta_no_b0x = eta[i] - b0x[i] * gc_row
                    ed_row = _expected_deaths(eta_no_b0x + b0x[i] * gc_row, link, Ext[i])
                    step = _newton_step(
                        Dxt[i], ed_row, gc_row, wxt[i], link, Ext[i]
                    )
                    b0x[i] += step
                    eta[i] = eta_no_b0x + b0x[i] * gc_row
                    edeath[i] = _expected_deaths(eta[i], link, Ext[i])

        # --- Convergence check ---
        dev = _deviance()
        rel_change = abs(dev - prev_dev) / (abs(prev_dev) + _EPS)
        n_iter = iteration + 1

        if verbose:
            print(f"Iter {n_iter:4d}: deviance = {dev:.4f}, rel_change = {rel_change:.2e}")

        if rel_change < tol:
            converged = True
            break
        prev_dev = dev

    # --- Apply identifiability constraints ---
    if model.const_fun is not None:
        ax, bx, kt, b0x, gc = model.const_fun(
            ax, bx, kt, b0x, gc, ages, years, cohorts
        )

    # --- Final rates and log-likelihood ---
    # Recompute eta after possible constraint changes (using vectorised _build_eta)
    eta_final = _build_eta()
    # For Poisson, _build_eta already adds oxt; for Binomial it does not.
    # Split: eta_log includes offset, eta_final (for Binomial) does not.
    if link == "log":
        eta_log = eta_final  # _build_eta already added oxt for log link
        eta_no_offset = eta_final - oxt
    else:
        eta_log = eta_final + oxt
        eta_no_offset = eta_final

    if link == "log":
        fitted_deaths = np.exp(np.clip(eta_log, -_CLIP, _CLIP))
        fitted_rates = fitted_deaths / np.maximum(Ext, _EPS)
        loglik = poisson_loglik(Dxt, fitted_deaths, wxt)
        deviance = poisson_deviance(Dxt, fitted_deaths, wxt)
    else:
        fitted_rates = invlogit(eta_final)
        fitted_deaths = fitted_rates * Ext
        loglik = binomial_loglik(Dxt, fitted_rates, Ext, wxt)
        deviance = binomial_deviance(Dxt, fitted_rates, Ext, wxt)

    fitted_deaths = np.where(wxt > 0, fitted_deaths, 0.0)
    fitted_rates = np.where(wxt > 0, fitted_rates, 0.0)

    nobs = int(wxt.sum())
    npar = n_ages + n_ages + n_years  # ax + bx + kt
    if has_cohort and cohort_np:
        npar += n_ages + n_cohorts   # b0x + gc
    elif has_cohort:
        npar += n_cohorts             # gc only (b0x parametric)

    return FitStMoMo(
        model=model,
        ax=ax,
        bx=bx,
        kt=kt,
        b0x=b0x,
        gc=gc,
        Dxt=Dxt,
        Ext=Ext,
        wxt=wxt,
        oxt=oxt,
        ages=ages,
        years=years,
        cohorts=cohorts,
        fitted_rates=fitted_rates,
        fitted_deaths=fitted_deaths,
        loglik=loglik,
        deviance=deviance,
        npar=npar,
        nobs=nobs,
        converged=converged,
        n_iter=n_iter,
    )
