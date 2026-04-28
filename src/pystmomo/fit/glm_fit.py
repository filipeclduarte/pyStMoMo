"""Parametric GAPC fitting via GLM / IRLS (Path A).

Used for models where all age functions are parametric: CBD, APC, M6, M7, M8.

Full-rank designs (CBD, no-cohort models):
    Standard statsmodels GLM IRLS.

Rank-deficient designs (APC, M6, M7, M8 — static age + period + cohort):
    Hand-coded IRLS with the minimum-norm solution at each step (pinv).
    This matches the behaviour of R's gnm package.
"""
from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm

from ..core.design import build_design_matrix, unpack_params
from ..core.stmomo import StMoMo
from .families import binomial_deviance, binomial_loglik, poisson_deviance, poisson_loglik
from .fit_result import FitStMoMo

_CLIP = 30.0
_EPS = 1e-15


def _is_rank_deficient(model: StMoMo) -> bool:
    """True when the design matrix may be rank-deficient.

    Any model with a cohort age function is potentially rank-deficient because
    the cohort anti-diagonals are structurally related to the age and period
    dimensions.  We use the pseudoinverse IRLS path for all such models.
    """
    return model.cohort_age_fun is not None


def _irls_pinv(
    X: np.ndarray,
    y: np.ndarray,
    link: str,
    offset: np.ndarray | None,
    freq_weights: np.ndarray,
    Ext: np.ndarray,
    maxiter: int = 300,
    tol: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """Hand-coded IRLS with minimum-norm (pseudoinverse) solver.

    Suitable for rank-deficient design matrices (APC, M6, M7, M8).

    Returns
    -------
    params:
        Minimum-norm parameter vector.
    converged:
        Whether IRLS converged.
    """
    n_obs, n_params = X.shape
    offset_ = offset if offset is not None else np.zeros(n_obs)

    # Initialise with OLS on the transformed response
    if link == "log":
        # log(y + 0.5) minus offset as starting eta
        eta = np.log(np.maximum(y + 0.5, _EPS)) - offset_
    else:
        q0 = np.clip(y / np.maximum(Ext, _EPS), _EPS, 1 - _EPS)
        eta = np.log(q0 / (1 - q0))

    params, _, _, _ = np.linalg.lstsq(X, eta, rcond=None)
    prev_dev = np.inf
    converged = False

    for it in range(maxiter):
        linear_pred = X @ params + offset_

        if link == "log":
            mu = np.exp(np.clip(linear_pred, -_CLIP, _CLIP))    # expected deaths
            # Poisson: V = mu, link derivative = 1/mu
            V = mu
            dmu_deta = mu                                         # d(mu)/d(eta) = mu for log
        else:
            q = 1.0 / (1.0 + np.exp(-np.clip(linear_pred, -_CLIP, _CLIP)))
            mu = Ext * q                                          # expected deaths
            V = Ext * q * (1 - q)                                # Binomial variance
            dmu_deta = V                                          # d(Eq)/d(eta)

        # Adjusted dependent variable (working response)
        resid = y - mu
        z = linear_pred + resid / np.maximum(dmu_deta, _EPS) - offset_

        # IRLS weights
        W = freq_weights * np.maximum(V, _EPS)

        # Weighted design matrix
        sqW = np.sqrt(W)
        Xw = sqW[:, None] * X
        zw = sqW * z

        # Truncated-SVD pseudoinverse (minimum-norm, handles rank deficiency)
        # rcond=1e-10 zeroes singular values < 1e-10 * max(s)
        new_params, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=1e-10)

        # Convergence on deviance
        if link == "log":
            dev = poisson_deviance(y, np.exp(np.clip(X @ new_params + offset_, -_CLIP, _CLIP)), freq_weights)
        else:
            q_new = 1.0 / (1.0 + np.exp(-np.clip(X @ new_params + offset_, -_CLIP, _CLIP)))
            dev = binomial_deviance(y, q_new, Ext, freq_weights)

        rel_change = abs(dev - prev_dev) / (abs(prev_dev) + _EPS)
        params = new_params
        prev_dev = dev

        if rel_change < tol and it > 0:
            converged = True
            break

    return params, converged


def fit_parametric(
    model: StMoMo,
    Dxt: np.ndarray,
    Ext: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    cohorts: np.ndarray,
    wxt: np.ndarray,
    oxt: np.ndarray,
) -> FitStMoMo:
    """Fit a fully-parametric GAPC model.

    Full-rank models (CBD, no-cohort): statsmodels GLM.
    Rank-deficient models (APC, M6, M7, M8): IRLS with pseudoinverse.

    Parameters
    ----------
    model:
        A ``StMoMo`` with ``is_fully_parametric == True``.
    Dxt, Ext, ages, years, cohorts, wxt, oxt:
        Mortality data and metadata.

    Returns
    -------
    FitStMoMo
    """
    X, col_map, row_mask = build_design_matrix(model, ages, years, cohorts, wxt)
    n_ages, n_years = len(ages), len(years)

    y_flat = Dxt.ravel()
    Ext_flat = Ext.ravel()
    w_flat = wxt.ravel()
    oxt_flat = oxt.ravel()

    y_obs = y_flat[row_mask]
    Ext_obs = Ext_flat[row_mask]
    w_obs = w_flat[row_mask]
    oxt_obs = oxt_flat[row_mask]

    X_dense = np.asarray(X.todense())
    n_obs, n_params = X_dense.shape

    rank_deficient = _is_rank_deficient(model)

    if rank_deficient:
        params, sm_converged = _irls_pinv(
            X_dense, y_obs, model.link, oxt_obs if model.link == "log" else None,
            w_obs, Ext_obs, maxiter=300, tol=1e-8,
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model.link == "log":
                family = sm.families.Poisson(link=sm.families.links.Log())
                glm_result = sm.GLM(
                    endog=y_obs,
                    exog=X_dense,
                    family=family,
                    offset=oxt_obs,
                    freq_weights=w_obs,
                ).fit(method="IRLS", maxiter=300, tol=1e-8, disp=False)
            else:
                family = sm.families.Binomial(link=sm.families.links.Logit())
                q_obs = y_obs / np.maximum(Ext_obs, 1e-15)
                glm_result = sm.GLM(
                    endog=q_obs,
                    exog=X_dense,
                    family=family,
                    freq_weights=Ext_obs * w_obs,
                ).fit(method="IRLS", maxiter=300, tol=1e-8, disp=False)
            params = glm_result.params
            sm_converged = glm_result.converged

    ax_raw, _, kt, _, gc = unpack_params(params, col_map, ages, years, cohorts)

    # For parametric models, bx stores the age-function evaluations
    N = model.N
    bx = (
        np.column_stack([model.period_age_fun[i](ages) for i in range(N)])
        if N > 0 else np.empty((n_ages, 0))
    )

    # b0x: evaluated cohort age-modulating function (for display; absorbed into design)
    b0x: np.ndarray | None = (
        model.cohort_age_fun(ages) if model.cohort_age_fun is not None else None
    )

    ax = ax_raw

    # Apply identifiability constraints
    if model.const_fun is not None:
        ax, bx, kt, b0x, gc = model.const_fun(ax, bx, kt, b0x, gc, ages, years, cohorts)

    # Reconstruct linear predictor (excluding offset)
    eta_linear = np.zeros((n_ages, n_years))
    if ax is not None:
        eta_linear += ax[:, None]
    if bx.size > 0 and kt.size > 0:
        eta_linear += bx @ kt
    if gc is not None and b0x is not None:
        from ..core.predictor import _cohort_index_matrix
        cohort_mat = _cohort_index_matrix(ages, years, cohorts, gc)
        eta_linear += b0x[:, None] * cohort_mat

    if model.link == "log":
        fitted_rates = np.exp(np.clip(eta_linear, -_CLIP, _CLIP))
        fitted_deaths = fitted_rates * Ext
    else:
        fitted_rates = 1.0 / (1.0 + np.exp(-np.clip(eta_linear, -_CLIP, _CLIP)))
        fitted_deaths = fitted_rates * Ext

    fitted_deaths = np.where(wxt > 0, fitted_deaths, 0.0)
    fitted_rates = np.where(wxt > 0, fitted_rates, 0.0)

    if model.link == "log":
        loglik = poisson_loglik(Dxt, fitted_deaths, wxt)
        deviance = poisson_deviance(Dxt, fitted_deaths, wxt)
    else:
        loglik = binomial_loglik(Dxt, fitted_rates, Ext, wxt)
        deviance = binomial_deviance(Dxt, fitted_rates, Ext, wxt)

    nobs = int(wxt.sum())

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
        npar=n_params,
        nobs=nobs,
        converged=sm_converged,
        n_iter=-1,
    )
