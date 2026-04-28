"""Residual computations for fitted StMoMo models."""
from __future__ import annotations

import numpy as np

from ..fit.fit_result import FitStMoMo

_EPS = 1e-15


def deviance_residuals(fit: FitStMoMo) -> np.ndarray:
    """Signed sqrt of pointwise deviance contribution, shape (n_ages, n_years).

    For Poisson (link='log'):
      - D > 0: sign(D - D̂) * sqrt(2*(D*log(D/D̂) - (D - D̂))) * sqrt(wxt)
      - D = 0: -sqrt(2 * D̂) * sqrt(wxt)   [sign = -1, contribution = 2*D̂]

    For Binomial (link='logit'):
      sign(D - E*q̂) * sqrt(2*(D*log(D/(E*q̂)) + (E-D)*log((E-D)/(E*(1-q̂))))) * sqrt(wxt)

    Cells with wxt=0 are set to 0.

    Note: wxt is binary 0/1, so sqrt(wxt) == wxt, but we use sqrt(wxt)
    so that squared residuals sum to the weighted deviance.
    """
    D = fit.Dxt
    Dhat = fit.fitted_deaths
    E = fit.Ext
    wxt = fit.wxt
    link = fit.model.link

    out = np.zeros_like(D, dtype=float)
    sqrt_w = np.sqrt(np.maximum(wxt, 0.0))

    if link == "log":
        # Poisson
        pos = wxt > 0

        # D > 0
        pos_d = pos & (D > 0)
        d = D[pos_d]
        dhat = np.maximum(Dhat[pos_d], _EPS)
        contrib = 2.0 * (d * np.log(d / dhat) - (d - dhat))
        out[pos_d] = np.sign(d - dhat) * np.sqrt(np.maximum(contrib, 0.0)) * sqrt_w[pos_d]

        # D == 0
        pos_zero = pos & (D <= 0)
        dhat0 = np.maximum(Dhat[pos_zero], 0.0)
        out[pos_zero] = -np.sqrt(2.0 * dhat0) * sqrt_w[pos_zero]

    else:
        # Binomial
        qhat = np.clip(fit.fitted_rates, _EPS, 1.0 - _EPS)
        Eqhat = E * qhat  # = Dhat
        pos = wxt > 0

        d = D[pos]
        e = E[pos]
        q = qhat[pos]
        eqhat = Eqhat[pos]
        e_minus_d = np.maximum(e - d, 0.0)
        e_minus_eqhat = e * (1.0 - q)

        # obs_q for log term; handle d=0 and d=e separately
        log_term1 = np.where(d > 0, d * np.log(np.maximum(d / np.maximum(eqhat, _EPS), _EPS)), 0.0)
        log_term2 = np.where(
            e_minus_d > 0,
            e_minus_d * np.log(np.maximum(e_minus_d / np.maximum(e_minus_eqhat, _EPS), _EPS)),
            0.0,
        )
        contrib = 2.0 * (log_term1 + log_term2)
        out[pos] = (
            np.sign(d - eqhat) * np.sqrt(np.maximum(contrib, 0.0)) * sqrt_w[pos]
        )

    return out


def pearson_residuals(fit: FitStMoMo) -> np.ndarray:
    """Pearson residuals, shape (n_ages, n_years).

    For Poisson: (D - D̂) / sqrt(D̂) * sqrt(wxt)
    For Binomial: (D - E*q̂) / sqrt(E*q̂*(1-q̂)) * sqrt(wxt)

    Cells with wxt=0 or zero variance are set to 0.
    """
    D = fit.Dxt
    Dhat = fit.fitted_deaths
    E = fit.Ext
    wxt = fit.wxt
    link = fit.model.link

    out = np.zeros_like(D, dtype=float)
    sqrt_w = np.sqrt(np.maximum(wxt, 0.0))
    pos = wxt > 0

    if link == "log":
        # Poisson: variance = D̂
        dhat = np.maximum(Dhat[pos], _EPS)
        out[pos] = (D[pos] - Dhat[pos]) / np.sqrt(dhat) * sqrt_w[pos]
    else:
        # Binomial: variance = E * q̂ * (1 - q̂)
        qhat = np.clip(fit.fitted_rates[pos], _EPS, 1.0 - _EPS)
        e = E[pos]
        var = e * qhat * (1.0 - qhat)
        out[pos] = (D[pos] - e * qhat) / np.sqrt(np.maximum(var, _EPS)) * sqrt_w[pos]

    return out


def response_residuals(fit: FitStMoMo) -> np.ndarray:
    """Raw (response) residuals, shape (n_ages, n_years).

    (D - D̂) * wxt
    """
    return (fit.Dxt - fit.fitted_deaths) * fit.wxt
