"""Residual bootstrap for GAPC mortality models.

Resamples deviance residuals from the fitted model, inverts them to obtain
bootstrap death counts, and refits.

References:
- Renshaw, A.E. & Haberman, S. (2008). On simulation-based approaches to
  risk measurement in mortality with specific reference to Poisson Lee-Carter
  modelling. *IME*, 42(2), 797–816.
- Debón, A., Montes, F. & Puig, F. (2008). Modelling and Forecasting Mortality
  in Spain. *European Journal of Operational Research*, 189(3), 624–637.
"""
from __future__ import annotations

import numpy as np

from ..diagnostics.residuals import deviance_residuals
from ..fit.fit_result import FitStMoMo
from .boot_result import BootStMoMo

_EPS = 1e-15


def residual_bootstrap(
    fit: FitStMoMo,
    nboot: int = 500,
    *,
    seed: int | None = None,
    n_jobs: int = 1,
) -> BootStMoMo:
    """Residual bootstrap for parameter uncertainty.

    Resamples deviance residuals from the fitted model, converts them back to
    death counts, and refits the model on each bootstrap sample.

    Parameters
    ----------
    fit:
        Fitted model to bootstrap.
    nboot:
        Number of bootstrap replicates.
    seed:
        Random seed.
    n_jobs:
        Number of parallel jobs (requires ``joblib``).

    Returns
    -------
    BootStMoMo
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=nboot).tolist()

    # Compute deviance residuals once
    res = deviance_residuals(fit)   # (n_ages, n_years), masked cells = 0

    def _one_replicate(s: int) -> FitStMoMo | None:
        rng_b = np.random.default_rng(s)
        try:
            D_boot = _resample_residuals(fit, res, rng_b)
            return fit.model.fit(
                D_boot, fit.Ext, fit.ages, fit.years,
                wxt=fit.wxt, oxt=fit.oxt,
            )
        except Exception:
            return None

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_one_replicate)(s) for s in seeds
            )
        except ImportError:
            results = [_one_replicate(s) for s in seeds]
    else:
        results = [_one_replicate(s) for s in seeds]

    valid_fits = [r for r in results if r is not None]
    return BootStMoMo(
        base_fit=fit,
        nboot=nboot,
        method="residual",
        fits=valid_fits,
    )


def _resample_residuals(
    fit: FitStMoMo,
    res: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample deviance residuals and invert to death counts."""
    mask = fit.wxt > 0
    active_res = res[mask]
    boot_res = rng.choice(active_res, size=active_res.shape, replace=True)

    D_boot = fit.Dxt.copy()
    D_boot_flat = D_boot[mask]
    fitted_flat = fit.fitted_deaths[mask]

    if fit.model.link == "log":
        # Invert Poisson deviance residual: r = sign(D - D̂) * sqrt(deviance_contrib)
        # D* = D̂ * exp(r * sqrt(D̂/D̂)) ... use Haberman approximation:
        # D* ≈ (sqrt(D̂) + r/2)^2  (sign-preserving)
        D_star = (np.sqrt(np.maximum(fitted_flat, _EPS)) + boot_res / 2.0) ** 2
        D_boot_flat = np.maximum(D_star, 0.0)
    else:
        # Binomial: invert via D* = q_hat * E + r * sqrt(E * q̂ * (1-q̂))
        q_hat = np.clip(fit.fitted_rates[mask], _EPS, 1 - _EPS)
        E_flat = fit.Ext[mask]
        se = np.sqrt(E_flat * q_hat * (1 - q_hat))
        D_boot_flat = np.clip(E_flat * q_hat + boot_res * se, 0, E_flat)

    D_boot[mask] = D_boot_flat
    return D_boot
