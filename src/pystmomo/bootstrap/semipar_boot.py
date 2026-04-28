"""Semiparametric bootstrap for GAPC mortality models.

Generates death counts from assumed distributional form (Poisson or Binomial)
with expected values equal to the fitted deaths from the base model.

Reference: Brouhns, N., Denuit, M. & Vermunt, J.K. (2005). Measuring the
Longevity Risk in Mortality Projections. *Bulletin of the Swiss Association
of Actuaries*, 2, 105–130.
"""
from __future__ import annotations

import numpy as np

from ..fit.fit_result import FitStMoMo
from .boot_result import BootStMoMo


def semiparametric_bootstrap(
    fit: FitStMoMo,
    nboot: int = 500,
    *,
    seed: int | None = None,
    n_jobs: int = 1,
) -> BootStMoMo:
    """Semiparametric bootstrap for parameter uncertainty.

    Resamples death counts from their assumed distribution (Poisson or
    Binomial) conditional on the fitted expected deaths, then refits the
    model on each bootstrap sample.

    Parameters
    ----------
    fit:
        Fitted model to bootstrap.
    nboot:
        Number of bootstrap replicates.
    seed:
        Random seed.
    n_jobs:
        Number of parallel jobs.  Requires ``joblib`` (``pip install
        pystmomo[parallel]``).  Defaults to 1 (sequential).

    Returns
    -------
    BootStMoMo

    Examples
    --------
    >>> from pystmomo import lc, load_ew_male, semiparametric_bootstrap
    >>> data = load_ew_male()
    >>> fit = lc().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    >>> boot = semiparametric_bootstrap(fit, nboot=100, seed=0)
    >>> lo, hi = boot.parameter_ci("kt", level=0.95)
    >>> lo.shape
    (1, 51)
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=nboot).tolist()

    def _one_replicate(s: int) -> FitStMoMo | None:
        rng_b = np.random.default_rng(s)
        try:
            D_boot = _sample_deaths(fit, rng_b)
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
    boot = BootStMoMo(
        base_fit=fit,
        nboot=nboot,
        method="semiparametric",
        fits=valid_fits,
    )
    return boot


def _sample_deaths(fit: FitStMoMo, rng: np.random.Generator) -> np.ndarray:
    """Sample death counts from the model's assumed distribution."""
    if fit.model.link == "log":
        return rng.poisson(np.maximum(fit.fitted_deaths, 0.0)).astype(float)
    else:
        # Binomial: D* ~ Binomial(E, q̂)
        n = np.round(fit.Ext).astype(int)
        q = np.clip(fit.fitted_rates, 1e-15, 1 - 1e-15)
        D_boot = rng.binomial(np.maximum(n, 0), q).astype(float)
        return D_boot
