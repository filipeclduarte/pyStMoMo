"""FitStMoMo result class."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..bootstrap.boot_result import BootStMoMo
    from ..core.stmomo import StMoMo
    from ..forecast.forecast_result import ForStMoMo
    from ..simulate.sim_result import SimStMoMo


class FitStMoMo:
    """Result of fitting a :class:`~pystmomo.core.StMoMo` model.

    Attributes
    ----------
    model:
        The StMoMo model specification that was fitted.
    ax:
        Static age function coefficients α_x, shape (n_ages,), or None.
    bx:
        Period age-modulating functions β_x, shape (n_ages, N).
        For parametric models, this stores the age-function evaluations.
        For non-parametric models (LC, RH), this is the freely-fitted β_x.
    kt:
        Period indexes κ_t, shape (N, n_years).
    b0x:
        Cohort age-modulating function β_x^(0), shape (n_ages,), or None.
    gc:
        Cohort index γ_c, shape (n_cohorts,), or None.
    Dxt:
        Observed deaths, shape (n_ages, n_years).
    Ext:
        Exposures, shape (n_ages, n_years).
    wxt:
        Binary weight matrix, shape (n_ages, n_years).
    oxt:
        Log-exposure offset, shape (n_ages, n_years).
    ages, years, cohorts:
        Age, year, and cohort vectors.
    fitted_rates:
        Fitted mortality rates μ_xt (Poisson) or q_xt (Binomial).
    fitted_deaths:
        Fitted expected deaths E_xt · μ_xt.
    loglik:
        Maximised log-likelihood.
    deviance:
        Model deviance.
    npar:
        Number of estimated parameters.
    nobs:
        Number of observations (weighted cells).
    converged:
        Whether the optimisation converged.
    n_iter:
        Number of IRLS iterations (bilinear path only; -1 for GLM path).
    """

    def __init__(
        self,
        model: StMoMo,
        ax: np.ndarray | None,
        bx: np.ndarray,
        kt: np.ndarray,
        b0x: np.ndarray | None,
        gc: np.ndarray | None,
        Dxt: np.ndarray,
        Ext: np.ndarray,
        wxt: np.ndarray,
        oxt: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        cohorts: np.ndarray,
        fitted_rates: np.ndarray,
        fitted_deaths: np.ndarray,
        loglik: float,
        deviance: float,
        npar: int,
        nobs: int,
        converged: bool = True,
        n_iter: int = -1,
    ) -> None:
        self.model = model
        self.ax = ax
        self.bx = bx
        self.kt = kt
        self.b0x = b0x
        self.gc = gc
        self.Dxt = Dxt
        self.Ext = Ext
        self.wxt = wxt
        self.oxt = oxt
        self.ages = ages
        self.years = years
        self.cohorts = cohorts
        self.fitted_rates = fitted_rates
        self.fitted_deaths = fitted_deaths
        self.loglik = loglik
        self.deviance = deviance
        self.npar = npar
        self.nobs = nobs
        self.converged = converged
        self.n_iter = n_iter

    # ------------------------------------------------------------------
    # Convenience statistics
    # ------------------------------------------------------------------

    def aic(self) -> float:
        """Akaike Information Criterion: -2·loglik + 2·npar."""
        return -2.0 * self.loglik + 2.0 * self.npar

    def bic(self) -> float:
        """Bayesian Information Criterion: -2·loglik + npar·log(nobs)."""
        return -2.0 * self.loglik + self.npar * np.log(self.nobs)

    def residuals(
        self, kind: Literal["deviance", "pearson", "response"] = "deviance"
    ) -> np.ndarray:
        """Compute residuals.

        Parameters
        ----------
        kind:
            Type of residuals: ``"deviance"``, ``"pearson"``, or ``"response"``.

        Returns
        -------
        np.ndarray
            Residual matrix, shape (n_ages, n_years).  Masked cells are 0.
        """
        from ..diagnostics.residuals import (
            deviance_residuals,
            pearson_residuals,
            response_residuals,
        )

        if kind == "deviance":
            return deviance_residuals(self)
        elif kind == "pearson":
            return pearson_residuals(self)
        elif kind == "response":
            return response_residuals(self)
        else:
            raise ValueError(f"Unknown residual kind: {kind!r}")

    # ------------------------------------------------------------------
    # Downstream methods (delegate to submodules)
    # ------------------------------------------------------------------

    def forecast(
        self,
        h: int = 50,
        *,
        kt_method: Literal["mrwd", "arima"] = "mrwd",
        gc_method: Literal["arima", "mrwd"] = "arima",
        level: float = 0.95,
        **kwargs,
    ) -> ForStMoMo:
        """Forecast future mortality rates.  See :func:`~pystmomo.forecast.forecast`."""
        from ..forecast.forecast import forecast
        return forecast(self, h=h, kt_method=kt_method, gc_method=gc_method,
                        level=level, **kwargs)

    def simulate(
        self,
        nsim: int = 1000,
        h: int = 50,
        *,
        seed: int | None = None,
        **kwargs,
    ) -> SimStMoMo:
        """Simulate future mortality trajectories.  See :func:`~pystmomo.simulate.simulate`."""
        from ..simulate.simulate import simulate
        return simulate(self, nsim=nsim, h=h, seed=seed, **kwargs)

    def bootstrap(
        self,
        nboot: int = 500,
        *,
        method: Literal["semiparametric", "residual"] = "semiparametric",
        **kwargs,
    ) -> BootStMoMo:
        """Bootstrap parameter uncertainty.  See :func:`~pystmomo.bootstrap`."""
        from ..bootstrap.residual_boot import residual_bootstrap
        from ..bootstrap.semipar_boot import semiparametric_bootstrap
        if method == "semiparametric":
            return semiparametric_bootstrap(self, nboot=nboot, **kwargs)
        else:
            return residual_bootstrap(self, nboot=nboot, **kwargs)

    def __repr__(self) -> str:
        conv = "converged" if self.converged else "NOT converged"
        return (
            f"FitStMoMo(\n"
            f"  model = {self.model}\n"
            f"  ages  = {self.ages[0]}–{self.ages[-1]}  ({len(self.ages)} ages)\n"
            f"  years = {self.years[0]}–{self.years[-1]}  ({len(self.years)} years)\n"
            f"  nobs  = {self.nobs},  npar = {self.npar}\n"
            f"  loglik = {self.loglik:.2f},  deviance = {self.deviance:.2f}\n"
            f"  AIC = {self.aic():.2f},  BIC = {self.bic():.2f}\n"
            f"  {conv}\n"
            f")"
        )
