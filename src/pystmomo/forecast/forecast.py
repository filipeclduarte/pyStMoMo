"""Top-level forecast() function for StMoMo fitted models."""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..fit.fit_result import FitStMoMo
from .external import ExternalKtForecaster
from .forecast_result import ForStMoMo


def forecast(
    fit: FitStMoMo,
    h: int = 50,
    *,
    kt_method: Literal['mrwd', 'arima'] | ExternalKtForecaster = "mrwd",
    kt_arima_order: tuple = (0, 1, 0),
    gc_method: Literal['arima', 'mrwd'] | ExternalKtForecaster = "arima",
    gc_arima_order: tuple = (1, 1, 0),
    jump_choice: Literal["fit", "actual"] = "fit",
    level: float = 0.95,
) -> ForStMoMo:
    """Forecast future mortality rates from a fitted StMoMo model.

    Parameters
    ----------
    fit:
        A fitted :class:`~pystmomo.fit.FitStMoMo` object.
    h:
        Forecast horizon (number of future years).
    kt_method:
        Method for projecting period indexes: ``"mrwd"`` (multivariate random
        walk with drift) or ``"arima"`` (independent ARIMA per row).
    kt_arima_order:
        ARIMA (p, d, q) order used when ``kt_method="arima"``.
    gc_method:
        Method for projecting cohort indexes: ``"arima"`` or ``"mrwd"``.
    gc_arima_order:
        ARIMA order used when ``gc_method="arima"``.
    jump_choice:
        Starting-point convention.  ``"fit"`` uses the fitted end-point
        (default).  ``"actual"`` is not yet implemented.
    level:
        Confidence level for prediction intervals (e.g. 0.95).

    Returns
    -------
    ForStMoMo
        Forecast result object.
    """
    if jump_choice != "fit":
        raise NotImplementedError(
            "jump_choice='actual' is not yet implemented; use jump_choice='fit'."
        )

    # ------------------------------------------------------------------ #
    # 1.  Future years
    # ------------------------------------------------------------------ #
    years_f = np.arange(int(fit.years[-1]) + 1, int(fit.years[-1]) + h + 1)

    # ------------------------------------------------------------------ #
    # 2.  Forecast period indexes (kt)
    # ------------------------------------------------------------------ #
    kt_model = _fit_kt_model(fit.kt, kt_method, kt_arima_order)
    kt_f, kt_f_lower, kt_f_upper = kt_model.forecast(h, level=level)

    # ------------------------------------------------------------------ #
    # 3.  Forecast cohort indexes (gc) if the model has a cohort term
    # ------------------------------------------------------------------ #
    gc_model = None
    gc_f = None
    gc_f_lower = None
    gc_f_upper = None
    cohorts_f = np.array([], dtype=int)

    if fit.model.has_cohort and fit.gc is not None and len(fit.gc) > 0:
        # New cohorts needed: those born after the last observed cohort and
        # required to cover the future grid (ages[0] to ages[-1], years_f).
        # The latest cohort needed is  years_f[-1] - ages[0].
        max_obs_cohort = int(fit.cohorts[-1])
        max_needed_cohort = int(years_f[-1]) - int(fit.ages[0])
        if max_needed_cohort > max_obs_cohort:
            cohorts_f = np.arange(max_obs_cohort + 1, max_needed_cohort + 1)
            n_new = len(cohorts_f)
            # Fit gc time-series model on the observed cohort index
            gc_series = _clean_gc_series(fit.gc)
            gc_model = _fit_gc_model(gc_series, gc_method, gc_arima_order, n_new)
            gc_fc_mean, gc_fc_lower, gc_fc_upper = gc_model.forecast(n_new, level=level)
            # gc_fc_* are (1, n_new) from MRWD/ARIMA treating gc as a single series
            gc_f = gc_fc_mean[0, :]          # (n_new,)
            gc_f_lower = gc_fc_lower[0, :]   # (n_new,)
            gc_f_upper = gc_fc_upper[0, :]   # (n_new,)

    # ------------------------------------------------------------------ #
    # 4.  Assemble full gc covering the future grid, then compute rates
    # ------------------------------------------------------------------ #
    rates = _compute_forecast_rates(
        fit=fit,
        years_f=years_f,
        kt_f=kt_f,
        cohorts_f=cohorts_f,
        gc_f=gc_f,
    )

    return ForStMoMo(
        fit=fit,
        h=h,
        years_f=years_f,
        cohorts_f=cohorts_f,
        kt_f=kt_f,
        kt_f_lower=kt_f_lower,
        kt_f_upper=kt_f_upper,
        gc_f=gc_f,
        gc_f_lower=gc_f_lower,
        gc_f_upper=gc_f_upper,
        rates=rates,
        level=level,
        kt_model=kt_model,
        gc_model=gc_model,
    )


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _fit_kt_model(kt: np.ndarray, method, arima_order: tuple):
    """Fit (or return) a kt forecaster."""
    if isinstance(method, ExternalKtForecaster):
        return method
    if method == "mrwd":
        from .mrwd import MultivariateRandomWalkDrift
        return MultivariateRandomWalkDrift.fit(kt)
    elif method == "arima":
        from .arima_fc import IndependentArima
        return IndependentArima.fit(kt, order=arima_order, include_constant=True)
    else:
        raise ValueError(
            f"Unknown kt_method: {method!r}. Use 'mrwd', 'arima', or ExternalKtForecaster."
        )


def _fit_gc_model(gc_series: np.ndarray, method, arima_order: tuple, n_new: int):
    """Fit (or return) a gc forecaster."""
    if isinstance(method, ExternalKtForecaster):
        return method
    gc_2d = gc_series.reshape(1, -1)
    if method == "arima":
        from .arima_fc import IndependentArima
        return IndependentArima.fit(gc_2d, order=arima_order, include_constant=True)
    elif method == "mrwd":
        from .mrwd import MultivariateRandomWalkDrift
        return MultivariateRandomWalkDrift.fit(gc_2d)
    else:
        raise ValueError(
            f"Unknown gc_method: {method!r}. Use 'arima', 'mrwd', or ExternalKtForecaster."
        )


def _clean_gc_series(gc: np.ndarray) -> np.ndarray:
    """Remove leading/trailing NaN or Inf from cohort index series.

    Sparse cohorts at the boundary may have degenerate values; trim them
    so that statsmodels ARIMA receives a clean series.
    """
    gc = np.asarray(gc, dtype=float)
    finite_mask = np.isfinite(gc)
    if not finite_mask.any():
        raise ValueError("fit.gc contains no finite values; cannot forecast cohorts.")
    first = int(np.argmax(finite_mask))
    last = int(len(finite_mask) - 1 - np.argmax(finite_mask[::-1]))
    return gc[first : last + 1]


def _compute_forecast_rates(
    fit: FitStMoMo,
    years_f: np.ndarray,
    kt_f: np.ndarray,
    cohorts_f: np.ndarray,
    gc_f: np.ndarray | None,
) -> np.ndarray:
    """Compute forecast mortality rates from projected kt (and gc).

    No offset is used — rates are the model-implied μ or q in the future.

    Parameters
    ----------
    fit:
        Fitted model.
    years_f:
        Future years, shape (h,).
    kt_f:
        Forecast period indexes, shape (N, h).
    cohorts_f:
        New cohorts that were forecast, shape (n_new,).
    gc_f:
        Forecast values for new cohorts, shape (n_new,), or None.

    Returns
    -------
    np.ndarray
        Forecast rates, shape (n_ages, h).
    """
    from ..core.predictor import compute_rates

    ages = fit.ages
    n_ages = len(ages)
    h = len(years_f)

    # Build the eta_xt matrix (n_ages, h) — no offset
    # bx shape: (n_ages, N), kt_f shape: (N, h)
    eta = np.zeros((n_ages, h))

    if fit.ax is not None:
        eta += fit.ax[:, None]

    if fit.bx is not None and fit.bx.size > 0 and kt_f.size > 0:
        eta += fit.bx @ kt_f      # (n_ages, N) @ (N, h)

    if fit.model.has_cohort and fit.b0x is not None:
        # Build gc_full: dict covering both observed and forecast cohorts
        gc_full = _build_gc_dict(fit, cohorts_f, gc_f)
        # Fill in cohort contribution cell-by-cell
        coh_contribution = np.zeros((n_ages, h))
        for j, yr in enumerate(years_f):
            for i, age in enumerate(ages):
                c = int(yr) - int(age)
                if c in gc_full:
                    coh_contribution[i, j] = gc_full[c]
        eta += fit.b0x[:, None] * coh_contribution

    # No offset for future rates
    rates = compute_rates(eta, fit.model.link)
    return rates


def _build_gc_dict(
    fit: FitStMoMo,
    cohorts_f: np.ndarray,
    gc_f: np.ndarray | None,
) -> dict:
    """Return a dict mapping cohort year → gc value.

    Includes both observed and (if present) forecast cohorts.
    """
    gc_dict: dict = {}
    # Observed cohorts
    for c, g in zip(fit.cohorts.tolist(), fit.gc.tolist(), strict=False):
        if np.isfinite(g):
            gc_dict[int(c)] = g
    # Forecast cohorts
    if gc_f is not None and len(cohorts_f) > 0:
        for c, g in zip(cohorts_f.tolist(), gc_f.tolist(), strict=False):
            gc_dict[int(c)] = g
    return gc_dict
