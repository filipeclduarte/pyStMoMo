"""Period cross-validation for fitted StMoMo models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo


def cv_stmomo(
    fit: FitStMoMo,
    n_folds: int = 5,
    metric: Literal["mse", "log_mse"] = "mse",
) -> dict:
    """Period-based leave-last-out cross-validation.

    Holds out the last ``n_test_years = n_years // n_folds`` consecutive years,
    refits the model on the remaining years, forecasts ``h = n_test_years`` steps,
    and compares predicted rates to observed rates.

    Parameters
    ----------
    fit:
        A fitted FitStMoMo object.
    n_folds:
        Number of folds.  Determines the test window size as
        ``n_test_years = n_years // n_folds``.
    metric:
        Primary metric to return in 'metric_value': ``"mse"`` (mean squared
        error on rates) or ``"log_mse"`` (MSE on log rates).

    Returns
    -------
    dict with keys:
        * ``'mse'``: MSE between observed and predicted rates on the test set.
        * ``'log_mse'``: MSE on log scale.
        * ``'years_test'``: array of held-out year labels.
        * ``'rates_obs'``: observed rates, shape (n_ages, n_test_years).
        * ``'rates_pred'``: predicted rates, shape (n_ages, n_test_years).
        * ``'metric_value'``: value of the requested metric.
    """
    n_years = len(fit.years)
    n_test = n_years // n_folds
    if n_test < 1:
        raise ValueError(
            f"n_folds={n_folds} too large for n_years={n_years}: "
            f"n_test_years = n_years // n_folds = {n_test} < 1"
        )

    n_train = n_years - n_test
    years_train = fit.years[:n_train]
    years_test = fit.years[n_train:]

    Dxt_train = fit.Dxt[:, :n_train]
    Ext_train = fit.Ext[:, :n_train]
    wxt_train = fit.wxt[:, :n_train]

    # Refit on training data
    fit_train = fit.model.fit(
        Dxt_train,
        Ext_train,
        ages=fit.ages,
        years=years_train,
        wxt=wxt_train,
    )

    # Forecast h steps ahead
    fc = fit_train.forecast(h=n_test)

    # Predicted rates: central forecast, shape (n_ages, n_test)
    # ForStMoMo stores rates_central (or similar); use duck access
    if hasattr(fc, "rates_central"):
        rates_pred = np.asarray(fc.rates_central)
    elif hasattr(fc, "rates"):
        rates_pred = np.asarray(fc.rates)
    else:
        raise AttributeError(
            "ForStMoMo object has neither 'rates_central' nor 'rates' attribute."
        )

    # Observed crude rates from held-out years
    Ext_test = fit.Ext[:, n_train:]
    Dxt_test = fit.Dxt[:, n_train:]
    _EPS = 1e-15
    rates_obs = np.where(Ext_test > 0, Dxt_test / np.maximum(Ext_test, _EPS), 0.0)

    # Compute metrics (only on cells where both are positive)
    valid = (rates_obs > 0) & (rates_pred > 0)
    diff = rates_obs - rates_pred
    mse = float(np.mean(diff[valid] ** 2)) if valid.any() else float("nan")
    log_mse = float(
        np.mean((np.log(rates_obs[valid]) - np.log(rates_pred[valid])) ** 2)
    ) if valid.any() else float("nan")

    metric_value = mse if metric == "mse" else log_mse

    return {
        "mse": mse,
        "log_mse": log_mse,
        "years_test": years_test,
        "rates_obs": rates_obs,
        "rates_pred": rates_pred,
        "metric_value": metric_value,
    }
