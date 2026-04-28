"""Input validation utilities."""
from __future__ import annotations

import numpy as np


def check_mortality_data(
    Dxt: np.ndarray,
    Ext: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
) -> None:
    """Raise ValueError if mortality data has inconsistent shapes or bad values."""
    if Dxt.ndim != 2:
        raise ValueError(f"Dxt must be 2-D, got shape {Dxt.shape}")
    if Ext.ndim != 2:
        raise ValueError(f"Ext must be 2-D, got shape {Ext.shape}")
    if Dxt.shape != Ext.shape:
        raise ValueError(
            f"Dxt shape {Dxt.shape} does not match Ext shape {Ext.shape}"
        )
    n_ages, n_years = Dxt.shape
    if len(ages) != n_ages:
        raise ValueError(f"len(ages)={len(ages)} does not match Dxt n_ages={n_ages}")
    if len(years) != n_years:
        raise ValueError(f"len(years)={len(years)} does not match Dxt n_years={n_years}")
    if np.any(Ext < 0):
        raise ValueError("Exposures Ext must be non-negative.")
    if np.any(Dxt < 0):
        raise ValueError("Deaths Dxt must be non-negative.")
