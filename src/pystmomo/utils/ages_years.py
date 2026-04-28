"""Cohort and weight matrix utilities."""
from __future__ import annotations

import numpy as np


def compute_cohorts(ages: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Return the sorted cohort vector covering all (age, year) cells.

    Cohort c = year - age.
    """
    c_min = int(years[0]) - int(ages[-1])
    c_max = int(years[-1]) - int(ages[0])
    return np.arange(c_min, c_max + 1)


def make_weight_matrix(
    Dxt: np.ndarray,
    Ext: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    *,
    min_cohort_obs: int = 3,
) -> np.ndarray:
    """Build a binary weight matrix masking out low-data cells.

    Sets weight = 0 for:
    - Zero exposure cells.
    - NaN death or exposure cells.
    - Cohorts with fewer than *min_cohort_obs* observed cells.

    Parameters
    ----------
    min_cohort_obs:
        Minimum number of observations required for a cohort to be included.

    Returns
    -------
    np.ndarray
        Binary weight matrix, shape (n_ages, n_years), dtype float.
    """
    wxt = np.ones(Dxt.shape, dtype=float)
    wxt[Ext <= 0] = 0.0
    wxt[np.isnan(Dxt) | np.isnan(Ext)] = 0.0

    # Mask cohorts with too few observations
    n_ages, n_years = Dxt.shape
    for j, yr in enumerate(years):
        for i, age in enumerate(ages):
            c = int(yr) - int(age)
            # Count all cells in this cohort
            mask = np.zeros(Dxt.shape, dtype=bool)
            for jj, yr2 in enumerate(years):
                for ii, age2 in enumerate(ages):
                    if int(yr2) - int(age2) == c:
                        mask[ii, jj] = True
            if mask.sum() < min_cohort_obs:
                wxt[i, j] = 0.0

    return wxt


def make_weight_matrix_fast(
    Dxt: np.ndarray,
    Ext: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    *,
    min_cohort_obs: int = 3,
) -> np.ndarray:
    """Vectorised version of :func:`make_weight_matrix` (much faster)."""
    n_ages, n_years = Dxt.shape
    wxt = np.ones((n_ages, n_years), dtype=float)
    wxt[Ext <= 0] = 0.0
    wxt[np.isnan(Dxt) | np.isnan(Ext)] = 0.0

    # Cohort anti-diagonal sums
    cohort_grid = years[None, :] - ages[:, None]  # (n_ages, n_years)
    cohorts = compute_cohorts(ages, years)
    for c in cohorts:
        mask = cohort_grid == c
        if mask.sum() < min_cohort_obs:
            wxt[mask] = 0.0

    return wxt
