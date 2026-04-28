"""Linear predictor computation for GAPC models."""
from __future__ import annotations

import numpy as np


def compute_eta(
    ax: np.ndarray | None,
    bx: np.ndarray,
    kt: np.ndarray,
    b0x: np.ndarray | None,
    gc: np.ndarray | None,
    ages: np.ndarray,
    years: np.ndarray,
    cohorts: np.ndarray,
    oxt: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the linear predictor matrix η_xt.

    η_xt = α_x + Σ_i β_x^(i) κ_t^(i) + β_x^(0) γ_{t-x} + oxt

    Parameters
    ----------
    ax:
        Static age function coefficients, shape (n_ages,), or None.
    bx:
        Period age-modulating functions, shape (n_ages, N).
    kt:
        Period indexes, shape (N, n_years).
    b0x:
        Cohort age-modulating function, shape (n_ages,), or None.
    gc:
        Cohort index values, shape (n_cohorts,), or None.
    ages:
        Age vector, shape (n_ages,).
    years:
        Year vector, shape (n_years,).
    cohorts:
        Cohort vector (years[0]-ages[-1] .. years[-1]-ages[0]).
    oxt:
        Log-exposure offset matrix, shape (n_ages, n_years), or None.

    Returns
    -------
    np.ndarray
        Linear predictor matrix, shape (n_ages, n_years).
    """
    n_ages = len(ages)
    n_years = len(years)
    eta = np.zeros((n_ages, n_years))

    if ax is not None:
        eta += ax[:, None]

    if bx.size > 0 and kt.size > 0:
        # eta += bx @ kt  — (n_ages, N) @ (N, n_years)
        eta += bx @ kt

    if b0x is not None and gc is not None and len(gc) > 0:
        cohort_mat = _cohort_index_matrix(ages, years, cohorts, gc)
        eta += b0x[:, None] * cohort_mat

    if oxt is not None:
        eta += oxt

    return eta


def _cohort_index_matrix(
    ages: np.ndarray,
    years: np.ndarray,
    cohorts: np.ndarray,
    gc: np.ndarray,
) -> np.ndarray:
    """Build the cohort-index matrix C where C[i, j] = γ_{years[j] - ages[i]}.

    Parameters
    ----------
    ages, years, cohorts:
        Age, year, and cohort vectors.
    gc:
        Cohort effect values, one per cohort in *cohorts*.

    Returns
    -------
    np.ndarray
        Shape (n_ages, n_years).  Cells where cohort is outside *cohorts* are
        set to 0.0 (will be masked by wxt).
    """
    # Vectorised: build a lookup array indexed by cohort value
    c_min = int(cohorts[0])
    c_max = int(cohorts[-1])
    n_lookup = c_max - c_min + 2  # +1 for range, +1 for sentinel
    # Pad with a zero sentinel at the end for out-of-range cohorts
    gc_lookup = np.zeros(n_lookup)
    gc_lookup[:len(gc)] = gc

    cohort_grid = years[None, :] - ages[:, None]  # (n_ages, n_years)
    idx = (cohort_grid - c_min).astype(int)
    # Map out-of-range indices to the sentinel (last element = 0.0)
    sentinel = n_lookup - 1
    idx = np.where((idx >= 0) & (idx < len(gc)), idx, sentinel)
    return gc_lookup[idx]


def compute_rates(
    eta: np.ndarray,
    link: str,
    oxt: np.ndarray | None = None,
) -> np.ndarray:
    """Convert linear predictor to mortality rates.

    Parameters
    ----------
    eta:
        Linear predictor (already including oxt if used in eta assembly).
    link:
        ``"log"`` → μ_xt = exp(η),  ``"logit"`` → q_xt = 1/(1+exp(-η)).
    oxt:
        If the offset has *not* been included in *eta*, pass it here and it
        will be subtracted before computing rates (so rates are age-period
        specific, not total).

    Returns
    -------
    np.ndarray
        Mortality rates (μ or q), same shape as *eta*.
    """
    if oxt is not None:
        eta = eta - oxt
    if link == "log":
        return np.exp(np.clip(eta, -30.0, 30.0))
    elif link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    else:
        raise ValueError(f"Unknown link: {link!r}. Use 'log' or 'logit'.")


def logit(p: np.ndarray) -> np.ndarray:
    """Element-wise logit transform with numerical clipping."""
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return np.log(p / (1.0 - p))


def invlogit(x: np.ndarray) -> np.ndarray:
    """Element-wise inverse logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
