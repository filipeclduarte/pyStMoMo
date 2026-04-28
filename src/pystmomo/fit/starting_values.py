"""SVD-based starting value computation for bilinear models (LC, RH)."""
from __future__ import annotations

import numpy as np

from ..core.predictor import logit
from ..utils.linalg import weighted_svd


def svd_starting_values(
    Dxt: np.ndarray,
    Ext: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    wxt: np.ndarray,
    link: str,
    n_components: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD-based starting values for LC/RH models.

    Steps:
    1. Compute raw mortality rates with continuity correction.
    2. Apply link function: log-rates (Poisson) or logit-rates (Binomial).
    3. Estimate α_x as time-average of transformed rates.
    4. Centre to get Z_xt = g(μ_xt) - α_x.
    5. Weighted SVD of Z to get β_x and κ_t.

    Parameters
    ----------
    n_components:
        Number of bilinear components (1 for LC, 1+ for RH).

    Returns
    -------
    ax, bx, kt:
        Initial estimates.  bx has shape (n_ages, n_components),
        kt has shape (n_components, n_years).
    """
    n_ages, n_years = Dxt.shape

    # Raw rate with continuity correction (avoids log(0))
    raw_q = (Dxt + 0.5) / np.maximum(Ext + 1.0, 1e-15)

    if link == "log":
        g_mu = np.log(np.clip(raw_q, 1e-15, None))
    else:
        g_mu = logit(raw_q)

    # Mask non-observed cells
    g_mu_masked = np.where(wxt > 0, g_mu, np.nan)

    # ax = row means (ignoring NaN)
    ax = np.nanmean(g_mu_masked, axis=1)

    # Centre
    Z = g_mu_masked - ax[:, None]
    Z = np.where(wxt > 0, Z, 0.0)

    # Row and col weights for weighted SVD
    row_weights = np.where(wxt.sum(axis=1) > 0, 1.0, 0.0)
    col_weights = np.where(wxt.sum(axis=0) > 0, 1.0, 0.0)

    U, s, Vt = weighted_svd(Z, row_weights=row_weights, col_weights=col_weights)

    bx = U[:, :n_components] * 1.0   # (n_ages, n_components)
    kt = (s[:n_components, None] * Vt[:n_components, :])  # (n_components, n_years)

    # Ensure first β_x is positive (sign convention)
    for k in range(n_components):
        if bx[:, k].sum() < 0:
            bx[:, k] *= -1.0
            kt[k] *= -1.0

    return ax, bx, kt
