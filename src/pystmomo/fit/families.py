"""Likelihood and deviance computations for Poisson and Binomial families."""
from __future__ import annotations

import numpy as np

_EPS = 1e-15


def poisson_loglik(
    obs: np.ndarray,
    fitted: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Poisson log-likelihood (excluding factorial term)."""
    ind = weights > 0
    return float(np.sum(
        weights[ind] * (obs[ind] * np.log(np.maximum(fitted[ind], _EPS)) - fitted[ind])
    ))


def binomial_loglik(
    obs: np.ndarray,
    fitted_q: np.ndarray,
    Ext: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Binomial log-likelihood."""
    ind = weights > 0
    q = np.clip(fitted_q[ind], _EPS, 1 - _EPS)
    n = Ext[ind]
    d = obs[ind]
    return float(np.sum(weights[ind] * (d * np.log(q) + (n - d) * np.log(1.0 - q))))


def poisson_deviance(
    obs: np.ndarray,
    fitted: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Poisson deviance."""
    ind = (weights > 0) & (obs > 0)
    dev = 2.0 * np.sum(
        weights[ind] * (obs[ind] * np.log(obs[ind] / np.maximum(fitted[ind], _EPS))
                        - (obs[ind] - fitted[ind]))
    )
    # Add contribution of zero-death cells
    ind0 = (weights > 0) & (obs <= 0)
    dev += 2.0 * np.sum(weights[ind0] * fitted[ind0])
    return float(dev)


def binomial_deviance(
    obs: np.ndarray,
    fitted_q: np.ndarray,
    Ext: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Binomial deviance."""
    ind = weights > 0
    q = np.clip(fitted_q[ind], _EPS, 1 - _EPS)
    n = Ext[ind]
    d = obs[ind]
    obs_q = np.clip(d / np.maximum(n, _EPS), _EPS, 1 - _EPS)
    dev = 2.0 * np.sum(
        weights[ind] * (d * np.log(obs_q / q) + (n - d) * np.log((1 - obs_q) / (1 - q)))
    )
    return float(dev)
