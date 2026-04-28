"""Multivariate Random Walk with Drift (MRWD) for period indexes κ_t."""
from __future__ import annotations

import numpy as np
from scipy import stats


class MultivariateRandomWalkDrift:
    """Multivariate Random Walk with Drift fitted to a matrix of period indexes.

    The model is:
        Δκ_t = drift + ε_t,   ε_t ~ MVN(0, Σ)

    Forecast mean at step s:  last + s * drift
    Forecast variance at step s:
        Var(s) = s * Σ  +  (s²/T) * Σ
    where T = number of differences (n_years - 1).

    The second term accounts for estimation uncertainty in the drift.

    Attributes
    ----------
    drift:
        Shape (N,) — mean of first differences.
    sigma:
        Shape (N, N) — sample covariance of first differences.
    last:
        Shape (N,) — last observed κ_t values.
    n_years:
        Number of fitted years (T+1); T = n_years - 1 differences are used.
    """

    def __init__(
        self,
        drift: np.ndarray,
        sigma: np.ndarray,
        last: np.ndarray,
        n_years: int,
    ) -> None:
        self.drift = drift
        self.sigma = sigma
        self.last = last
        self.n_years = n_years

    @classmethod
    def fit(cls, kt: np.ndarray) -> "MultivariateRandomWalkDrift":
        """Fit MRWD to observed period indexes.

        Parameters
        ----------
        kt:
            Period indexes, shape (N, n_years).

        Returns
        -------
        MultivariateRandomWalkDrift
        """
        kt = np.atleast_2d(kt)
        # First differences along years axis: shape (N, n_years-1)
        diff = np.diff(kt, axis=1)
        drift = diff.mean(axis=1)                  # (N,)
        # Covariance of differences: rowvar=True means each row is a variable
        if diff.shape[0] == 1:
            # Single period index: 1x1 covariance
            sigma = np.var(diff, ddof=1, keepdims=False).reshape(1, 1)
        else:
            sigma = np.cov(diff, rowvar=True)      # (N, N)
        last = kt[:, -1]                           # (N,)
        return cls(drift=drift, sigma=sigma, last=last, n_years=kt.shape[1])

    def forecast(
        self,
        h: int,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point forecast and confidence interval for h steps ahead.

        Parameters
        ----------
        h:
            Forecast horizon.
        level:
            Confidence level (e.g. 0.95).

        Returns
        -------
        mean, lower, upper
            Each shape (N, h).
        """
        N = len(self.drift)
        T = self.n_years - 1          # number of differences used
        z = stats.norm.ppf(0.5 + level / 2.0)

        mean = np.zeros((N, h))
        lower = np.zeros((N, h))
        upper = np.zeros((N, h))

        for s in range(1, h + 1):
            # Forecast mean
            mean[:, s - 1] = self.last + s * self.drift

            # Forecast covariance: s*Σ + (s²/T)*Σ  = (s + s²/T) * Σ
            factor = s + (s ** 2) / T
            var_diag = factor * np.diag(self.sigma)     # (N,)
            sd = np.sqrt(np.maximum(var_diag, 0.0))

            lower[:, s - 1] = mean[:, s - 1] - z * sd
            upper[:, s - 1] = mean[:, s - 1] + z * sd

        return mean, lower, upper

    def simulate(
        self,
        h: int,
        nsim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate future trajectories using Cholesky-based MVN innovations.

        Parameters
        ----------
        h:
            Forecast horizon.
        nsim:
            Number of simulations.
        rng:
            NumPy random generator.

        Returns
        -------
        np.ndarray
            Shape (N, h, nsim).
        """
        N = len(self.drift)
        # Cholesky decomposition of sigma
        try:
            L = np.linalg.cholesky(self.sigma)
        except np.linalg.LinAlgError:
            # Add small jitter for numerical stability
            L = np.linalg.cholesky(self.sigma + 1e-10 * np.eye(N))

        paths = np.zeros((N, h, nsim))
        kt_prev = np.tile(self.last[:, None], (1, nsim))  # (N, nsim)

        for s in range(h):
            # Standard normals: (N, nsim)
            z = rng.standard_normal((N, nsim))
            # Correlated MVN innovation: L @ z, shape (N, nsim)
            innov = L @ z
            kt_prev = kt_prev + self.drift[:, None] + innov
            paths[:, s, :] = kt_prev

        return paths
