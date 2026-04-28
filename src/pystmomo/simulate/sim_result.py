"""SimStMoMo result class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo


@dataclass
class SimStMoMo:
    """Result of simulating future mortality trajectories.

    Attributes
    ----------
    fit:
        The fitted model used to generate simulations.
    h:
        Forecast horizon (number of future years).
    nsim:
        Number of simulated sample paths.
    rates:
        Simulated mortality rates, shape (n_ages, h, nsim).
        For Poisson log link: central death rates μ_xt.
        For Binomial logit link: initial death probabilities q_xt.
    years_f:
        Future calendar years, shape (h,).
    ages:
        Age vector from the fitted model.
    kt_s:
        Simulated period indexes, shape (N, h, nsim).
    gc_s:
        Simulated cohort effects for new cohorts, shape (n_new_cohorts, nsim),
        or ``None`` if the model has no cohort effect.
    seed:
        Random seed used (for reproducibility).
    """

    fit: FitStMoMo
    h: int
    nsim: int
    rates: np.ndarray         # (n_ages, h, nsim)
    years_f: np.ndarray       # (h,)
    ages: np.ndarray          # (n_ages,)
    kt_s: np.ndarray          # (N, h, nsim)
    gc_s: np.ndarray | None   # (n_new_cohorts, nsim) or None
    seed: int | None

    def quantile(self, q: float) -> np.ndarray:
        """Quantile of simulated rates across paths.

        Parameters
        ----------
        q:
            Quantile in [0, 1].

        Returns
        -------
        np.ndarray
            Shape (n_ages, h).
        """
        return np.quantile(self.rates, q, axis=2)

    def mean(self) -> np.ndarray:
        """Mean simulated rate, shape (n_ages, h)."""
        return self.rates.mean(axis=2)

    def __repr__(self) -> str:
        return (
            f"SimStMoMo(h={self.h}, nsim={self.nsim}, "
            f"ages={self.ages[0]}–{self.ages[-1]}, "
            f"years_f={self.years_f[0]}–{self.years_f[-1]})"
        )
