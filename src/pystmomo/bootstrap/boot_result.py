"""BootStMoMo result class."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo


@dataclass
class BootStMoMo:
    """Result of bootstrap uncertainty quantification.

    Attributes
    ----------
    base_fit:
        The original fitted model.
    nboot:
        Number of bootstrap replicates.
    method:
        Bootstrap method used: ``"semiparametric"`` or ``"residual"``.
    fits:
        List of refitted :class:`FitStMoMo` objects (one per replicate).
    ax_b:
        Bootstrap distribution of α_x, shape (nboot, n_ages) or None.
    bx_b:
        Bootstrap distribution of β_x, shape (nboot, n_ages, N).
    kt_b:
        Bootstrap distribution of κ_t, shape (nboot, N, n_years).
    b0x_b:
        Bootstrap distribution of β_x^(0), shape (nboot, n_ages) or None.
    gc_b:
        Bootstrap distribution of γ_c, shape (nboot, n_cohorts) or None.
    """

    base_fit: FitStMoMo
    nboot: int
    method: Literal["semiparametric", "residual"]
    fits: list[FitStMoMo] = field(default_factory=list)

    @property
    def ax_b(self) -> np.ndarray | None:
        """Bootstrap ax, shape (nboot, n_ages)."""
        if not self.fits or self.fits[0].ax is None:
            return None
        return np.stack([f.ax for f in self.fits])

    @property
    def bx_b(self) -> np.ndarray:
        """Bootstrap bx, shape (nboot, n_ages, N)."""
        return np.stack([f.bx for f in self.fits])

    @property
    def kt_b(self) -> np.ndarray:
        """Bootstrap kt, shape (nboot, N, n_years)."""
        return np.stack([f.kt for f in self.fits])

    @property
    def b0x_b(self) -> np.ndarray | None:
        """Bootstrap b0x, shape (nboot, n_ages) or None."""
        if not self.fits or self.fits[0].b0x is None:
            return None
        return np.stack([f.b0x for f in self.fits])

    @property
    def gc_b(self) -> np.ndarray | None:
        """Bootstrap gc, shape (nboot, n_cohorts) or None."""
        if not self.fits or self.fits[0].gc is None:
            return None
        return np.stack([f.gc for f in self.fits])

    def parameter_ci(
        self,
        param: str = "kt",
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap confidence interval for a parameter.

        Parameters
        ----------
        param:
            One of ``"ax"``, ``"bx"``, ``"kt"``, ``"b0x"``, ``"gc"``.
        level:
            Confidence level in (0, 1).

        Returns
        -------
        lower, upper:
            Arrays of the same shape as the parameter, at the requested
            confidence level.
        """
        alpha = (1.0 - level) / 2.0
        arr = getattr(self, f"{param}_b")
        if arr is None:
            raise ValueError(f"Parameter '{param}' is not available in this bootstrap.")
        lower = np.quantile(arr, alpha, axis=0)
        upper = np.quantile(arr, 1 - alpha, axis=0)
        return lower, upper

    def __repr__(self) -> str:
        return (
            f"BootStMoMo(method={self.method!r}, nboot={self.nboot}, "
            f"n_fits_ok={len(self.fits)})"
        )
