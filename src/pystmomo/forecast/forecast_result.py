"""ForStMoMo: result of forecasting a fitted StMoMo model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo


@dataclass
class ForStMoMo:
    """Result of :func:`~pystmomo.forecast.forecast`.

    Attributes
    ----------
    fit:
        The fitted model that was projected.
    h:
        Forecast horizon (number of future years).
    years_f:
        Future year labels, shape (h,).
    cohorts_f:
        All future cohorts that were forecast (those beyond max(fit.cohorts)),
        shape (n_new_cohorts,).  Empty array when the model has no cohort term.
    kt_f:
        Central forecast of period indexes, shape (N, h).
    kt_f_lower:
        Lower confidence band for kt, shape (N, h).
    kt_f_upper:
        Upper confidence band for kt, shape (N, h).
    gc_f:
        Central forecast of new cohort indexes, shape (n_new_cohorts,).
        None when the model has no cohort term.
    gc_f_lower:
        Lower band for gc, shape (n_new_cohorts,).  None if no cohort.
    gc_f_upper:
        Upper band for gc, shape (n_new_cohorts,).  None if no cohort.
    rates:
        Central forecast mortality rates, shape (n_ages, h).
    level:
        Confidence level used for intervals.
    kt_model:
        The fitted MRWD or IndependentArima used to project kt.
    gc_model:
        The fitted model used to project gc, or None.
    """

    fit: FitStMoMo
    h: int
    years_f: np.ndarray
    cohorts_f: np.ndarray
    kt_f: np.ndarray
    kt_f_lower: np.ndarray
    kt_f_upper: np.ndarray
    gc_f: np.ndarray | None
    gc_f_lower: np.ndarray | None
    gc_f_upper: np.ndarray | None
    rates: np.ndarray
    level: float
    kt_model: object
    gc_model: object | None = field(default=None)

    def __repr__(self) -> str:
        n_ages = self.rates.shape[0]
        return (
            f"ForStMoMo(\n"
            f"  model  = {self.fit.model}\n"
            f"  years  = {self.years_f[0]}–{self.years_f[-1]}  (h={self.h})\n"
            f"  n_ages = {n_ages}\n"
            f"  rates  shape = {self.rates.shape}\n"
            f"  level  = {self.level:.0%}\n"
            f")"
        )
