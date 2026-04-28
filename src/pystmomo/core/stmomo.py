"""Core StMoMo model specification class."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from .age_functions import AgeFunction, NonParametricAgeFun
from ..utils.ages_years import compute_cohorts, make_weight_matrix_fast
from ..utils.validation import check_mortality_data

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo


class StMoMo:
    """Generalised Age-Period-Cohort (GAPC) stochastic mortality model.

    The linear predictor is::

        η_xt = α_x + Σ_i β_x^(i) κ_t^(i) + β_x^(0) γ_{t-x}

    with death counts modelled as:

    * Poisson (link = ``"log"``): D_xt ~ Poisson(E_xt · μ_xt)
    * Binomial (link = ``"logit"``): D_xt ~ Binomial(E_xt, q_xt)

    Parameters
    ----------
    link:
        Link function: ``"log"`` or ``"logit"``.
    static_age_fun:
        Whether to include a static age term α_x.
    period_age_fun:
        Tuple of age-modulating functions for period terms β_x^(i).
        Use :class:`~pystmomo.core.age_functions.NonParametricAgeFun` to
        request a freely-fitted (non-parametric) β_x.
    cohort_age_fun:
        Age-modulating function for the cohort term β_x^(0), or ``None``
        for no cohort effect.
    const_fun:
        Post-fit identifiability constraint function.
        Signature: ``(ax, bx, kt, b0x, gc, ages, years, cohorts) -> (ax, bx, kt, b0x, gc)``.
    text_formula:
        Human-readable formula string for display.

    Examples
    --------
    Define the Lee-Carter model:

    >>> from pystmomo import lc
    >>> model = lc()
    >>> print(model)
    StMoMo(link='log', formula='η_xt = α_x + β_x κ_t')
    """

    def __init__(
        self,
        link: Literal["log", "logit"],
        static_age_fun: bool,
        period_age_fun: tuple[AgeFunction, ...] = (),
        cohort_age_fun: AgeFunction | None = None,
        const_fun: Callable | None = None,
        text_formula: str = "",
    ) -> None:
        self.link = link
        self.static_age_fun = static_age_fun
        self.period_age_fun = tuple(period_age_fun)
        self.cohort_age_fun = cohort_age_fun
        self.const_fun = const_fun
        self.text_formula = text_formula

    @property
    def N(self) -> int:
        """Number of period terms (bilinear components)."""
        return len(self.period_age_fun)

    @property
    def has_cohort(self) -> bool:
        """Whether the model includes a cohort effect."""
        return self.cohort_age_fun is not None

    @property
    def is_fully_parametric(self) -> bool:
        """True if all age functions are parametric (enabling GLM path)."""
        return all(
            not isinstance(af, NonParametricAgeFun) for af in self.period_age_fun
        ) and (
            self.cohort_age_fun is None
            or not isinstance(self.cohort_age_fun, NonParametricAgeFun)
        )

    def fit(
        self,
        Dxt: np.ndarray,
        Ext: np.ndarray,
        ages: np.ndarray,
        years: np.ndarray,
        *,
        wxt: np.ndarray | None = None,
        oxt: np.ndarray | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> "FitStMoMo":
        """Fit the model to mortality data.

        Parameters
        ----------
        Dxt:
            Deaths matrix, shape (n_ages, n_years).
        Ext:
            Exposures matrix (central or initial), shape (n_ages, n_years).
        ages:
            Age labels, length n_ages.
        years:
            Calendar year labels, length n_years.
        wxt:
            Optional binary weight matrix.  Cells with weight 0 are excluded.
            Defaults to the automatic weight matrix that masks zero-exposure
            cells and sparse cohorts.
        oxt:
            Log-exposure offset (log E_xt for Poisson; log E_xt for Binomial
            initial exposures).  If ``None``, computed automatically from Ext.
        max_iter:
            Maximum IRLS iterations (bilinear path only).
        tol:
            Convergence tolerance on relative deviance change (bilinear path).
        verbose:
            Print iteration diagnostics (bilinear path only).

        Returns
        -------
        FitStMoMo
            Fitted model result.
        """
        ages = np.asarray(ages, dtype=int)
        years = np.asarray(years, dtype=int)
        Dxt = np.asarray(Dxt, dtype=float)
        Ext = np.asarray(Ext, dtype=float)

        check_mortality_data(Dxt, Ext, ages, years)
        cohorts = compute_cohorts(ages, years)

        if wxt is None:
            wxt = make_weight_matrix_fast(Dxt, Ext, ages, years)
        else:
            wxt = np.asarray(wxt, dtype=float)

        if oxt is None:
            oxt = np.log(np.where(Ext > 0, Ext, 1.0))

        if self.is_fully_parametric:
            from ..fit.glm_fit import fit_parametric
            return fit_parametric(
                self, Dxt, Ext, ages, years, cohorts, wxt, oxt
            )
        else:
            from ..fit.bilinear_fit import fit_bilinear
            return fit_bilinear(
                self, Dxt, Ext, ages, years, cohorts, wxt, oxt,
                max_iter=max_iter, tol=tol, verbose=verbose,
            )

    def __repr__(self) -> str:
        return (
            f"StMoMo(link={self.link!r}, "
            f"N={self.N}, "
            f"cohort={self.has_cohort}, "
            f"formula={self.text_formula!r})"
        )

    def __str__(self) -> str:
        return f"StMoMo(link={self.link!r}, formula={self.text_formula!r})"
