"""Renshaw-Haberman (RH) mortality model."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import NonParametricAgeFun
from ..core.constraints import _rh_constraint
from ..core.stmomo import StMoMo


def rh(link: Literal["log", "logit"] = "log") -> StMoMo:
    """Renshaw-Haberman (RH) mortality model.

    An extension of Lee-Carter with a cohort effect::

        η_xt = α_x + β_x^(1) κ_t + β_x^(0) γ_{t-x}

    where β_x^(0) is a freely-fitted cohort age modulation.

    Identifiability constraints: LC sum constraint + zero-mean cohort effect.

    Parameters
    ----------
    link:
        ``"log"`` (Poisson, default) or ``"logit"`` (Binomial).

    Returns
    -------
    StMoMo

    References
    ----------
    Renshaw, A.E. & Haberman, S. (2006). A Cohort-Based Extension to the
    Lee-Carter Model for Mortality Reduction Factors. *IME*, 38(3), 556–570.

    Examples
    --------
    >>> from pystmomo import rh, load_ew_male
    >>> data = load_ew_male()
    >>> fit = rh().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    """
    return StMoMo(
        link=link,
        static_age_fun=True,
        period_age_fun=(NonParametricAgeFun(),),
        cohort_age_fun=NonParametricAgeFun(),
        const_fun=_rh_constraint,
        text_formula="η_xt = α_x + β_x κ_t + β_x^(0) γ_{t-x}",
    )
