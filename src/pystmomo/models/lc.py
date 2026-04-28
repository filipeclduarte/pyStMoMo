"""Lee-Carter mortality model (LC)."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import NonParametricAgeFun
from ..core.constraints import _lc_sum_constraint
from ..core.stmomo import StMoMo


def lc(link: Literal["log", "logit"] = "log") -> StMoMo:
    """Lee-Carter (LC) mortality model.

    Linear predictor::

        η_xt = α_x + β_x κ_t

    Identifiability constraint: Σ_x β_x = 1, mean(κ_t) absorbed into α_x.

    Parameters
    ----------
    link:
        ``"log"`` (Poisson, default) or ``"logit"`` (Binomial).

    Returns
    -------
    StMoMo

    References
    ----------
    Lee, R.D. & Carter, L.R. (1992). Modeling and Forecasting U.S. Mortality.
    *JASA*, 87(419), 659–671.

    Examples
    --------
    >>> from pystmomo import lc, load_ew_male
    >>> model = lc()
    >>> data = load_ew_male()
    >>> fit = model.fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    """
    return StMoMo(
        link=link,
        static_age_fun=True,
        period_age_fun=(NonParametricAgeFun(),),
        cohort_age_fun=None,
        const_fun=_lc_sum_constraint,
        text_formula="η_xt = α_x + β_x κ_t",
    )
