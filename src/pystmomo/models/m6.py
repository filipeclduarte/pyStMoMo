"""CBD model with cohort effect (M6)."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import ConstantAgeFun, LinearAgeFun
from ..core.constraints import _m6_constraint
from ..core.stmomo import StMoMo


def m6(link: Literal["logit", "log"] = "logit") -> StMoMo:
    """CBD model extended with a cohort effect (M6).

    Linear predictor::

        η_xt = κ_t^(1) + (x - x̄) κ_t^(2) + γ_{t-x}

    Identifiability constraint: zero-mean cohort effect.

    Parameters
    ----------
    link:
        ``"logit"`` (Binomial, default) or ``"log"`` (Poisson).

    Returns
    -------
    StMoMo

    References
    ----------
    Cairns, A.J.G., Blake, D., Dowd, K., Coughlan, G.D., Epstein, D., Ong, A.
    & Balevich, I. (2009). A Quantitative Comparison of Stochastic Mortality
    Models Using Data From England and Wales and the United States.
    *NAAJ*, 13(1), 1–35.

    Examples
    --------
    >>> from pystmomo import m6, load_ew_male
    >>> data = load_ew_male()
    >>> fit = m6().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    """
    return StMoMo(
        link=link,
        static_age_fun=False,
        period_age_fun=(ConstantAgeFun(), LinearAgeFun()),
        cohort_age_fun=ConstantAgeFun(),
        const_fun=_m6_constraint,
        text_formula="η_xt = κ_t^(1) + (x - x̄) κ_t^(2) + γ_{t-x}",
    )
