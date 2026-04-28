"""CBD model with quadratic age and cohort effect (M7)."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import ConstantAgeFun, LinearAgeFun, QuadraticAgeFun
from ..core.constraints import _m7_constraint
from ..core.stmomo import StMoMo


def m7(link: Literal["logit", "log"] = "logit") -> StMoMo:
    """CBD model with quadratic age term and cohort effect (M7).

    Linear predictor::

        η_xt = κ_t^(1) + (x - x̄) κ_t^(2) + ((x - x̄)² - σ²_x) κ_t^(3) + γ_{t-x}

    Three identifiability constraints are applied to the cohort effect:
    zero mean, zero linear trend, and zero quadratic trend.

    Parameters
    ----------
    link:
        ``"logit"`` (Binomial, default) or ``"log"`` (Poisson).

    Returns
    -------
    StMoMo

    References
    ----------
    Cairns et al. (2009) — see :func:`m6`.

    Examples
    --------
    >>> from pystmomo import m7, load_ew_male
    >>> data = load_ew_male()
    >>> fit = m7().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    """
    return StMoMo(
        link=link,
        static_age_fun=False,
        period_age_fun=(ConstantAgeFun(), LinearAgeFun(), QuadraticAgeFun()),
        cohort_age_fun=ConstantAgeFun(),
        const_fun=_m7_constraint,
        text_formula=(
            "η_xt = κ_t^(1) + (x - x̄) κ_t^(2) + "
            "((x - x̄)² - σ²_x) κ_t^(3) + γ_{t-x}"
        ),
    )
