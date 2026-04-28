"""Cairns-Blake-Dowd (CBD) mortality model."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import ConstantAgeFun, LinearAgeFun
from ..core.stmomo import StMoMo


def cbd(link: Literal["logit", "log"] = "logit") -> StMoMo:
    """Cairns-Blake-Dowd (CBD) mortality model.

    Linear predictor::

        η_xt = κ_t^(1) + (x - x̄) κ_t^(2)

    No identifiability constraints are needed as the two period indexes are
    separately identified by the intercept and slope over age.

    Parameters
    ----------
    link:
        ``"logit"`` (Binomial, default) or ``"log"`` (Poisson).

    Returns
    -------
    StMoMo

    References
    ----------
    Cairns, A.J.G., Blake, D. & Dowd, K. (2006). A Two-Factor Model for
    Stochastic Mortality with Parameter Uncertainty. *Journal of Risk and
    Insurance*, 73(4), 687–718.

    Examples
    --------
    >>> from pystmomo import cbd, load_ew_male
    >>> data = load_ew_male()
    >>> fit = cbd().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    """
    return StMoMo(
        link=link,
        static_age_fun=False,
        period_age_fun=(ConstantAgeFun(), LinearAgeFun()),
        cohort_age_fun=None,
        const_fun=None,
        text_formula="η_xt = κ_t^(1) + (x - x̄) κ_t^(2)",
    )
