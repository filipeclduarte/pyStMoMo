"""Age-Period-Cohort (APC) mortality model."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import ConstantAgeFun
from ..core.constraints import _apc_constraint
from ..core.stmomo import StMoMo


def apc(link: Literal["log", "logit"] = "log") -> StMoMo:
    """Age-Period-Cohort (APC) mortality model.

    Linear predictor::

        η_xt = α_x + κ_t + γ_{t-x}

    Identifiability constraints (following Villegas et al. 2018):

    * Zero-mean period index: mean(κ_t) = 0, absorbed into α_x.
    * Zero-mean cohort index: mean(γ_c) = 0.
    * Zero linear trend in cohort index: no linear trend in γ_c.

    Parameters
    ----------
    link:
        ``"log"`` (Poisson, default) or ``"logit"`` (Binomial).

    Returns
    -------
    StMoMo

    References
    ----------
    Currie, I.D. (2006). Smoothing and Forecasting Mortality Rates with P-splines.
    Talk at the Institute of Actuaries.

    Examples
    --------
    >>> from pystmomo import apc, load_ew_male
    >>> data = load_ew_male()
    >>> fit = apc().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
    """
    return StMoMo(
        link=link,
        static_age_fun=True,
        period_age_fun=(ConstantAgeFun(),),
        cohort_age_fun=ConstantAgeFun(),
        const_fun=_apc_constraint,
        text_formula="η_xt = α_x + κ_t + γ_{t-x}",
    )
