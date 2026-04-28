"""CBD model with age-modulated cohort effect (M8)."""
from __future__ import annotations

from typing import Literal

from ..core.age_functions import CenteredCohortAgeFun, ConstantAgeFun, LinearAgeFun
from ..core.constraints import _m8_constraint
from ..core.stmomo import StMoMo


def m8(
    link: Literal["logit", "log"] = "logit",
    xc: float | None = None,
) -> StMoMo:
    """CBD model with age-modulated cohort effect (M8).

    Linear predictor::

        η_xt = κ_t^(1) + (x - x̄) κ_t^(2) + (x_c - x) γ_{t-x}

    where x_c is a fixed reference age (typically ``ages[-1] + 0.5``).

    Identifiability constraint: zero-mean cohort effect.

    Parameters
    ----------
    link:
        ``"logit"`` (Binomial, default) or ``"log"`` (Poisson).
    xc:
        Reference age.  Defaults to ``89.5`` (for ages 55–89 data).

    Returns
    -------
    StMoMo

    References
    ----------
    Cairns et al. (2009) — see :func:`m6`.

    Examples
    --------
    >>> from pystmomo import m8, load_ew_male
    >>> data = load_ew_male()
    >>> fit = m8(xc=float(data.ages[-1]) + 0.5).fit(
    ...     data.deaths, data.exposures, ages=data.ages, years=data.years
    ... )
    """
    xc_val = 89.5 if xc is None else xc
    return StMoMo(
        link=link,
        static_age_fun=False,
        period_age_fun=(ConstantAgeFun(), LinearAgeFun()),
        cohort_age_fun=CenteredCohortAgeFun(xc=xc_val),
        const_fun=_m8_constraint,
        text_formula=f"η_xt = κ_t^(1) + (x - x̄) κ_t^(2) + ({xc_val} - x) γ_{{t-x}}",
    )
