"""Age-modulating functions for GAPC mortality models.

Each function maps the age vector to a weight vector used in the linear
predictor:  η_xt = α_x + Σ_i f_i(x) κ_t^(i) + f_0(x) γ_{t-x}.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AgeFunction(Protocol):
    """Protocol satisfied by all age functions."""

    is_parametric: bool

    def __call__(self, ages: np.ndarray) -> np.ndarray:
        """Evaluate age function on *ages* vector.

        Parameters
        ----------
        ages:
            Array of integer ages, shape (n_ages,).

        Returns
        -------
        np.ndarray
            Weight vector, shape (n_ages,).
        """
        ...

    def __repr__(self) -> str: ...


class NonParametricAgeFun:
    """Sentinel that signals β_x must be fitted freely (Lee-Carter, RH).

    When this appears in ``period_age_fun``, the fitter switches to the
    bilinear block-coordinate IRLS path.
    """

    is_parametric: bool = False

    def __call__(self, ages: np.ndarray) -> np.ndarray:  # noqa: D102
        raise RuntimeError(
            "NonParametricAgeFun cannot be evaluated — "
            "it is a sentinel for the bilinear fitter."
        )

    def __repr__(self) -> str:
        return "NonParametricAgeFun()"


class ConstantAgeFun:
    """Age function f(x) = 1 (constant).

    Used in CBD: κ_t^(1) · 1.
    """

    is_parametric: bool = True

    def __call__(self, ages: np.ndarray) -> np.ndarray:
        return np.ones(len(ages))

    def __repr__(self) -> str:
        return "ConstantAgeFun()"


class LinearAgeFun:
    """Age function f(x) = x - mean(ages).

    Used in CBD: κ_t^(2) · (x - x̄).
    """

    is_parametric: bool = True

    def __call__(self, ages: np.ndarray) -> np.ndarray:
        return ages.astype(float) - ages.mean()

    def __repr__(self) -> str:
        return "LinearAgeFun()"


class QuadraticAgeFun:
    """Age function f(x) = (x - x̄)² - σ²_x.

    Centred quadratic used in M7.
    """

    is_parametric: bool = True

    def __call__(self, ages: np.ndarray) -> np.ndarray:
        dev = ages.astype(float) - ages.mean()
        return dev ** 2 - np.mean(dev ** 2)

    def __repr__(self) -> str:
        return "QuadraticAgeFun()"


class CenteredCohortAgeFun:
    """Age function f(x) = x_c - x for a fixed reference age x_c.

    Used in M8: γ_{t-x} · (x_c - x).

    Parameters
    ----------
    xc:
        Reference age, typically the highest age in the data plus 0.5.
    """

    is_parametric: bool = True

    def __init__(self, xc: float = 89.5) -> None:
        self.xc = xc

    def __call__(self, ages: np.ndarray) -> np.ndarray:
        return self.xc - ages.astype(float)

    def __repr__(self) -> str:
        return f"CenteredCohortAgeFun(xc={self.xc})"


class CallableAgeFun:
    """Wrap any callable f(ages) -> weights as a parametric age function.

    Parameters
    ----------
    fn:
        Callable that accepts a 1-D numpy array of ages and returns a 1-D
        array of weights of the same length.
    name:
        Optional name for display.
    """

    is_parametric: bool = True

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "custom") -> None:
        self._fn = fn
        self._name = name

    def __call__(self, ages: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(ages), dtype=float)

    def __repr__(self) -> str:
        return f"CallableAgeFun(name={self._name!r})"
