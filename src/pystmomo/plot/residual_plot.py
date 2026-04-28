"""Residual plots for fitted StMoMo models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo

_VALID_KINDS = ("deviance", "pearson", "response")


def _get_residuals(fit: "FitStMoMo", kind: str) -> np.ndarray:
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")
    return fit.residuals(kind=kind)


def plot_residual_heatmap(fit: "FitStMoMo", kind: str = "deviance") -> Figure:
    """Heatmap of residuals on the ages × years grid.

    Uses matplotlib ``imshow`` with the RdBu_r colormap, centred at zero.

    Parameters
    ----------
    fit:
        A fitted FitStMoMo object.
    kind:
        Type of residuals: ``"deviance"`` (default), ``"pearson"``, or
        ``"response"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    res = _get_residuals(fit, kind)

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    vmax = np.nanmax(np.abs(res))
    vmax = vmax if vmax > 0 else 1.0

    im = ax.imshow(
        res,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
        extent=[fit.years[0], fit.years[-1], fit.ages[-1], fit.ages[0]],
    )
    fig.colorbar(im, ax=ax, label=f"{kind.capitalize()} residual")
    ax.set_xlabel("Year")
    ax.set_ylabel("Age")
    ax.set_title(f"{kind.capitalize()} residuals — ages × years")

    fig.tight_layout()
    return fig


def plot_residual_scatter(fit: "FitStMoMo", kind: str = "deviance") -> Figure:
    """Residuals vs fitted log-rates scatter plot.

    Parameters
    ----------
    fit:
        A fitted FitStMoMo object.
    kind:
        Type of residuals: ``"deviance"`` (default), ``"pearson"``, or
        ``"response"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    res = _get_residuals(fit, kind)
    rates = fit.fitted_rates

    # Use only active cells
    mask = (fit.wxt > 0) & (rates > 0)
    log_rates = np.log(rates[mask])
    residuals = res[mask]

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.scatter(log_rates, residuals, s=12, alpha=0.4)
    ax.axhline(0.0, color="#dc2626", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Fitted log-rate")
    ax.set_ylabel(f"{kind.capitalize()} residual")
    ax.set_title(f"{kind.capitalize()} residuals vs fitted log-rates")

    fig.tight_layout()
    return fig
