"""Parameter plots for fitted StMoMo models."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..fit.fit_result import FitStMoMo


def plot_parameters(fit: "FitStMoMo", *, fig: Figure | None = None) -> Figure:
    """Multi-panel plot of fitted model parameters.

    Panels shown (only for non-None arrays):
    - α_x (ax) vs ages
    - β_x^(i) vs ages for each period term i
    - κ_t^(i) vs years for each period term i
    - β_x^(0) (b0x) vs ages (if present)
    - γ_c (gc) vs cohorts (if present)

    Parameters
    ----------
    fit:
        A fitted FitStMoMo object.
    fig:
        Optional existing Figure to plot into.  If None, a new Figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Each panel is (title, x-axis label, xvals, yvals)
    panels: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    if fit.ax is not None:
        panels.append(("$\\alpha_x$", "Age", fit.ages, fit.ax))

    if fit.bx is not None:
        bx = np.atleast_2d(fit.bx)
        # bx shape: (n_ages, N) where N = number of period terms
        if bx.ndim == 1:
            bx = bx[:, np.newaxis]
        for i in range(bx.shape[1]):
            label = f"$\\beta_x^{{({i+1})}}$" if bx.shape[1] > 1 else "$\\beta_x$"
            panels.append((label, "Age", fit.ages, bx[:, i]))

    if fit.kt is not None:
        kt = np.atleast_2d(fit.kt)
        # kt shape: (N, n_years)
        if kt.ndim == 1:
            kt = kt[np.newaxis, :]
        for i in range(kt.shape[0]):
            label = f"$\\kappa_t^{{({i+1})}}$" if kt.shape[0] > 1 else "$\\kappa_t$"
            panels.append((label, "Year", fit.years, kt[i, :]))

    if fit.b0x is not None:
        panels.append(("$\\beta_x^{{(0)}}$", "Age", fit.ages, fit.b0x))

    if fit.gc is not None:
        panels.append(("$\\gamma_c$", "Cohort", fit.cohorts, fit.gc))

    n_panels = len(panels)
    if n_panels == 0:
        if fig is None:
            fig = Figure(figsize=(6, 4))
        return fig

    # Layout: up to 3 columns
    ncols = min(n_panels, 3)
    nrows = (n_panels + ncols - 1) // ncols

    if fig is None:
        fig = Figure(figsize=(5 * ncols, 4 * nrows))

    axes = fig.subplots(nrows, ncols, squeeze=False)

    for idx, (label, xlabel, xvals, yvals) in enumerate(panels):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        ax.plot(xvals, yvals, color="steelblue", linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig
