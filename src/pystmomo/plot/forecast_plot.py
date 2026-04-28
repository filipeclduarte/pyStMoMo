"""Forecast and fan-chart plots for StMoMo forecast/simulation objects."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    pass  # ForStMoMo / SimStMoMo imported lazily inside functions


def plot_forecast(
    fc,
    *,
    ages: list[int] | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Plot κ_t forecasts with confidence bands.

    Expects ``fc`` to expose:
    - ``fc.kt_central``: shape (N, h) central forecast of period indexes
    - ``fc.kt_lower``: shape (N, h) lower confidence bound
    - ``fc.kt_upper``: shape (N, h) upper confidence bound
    - ``fc.years``: forecast year labels, length h
    - ``fc.fit``: original FitStMoMo (for historical kt and years)

    Falls back gracefully if confidence bands are absent.

    Parameters
    ----------
    fc:
        A ForStMoMo forecast object.
    ages:
        Ignored (present for API symmetry with rate-based plots).
    fig:
        Optional existing Figure; if None a new one is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    kt_central = np.atleast_2d(np.asarray(fc.kt_central))  # (N, h)
    N = kt_central.shape[0]

    has_bands = hasattr(fc, "kt_lower") and fc.kt_lower is not None
    if has_bands:
        kt_lower = np.atleast_2d(np.asarray(fc.kt_lower))
        kt_upper = np.atleast_2d(np.asarray(fc.kt_upper))

    years_fc = np.asarray(fc.years)

    # Historical kt from original fit (optional)
    has_hist = hasattr(fc, "fit") and fc.fit is not None
    if has_hist:
        kt_hist = np.atleast_2d(fc.fit.kt)   # (N, n_hist)
        years_hist = fc.fit.years
    else:
        kt_hist = None
        years_hist = None

    ncols = min(N, 3)
    nrows = (N + ncols - 1) // ncols
    if fig is None:
        fig = Figure(figsize=(5 * ncols, 4 * nrows))

    axes = fig.subplots(nrows, ncols, squeeze=False)

    for i in range(N):
        row = i // ncols
        col = i % ncols
        ax = axes[row][col]
        label = f"$\\kappa_t^{{({i+1})}}$" if N > 1 else "$\\kappa_t$"

        # Historical
        if has_hist:
            ax.plot(years_hist, kt_hist[i], label="Historical")

        # Forecast central
        ax.plot(years_fc, kt_central[i], label="Forecast")

        # Confidence bands
        if has_bands:
            ax.fill_between(
                years_fc,
                kt_lower[i],
                kt_upper[i],
                alpha=0.2,
                label="CI",
            )

        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.legend()

    # Hide unused subplots
    for i in range(N, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def plot_fan(
    sim,
    age: int,
    *,
    levels: tuple[float, ...] = (0.5, 0.8, 0.95),
    fig: Figure | None = None,
) -> Figure:
    """Fan chart of simulated mortality rates at a given age.

    Expects ``sim`` to expose:
    - ``sim.rates``: shape (n_ages, h, nsim) array of simulated rates
    - ``sim.ages``: age labels, length n_ages
    - ``sim.years``: forecast year labels, length h
    - ``sim.fit``: original FitStMoMo (optional, for historical rates)

    Parameters
    ----------
    sim:
        A SimStMoMo simulation object.
    age:
        The age at which to draw the fan chart.
    levels:
        Quantile coverage levels for fan bands (e.g. 0.95 → 2.5th–97.5th pct).
    fig:
        Optional existing Figure; if None a new one is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ages_arr = np.asarray(sim.ages)
    idx = np.where(ages_arr == age)[0]
    if len(idx) == 0:
        raise ValueError(f"Age {age} not found in simulation ages.")
    age_idx = int(idx[0])

    # sim.rates shape: (n_ages, h, nsim)
    rates_sim = np.asarray(sim.rates)[age_idx, :, :]  # (h, nsim)
    years_fc = np.asarray(sim.years)

    if fig is None:
        fig = Figure(figsize=(9, 5))

    ax = fig.add_subplot(111)

    # Central (median)
    median = np.median(rates_sim, axis=1)
    ax.plot(years_fc, median, label="Median")

    # Fan bands
    # Using a single color with decreasing alpha for fan levels is often cleaner
    for level in sorted(levels):
        lo_pct = 100.0 * (1.0 - level) / 2.0
        hi_pct = 100.0 - lo_pct
        lo = np.percentile(rates_sim, lo_pct, axis=1)
        hi = np.percentile(rates_sim, hi_pct, axis=1)
        ax.fill_between(years_fc, lo, hi, alpha=0.15,
                        label=f"{int(level * 100)}% CI")

    # Historical rates (optional)
    if hasattr(sim, "fit") and sim.fit is not None:
        hist_ages = np.asarray(sim.fit.ages)
        hist_idx = np.where(hist_ages == age)[0]
        if len(hist_idx) > 0:
            hist_rates = sim.fit.fitted_rates[int(hist_idx[0]), :]
            ax.plot(sim.fit.years, hist_rates, color="#64748b", label="Historical (fitted)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mortality rate")
    ax.set_title(f"Fan chart — age {age}")
    ax.legend()

    fig.tight_layout()
    return fig
