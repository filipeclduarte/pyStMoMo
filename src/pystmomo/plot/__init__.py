"""Plotting utilities for pyStMoMo."""
from .parameters import plot_parameters
from .residual_plot import plot_residual_heatmap, plot_residual_scatter
from .forecast_plot import plot_forecast, plot_fan
from ._style import set_style

# Apply premium styling globally when plot is imported
set_style()

__all__ = [
    "plot_parameters",
    "plot_residual_heatmap",
    "plot_residual_scatter",
    "plot_forecast",
    "plot_fan",
]
