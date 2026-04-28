"""Plotting utilities for pyStMoMo."""
from ._style import set_style
from .forecast_plot import plot_fan, plot_forecast
from .parameters import plot_parameters
from .residual_plot import plot_residual_heatmap, plot_residual_scatter

# Apply premium styling globally when plot is imported
set_style()

__all__ = [
    "plot_parameters",
    "plot_residual_heatmap",
    "plot_residual_scatter",
    "plot_forecast",
    "plot_fan",
]
