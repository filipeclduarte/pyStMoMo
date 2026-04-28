"""Unit tests for residual computation."""
import numpy as np
import pytest

from pystmomo.diagnostics.residuals import (
    deviance_residuals,
    pearson_residuals,
    response_residuals,
)


def test_deviance_residuals_squared_sum(lc_fit):
    res = deviance_residuals(lc_fit)
    # sum of squared deviance residuals == model deviance (over active cells)
    mask = lc_fit.wxt > 0
    np.testing.assert_allclose(
        np.sum(res[mask] ** 2), lc_fit.deviance, rtol=1e-4
    )


def test_pearson_residuals_shape(lc_fit):
    res = pearson_residuals(lc_fit)
    assert res.shape == (len(lc_fit.ages), len(lc_fit.years))


def test_response_residuals_shape(lc_fit):
    res = response_residuals(lc_fit)
    assert res.shape == (len(lc_fit.ages), len(lc_fit.years))


def test_masked_cells_are_zero(lc_fit):
    res = deviance_residuals(lc_fit)
    mask = lc_fit.wxt == 0
    np.testing.assert_array_equal(res[mask], 0.0)
