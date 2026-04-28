"""Unit tests for age functions."""
import numpy as np
import pytest

from pystmomo.core.age_functions import (
    ConstantAgeFun,
    LinearAgeFun,
    QuadraticAgeFun,
    CenteredCohortAgeFun,
    NonParametricAgeFun,
)

AGES = np.arange(55, 90, dtype=float)


def test_constant_fun():
    f = ConstantAgeFun()
    out = f(AGES)
    assert out.shape == AGES.shape
    np.testing.assert_array_equal(out, 1.0)
    assert f.is_parametric


def test_linear_fun_zero_mean():
    f = LinearAgeFun()
    out = f(AGES)
    assert out.shape == AGES.shape
    assert abs(np.mean(out)) < 1e-10


def test_quadratic_fun_zero_mean():
    f = QuadraticAgeFun()
    out = f(AGES)
    assert abs(np.mean(out)) < 1e-10


def test_centered_cohort_fun():
    f = CenteredCohortAgeFun(xc=89.5)
    out = f(AGES)
    np.testing.assert_allclose(out, 89.5 - AGES)


def test_nonparametric_raises():
    f = NonParametricAgeFun()
    with pytest.raises(RuntimeError):
        f(AGES)
    assert not f.is_parametric
