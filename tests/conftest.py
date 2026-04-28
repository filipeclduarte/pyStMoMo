"""Shared fixtures for pyStMoMo tests."""
import numpy as np
import pytest

import pystmomo as ps


@pytest.fixture(scope="session")
def ew_data():
    return ps.load_ew_male()


@pytest.fixture(scope="session")
def lc_fit(ew_data):
    return ps.lc().fit(
        ew_data.deaths, ew_data.exposures,
        ages=ew_data.ages, years=ew_data.years,
    )


@pytest.fixture(scope="session")
def cbd_fit(ew_data):
    return ps.cbd().fit(
        ew_data.deaths, ew_data.exposures,
        ages=ew_data.ages, years=ew_data.years,
    )


@pytest.fixture(scope="session")
def apc_fit(ew_data):
    return ps.apc().fit(
        ew_data.deaths, ew_data.exposures,
        ages=ew_data.ages, years=ew_data.years,
    )
