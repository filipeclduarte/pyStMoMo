"""Data loading utilities for pyStMoMo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent / "csv"


@dataclass
class StMoMoData:
    """Container for mortality data.

    Parameters
    ----------
    deaths:
        Death counts matrix, shape (n_ages, n_years).
    exposures:
        Central or initial exposures matrix, shape (n_ages, n_years).
    ages:
        Age labels, length n_ages.
    years:
        Calendar year labels, length n_years.
    type:
        ``"central"`` (central death rates μ_xt) or ``"initial"`` (initial
        death probabilities q_xt).
    series:
        Description of the population series, e.g. ``"males"``.
    label:
        Short identifier for the dataset.
    """

    deaths: np.ndarray
    exposures: np.ndarray
    ages: np.ndarray
    years: np.ndarray
    type: Literal["central", "initial"] = "central"
    series: str = ""
    label: str = ""

    @property
    def n_ages(self) -> int:
        return len(self.ages)

    @property
    def n_years(self) -> int:
        return len(self.years)

    def central2initial(self) -> "StMoMoData":
        """Convert central exposures to initial exposures.

        Uses the approximation ``E_xt^0 ≈ E_xt^c + 0.5 * D_xt``, which is
        standard in the HMD methodology.

        Returns
        -------
        StMoMoData
            New object with ``type = "initial"`` and adjusted exposures.
        """
        if self.type == "initial":
            return self
        new_exp = self.exposures + 0.5 * self.deaths
        return StMoMoData(
            deaths=self.deaths,
            exposures=new_exp,
            ages=self.ages.copy(),
            years=self.years.copy(),
            type="initial",
            series=self.series,
            label=self.label,
        )

    def subset(
        self,
        ages: np.ndarray | list[int] | None = None,
        years: np.ndarray | list[int] | None = None,
    ) -> "StMoMoData":
        """Return a subset of the data restricted to given ages and/or years."""
        age_idx = (
            np.isin(self.ages, ages)
            if ages is not None
            else np.ones(self.n_ages, dtype=bool)
        )
        year_idx = (
            np.isin(self.years, years)
            if years is not None
            else np.ones(self.n_years, dtype=bool)
        )
        return StMoMoData(
            deaths=self.deaths[np.ix_(age_idx, year_idx)],
            exposures=self.exposures[np.ix_(age_idx, year_idx)],
            ages=self.ages[age_idx],
            years=self.years[year_idx],
            type=self.type,
            series=self.series,
            label=self.label,
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"StMoMoData(label={self.label!r}, series={self.series!r}, "
            f"ages={self.ages[0]}–{self.ages[-1]}, "
            f"years={self.years[0]}–{self.years[-1]}, type={self.type!r})"
        )


def load_ew_male(
    ages: np.ndarray | list[int] | None = None,
    years: np.ndarray | list[int] | None = None,
) -> StMoMoData:
    """Load the bundled England & Wales male mortality dataset.

    The dataset contains synthetic deaths and central exposures for English and
    Welsh males, ages 0–100, calendar years 1961–2011.  It was generated to
    match published Human Mortality Database mortality patterns for illustrative
    purposes.

    Parameters
    ----------
    ages:
        If provided, restrict to these ages.  Defaults to ages 55–89, which
        are standard for CBD-family models.
    years:
        If provided, restrict to these calendar years.  Defaults to all years
        (1961–2011).

    Returns
    -------
    StMoMoData
        Central mortality data for the requested subset.

    Examples
    --------
    >>> import pystmomo as ps
    >>> data = ps.load_ew_male()
    >>> data.deaths.shape
    (35, 51)
    >>> data = ps.load_ew_male(ages=range(55, 90), years=range(1980, 2012))
    """
    dxt = pd.read_csv(_DATA_DIR / "EWMaleData_Dxt.csv", index_col="age")
    ext = pd.read_csv(_DATA_DIR / "EWMaleData_Ext.csv", index_col="age")

    all_ages = dxt.index.to_numpy(dtype=int)
    all_years = dxt.columns.to_numpy(dtype=int)

    if ages is None:
        ages_sel = np.arange(55, 90)
    else:
        ages_sel = np.asarray(ages, dtype=int)

    years_sel = np.asarray(years, dtype=int) if years is not None else all_years

    age_mask = np.isin(all_ages, ages_sel)
    year_mask = np.isin(all_years, years_sel)

    data = StMoMoData(
        deaths=dxt.values[np.ix_(age_mask, year_mask)].astype(float),
        exposures=ext.values[np.ix_(age_mask, year_mask)].astype(float),
        ages=all_ages[age_mask],
        years=all_years[year_mask],
        type="central",
        series="males",
        label="EWMale",
    )
    return data


def load_hmd_csv(
    deaths_file: str | Path,
    exposures_file: str | Path,
    *,
    series: str = "",
    label: str = "",
    exposure_type: Literal["central", "initial"] = "central",
) -> StMoMoData:
    """Load mortality data from HMD-format CSV files.

    HMD CSV files can be downloaded from `https://www.mortality.org` after
    registration.  The expected format has columns ``Year``, ``Age``, and
    ``Total`` (or ``Male`` / ``Female``).

    Parameters
    ----------
    deaths_file:
        Path to the HMD deaths CSV (e.g., ``Deaths_1x1.txt``).
    exposures_file:
        Path to the HMD exposures CSV (e.g., ``Exposures_1x1.txt``).
    series:
        Population series identifier (``"total"``, ``"male"``, ``"female"``).
    label:
        Short dataset label.
    exposure_type:
        Whether the exposures file contains central (``"central"``) or initial
        (``"initial"``) exposures.

    Returns
    -------
    StMoMoData
    """
    def _parse_hmd(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            na_values=".",
            dtype=str,
        )
        df.columns = df.columns.str.strip()
        # Handle "110+" age notation
        df["Age"] = df["Age"].str.replace("+", "", regex=False).astype(int)
        df["Year"] = df["Year"].astype(int)
        col_choices = ["Total", "Male", "Female"]
        val_col = next((c for c in col_choices if c in df.columns), df.columns[-1])
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        return df.pivot(index="Age", columns="Year", values=val_col)

    dxt_df = _parse_hmd(deaths_file)
    ext_df = _parse_hmd(exposures_file)

    common_ages = np.intersect1d(dxt_df.index.values, ext_df.index.values)
    common_years = np.intersect1d(dxt_df.columns.values, ext_df.columns.values)

    dxt_df = dxt_df.loc[common_ages, common_years]
    ext_df = ext_df.loc[common_ages, common_years]

    return StMoMoData(
        deaths=dxt_df.values.astype(float),
        exposures=ext_df.values.astype(float),
        ages=np.asarray(common_ages, dtype=int),
        years=np.asarray(common_years, dtype=int),
        type=exposure_type,
        series=series,
        label=label,
    )
