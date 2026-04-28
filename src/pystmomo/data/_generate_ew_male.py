"""
Script to regenerate the bundled EW male synthetic mortality data.

The data approximates England & Wales male central mortality rates 1961-2011
using a Lee-Carter model with parameters consistent with published estimates:

  log(mu_xt) = ax + bx * kt

  ax: HMD-style age pattern for EW males
  bx: age sensitivity (U-shaped, high at young/old ages)
  kt: linear improvement trend 1961→2011

Run this script from the repo root to regenerate the CSVs:
    python src/pystmomo/data/_generate_ew_male.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AGES = np.arange(0, 101)
YEARS = np.arange(1961, 2012)
N_AGES = len(AGES)
N_YEARS = len(YEARS)
RNG = np.random.default_rng(42)

# Approximate Lee-Carter ax for EW males (log mortality rates, 2005 level)
# Based on published HMD estimates; units: log(central mortality rate)
_AX_YOUNG = -3.5 - 0.08 * AGES[:10]          # ages 0-9
_AX_INFANT = np.array([-3.5])                   # age 0 override
_AX_CHILDHOOD = np.linspace(-5.5, -7.2, 15)    # ages 1-15
_AX_YOUNG_ADULT = np.linspace(-7.2, -5.8, 10)  # ages 16-25
_AX_ADULT = np.linspace(-5.8, -5.2, 15)        # ages 26-40
_AX_MIDDLE = np.linspace(-5.2, -3.8, 20)       # ages 41-60
_AX_OLD = np.linspace(-3.8, -1.5, 30)          # ages 61-90
_AX_OLDEST = np.linspace(-1.5, -0.3, 10)       # ages 91-100


def _make_ax() -> np.ndarray:
    ax = np.empty(N_AGES)
    ax[0] = -3.5
    ax[1:16] = np.linspace(-5.5, -7.2, 15)
    ax[16:26] = np.linspace(-7.2, -5.8, 10)
    ax[26:41] = np.linspace(-5.8, -5.2, 15)
    ax[41:61] = np.linspace(-5.2, -3.8, 20)
    ax[61:91] = np.linspace(-3.8, -1.5, 30)
    ax[91:] = np.linspace(-1.5, -0.3, 10)
    return ax


def _make_bx() -> np.ndarray:
    # bx: age sensitivity — higher at infant and oldest-old ages
    bx = np.ones(N_AGES) * 0.03
    bx[:5] = 0.06
    bx[70:] = np.linspace(0.03, 0.015, 31)
    bx /= bx.sum()   # identifiability: sum(bx) = 1
    return bx


def _make_kt() -> np.ndarray:
    # kt: linear downward trend (mortality improving over time)
    kt = np.linspace(25, -25, N_YEARS)
    return kt


def generate_ew_male_data(noise: bool = True) -> tuple[np.ndarray, np.ndarray]:
    ax = _make_ax()
    bx = _make_bx()
    kt = _make_kt()

    log_mu = ax[:, None] + bx[:, None] * kt[None, :]   # (101, 51)
    mu = np.exp(log_mu)

    # Mid-year exposure (synthetic, declining young, growing old population)
    Ext = np.zeros((N_AGES, N_YEARS))
    for i, age in enumerate(AGES):
        if age <= 15:
            base = 350_000 - age * 5_000
        elif age <= 65:
            base = 280_000 + (age - 15) * 2_000
        else:
            base = 380_000 - (age - 65) * 6_000
        base = max(base, 5_000)
        Ext[i, :] = base * (1 + 0.002 * np.arange(N_YEARS))

    expected_deaths = mu * Ext
    Dxt = RNG.poisson(expected_deaths).astype(float) if noise else expected_deaths

    return Dxt, Ext


def main() -> None:
    out = Path(__file__).parent / "csv"
    out.mkdir(exist_ok=True)

    Dxt, Ext = generate_ew_male_data(noise=True)

    pd.DataFrame(Dxt, index=AGES, columns=YEARS).to_csv(
        out / "EWMaleData_Dxt.csv", index_label="age"
    )
    pd.DataFrame(Ext, index=AGES, columns=YEARS).to_csv(
        out / "EWMaleData_Ext.csv", index_label="age"
    )
    print(f"Saved Dxt and Ext to {out}/")


if __name__ == "__main__":
    main()
