"""Build sparse GLM design matrices for fully-parametric GAPC models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from .stmomo import StMoMo


@dataclass
class ColMap:
    """Maps design-matrix column ranges to parameter blocks."""

    n_ages: int
    n_years: int
    n_cohorts: int
    ax_cols: list[int] = field(default_factory=list)
    kt_cols: list[list[int]] = field(default_factory=list)   # one list per period term
    gc_cols: list[int] = field(default_factory=list)

    def n_ax(self) -> int:
        return len(self.ax_cols)

    def n_kt(self, i: int) -> int:
        return len(self.kt_cols[i]) if i < len(self.kt_cols) else 0

    def n_gc(self) -> int:
        return len(self.gc_cols)

    def total_cols(self) -> int:
        return (
            len(self.ax_cols)
            + sum(len(c) for c in self.kt_cols)
            + len(self.gc_cols)
        )


def build_design_matrix(
    model: "StMoMo",
    ages: np.ndarray,
    years: np.ndarray,
    cohorts: np.ndarray,
    wxt: np.ndarray,
) -> tuple[sp.csr_matrix, ColMap, np.ndarray]:
    """Build the sparse design matrix X for a fully-parametric GAPC model.

    Cells where wxt == 0 are excluded (zero-weight rows are dropped).

    Parameters
    ----------
    model:
        Fully-parametric :class:`StMoMo` specification.
    ages, years, cohorts:
        Age, year, and cohort vectors.
    wxt:
        Weight matrix, shape (n_ages, n_years).

    Returns
    -------
    X:
        Sparse CSR design matrix.
    col_map:
        Column metadata for unpacking fitted parameters.
    row_mask:
        Boolean array, shape (n_ages * n_years,), indicating included rows.
    """
    n_ages, n_years = len(ages), len(years)
    n_cohorts = len(cohorts)
    n_cells = n_ages * n_years

    # Flat indices
    age_idx = np.repeat(np.arange(n_ages), n_years)      # (n_cells,)
    year_idx = np.tile(np.arange(n_years), n_ages)        # (n_cells,)

    row_mask = wxt.ravel() > 0
    n_obs = row_mask.sum()

    col_map = ColMap(n_ages=n_ages, n_years=n_years, n_cohorts=n_cohorts)

    blocks: list[sp.csr_matrix] = []
    col_offset = 0

    # --- α_x block (one dummy per age) ---
    if model.static_age_fun:
        rows = np.arange(n_cells)[row_mask]
        cols = age_idx[row_mask]
        data = np.ones(n_obs)
        ax_block = sp.csr_matrix(
            (data, (np.arange(n_obs), cols)), shape=(n_obs, n_ages)
        )
        blocks.append(ax_block)
        col_map.ax_cols = list(range(col_offset, col_offset + n_ages))
        col_offset += n_ages

    # --- κ_t^(i) blocks ---
    for i, af in enumerate(model.period_age_fun):
        # af is parametric (NonParametricAgeFun triggers the other path)
        fx = af(ages)   # shape (n_ages,)
        # For each year t, one parameter κ_t^(i).  Column t gets value f_i(x)
        # in rows where year_idx == t.
        rows_obs = np.arange(n_obs)
        cols_obs = year_idx[row_mask]
        data_obs = fx[age_idx[row_mask]]
        # Multiply f_i(x) by indicator of year column
        # Build as sparse: each obs row gets value f_i(x) in its year column
        kt_block = sp.csr_matrix(
            (data_obs, (rows_obs, cols_obs)), shape=(n_obs, n_years)
        )
        blocks.append(kt_block)
        col_map.kt_cols.append(list(range(col_offset, col_offset + n_years)))
        col_offset += n_years

    # --- γ_c block ---
    if model.cohort_age_fun is not None:
        f0 = model.cohort_age_fun(ages)   # shape (n_ages,)
        cohort_vals = years[year_idx] - ages[age_idx]  # (n_cells,)
        cohort_map = {int(c): j for j, c in enumerate(cohorts)}

        rows_list, cols_list, data_list = [], [], []
        obs_row = 0
        for cell in range(n_cells):
            if not row_mask[cell]:
                continue
            c = int(cohort_vals[cell])
            if c in cohort_map:
                rows_list.append(obs_row)
                cols_list.append(cohort_map[c])
                data_list.append(float(f0[age_idx[cell]]))
            obs_row += 1

        gc_block = sp.csr_matrix(
            (data_list, (rows_list, cols_list)), shape=(n_obs, n_cohorts)
        )
        blocks.append(gc_block)
        col_map.gc_cols = list(range(col_offset, col_offset + n_cohorts))
        col_offset += n_cohorts

    if not blocks:
        X = sp.csr_matrix((n_obs, 0))
    else:
        X = sp.hstack(blocks, format="csr")

    return X, col_map, row_mask


def unpack_params(
    params: np.ndarray,
    col_map: ColMap,
    ages: np.ndarray,
    years: np.ndarray,
    cohorts: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Unpack a flat GLM parameter vector into (ax, bx, kt, b0x, gc).

    Returns
    -------
    ax:
        Shape (n_ages,) or None.
    bx:
        Shape (n_ages, N) — empty 2-D array if N=0.
    kt:
        Shape (N, n_years).
    b0x:
        None (cohort age modulation fitted separately in bilinear path).
    gc:
        Shape (n_cohorts,) or None.
    """
    n_ages, n_years, n_cohorts = col_map.n_ages, col_map.n_years, col_map.n_cohorts

    ax: np.ndarray | None = None
    if col_map.ax_cols:
        ax = params[col_map.ax_cols]

    N = len(col_map.kt_cols)
    bx = np.empty((n_ages, 0))
    kt = np.empty((0, n_years))
    if N > 0:
        kt = np.empty((N, n_years))
        for i, cols in enumerate(col_map.kt_cols):
            kt[i] = params[cols]
        # bx is identity (age functions are pre-specified), stored as columns
        # We return bx as the age-function evaluations, shape (n_ages, N)
        bx = np.ones((n_ages, N))  # placeholder; caller uses age functions

    gc: np.ndarray | None = None
    if col_map.gc_cols:
        gc = params[col_map.gc_cols]

    return ax, bx, kt, None, gc
