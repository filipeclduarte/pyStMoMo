# Changelog

## [0.1.1] — 2026-04-28

### Fixed
- Test helper `_make_forecast_fn` broken for N>1 (3 test failures).
- M7 constraint had unused expression on line 157; cleaned up redistribution logic.

### Performance
- Vectorised cohort index matrix computation (`_cohort_index_matrix`) — replaced O(n_ages × n_years) Python loops with numpy array indexing.
- Vectorised bilinear fit `_build_eta` and final eta reconstruction.
- Vectorised forecast rate computation (`_compute_forecast_rates`).
- Fully vectorised simulation loop — replaced O(nsim × n_ages × h) Python iterations with `np.einsum` and broadcasting (~100× speedup for typical configurations).
- Vectorised new-cohort discovery in simulation via `np.unique`.

### Changed
- `_build_gc_dict` replaced by `_build_gc_arrays` returning sorted numpy arrays instead of a Python dict.
- Premium plot styling module (`_style.py`) added to the plot package.

## [0.1.0] — 2026-04-27

### Added
- Initial release of pyStMoMo.
- GAPC model specification framework (`StMoMo` class).
- Pre-built models: Lee-Carter (LC), CBD, APC, Renshaw-Haberman (RH), M6, M7, M8.
- Fitting via statsmodels GLM (parametric path) and block-coordinate IRLS (bilinear path).
- Forecasting with Multivariate Random Walk with Drift (MRWD) and independent ARIMA.
- Monte Carlo simulation of future mortality trajectories.
- Residual and semiparametric bootstrap for parameter uncertainty.
- Deviance, Pearson, and response residuals.
- Period cross-validation.
- Matplotlib plotting: parameter plots, forecast fan charts, residual heatmaps.
- Bundled England & Wales male mortality data (ages 55–89, years 1961–2011).
- MkDocs Material documentation with MathJax theory pages.
- Example Jupyter notebooks.
