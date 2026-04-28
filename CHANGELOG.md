# Changelog

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
