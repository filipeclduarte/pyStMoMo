# pyStMoMo

**Stochastic Mortality Modelling in Python** — a faithful Python port of the
[StMoMo](https://github.com/amvillegas/StMoMo) R package by Villegas et al.

pyStMoMo implements the **Generalised Age-Period-Cohort (GAPC)** framework for
fitting, forecasting, simulating, and validating stochastic mortality models.
It targets actuaries and demographers who work in Python and need production-quality
mortality tools without switching to R.

---

## Quick Start

```python
import pystmomo as ps

# Load bundled England & Wales male data (ages 55–89, years 1961–2011)
data = ps.load_ew_male()

# Fit a Lee-Carter model
fit = ps.lc().fit(data.deaths, data.exposures,
                  ages=data.ages, years=data.years)
print(fit)

# Forecast 20 years ahead
fc = ps.forecast(fit, h=20)

# Simulate 1 000 mortality trajectories
sim = ps.simulate(fit, nsim=1000, h=20, seed=42)

# Bootstrap parameter uncertainty (200 replicates)
boot = ps.semiparametric_bootstrap(fit, nboot=200, seed=0)

# Visualise
ps.plot_parameters(fit)
ps.plot_fan(sim, age=65)
```

---

## Available Models

| Function | Predictor | Link |
|---|---|---|
| `lc()` | α_x + β_x κ_t | log |
| `cbd()` | κ_t¹ + (x−x̄) κ_t² | logit |
| `apc()` | α_x + κ_t + γ_{t−x} | log |
| `rh()` | α_x + β_x κ_t + γ_{t−x} | log |
| `m6()` | CBD + γ_{t−x} | logit |
| `m7()` | CBD + quad + γ_{t−x} | logit |
| `m8(xc)` | CBD + (xc−x) γ_{t−x} | logit |

---

## Installation

```bash
pip install pystmomo
# With parallel bootstrap support:
pip install "pystmomo[parallel]"
```

---

## Key Features

- **7 pre-built GAPC models** (Lee-Carter, CBD, APC, Renshaw-Haberman, M6/M7/M8)
- **Two fitting paths**: statsmodels GLM (parametric) + block-coordinate IRLS with SVD init (bilinear)
- **Forecasting**: Multivariate Random Walk with Drift or per-index ARIMA
- **Simulation**: vectorised mortality trajectory simulation (n_ages × h × nsim)
- **Bootstrap**: semiparametric (Poisson/Binomial) and deviance residual resampling
- **Diagnostics**: deviance/Pearson residuals, period cross-validation
- **Plotting**: parameter panels, forecast fan charts, residual heatmaps
