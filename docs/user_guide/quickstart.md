# Getting Started

## Installation

```bash
pip install pystmomo
```

For parallel bootstrap (uses `joblib`):

```bash
pip install "pystmomo[parallel]"
```

## Loading Data

pyStMoMo ships with England & Wales male mortality data (ages 0–100, years 1961–2011):

```python
import pystmomo as ps

data = ps.load_ew_male()  # default ages 55–89
print(data.deaths.shape)    # (35, 51)
print(data.ages)            # [55 56 ... 89]
print(data.years)           # [1961 1962 ... 2011]
```

You can also subset:

```python
data_sub = data.subset(ages=range(60, 80), years=range(1980, 2012))
```

To load your own data in HMD format:

```python
data = ps.load_hmd_csv("Deaths_1x1.txt", "Exposures_1x1.txt",
                        ages=range(50, 90), years=range(1950, 2020))
```

## Fitting a Model

```python
fit = ps.lc().fit(
    data.deaths, data.exposures,
    ages=data.ages, years=data.years,
)
print(fit)
# FitStMoMo(model=lc, ages=55-89, years=1961-2011, deviance=1522.9, converged=True)
```

## Forecasting

```python
fc = ps.forecast(fit, h=20)
print(fc.rates.shape)  # (35, 20)

# Plot a fan chart
ps.plot_fan(ps.simulate(fit, nsim=1000, h=20, seed=42), age=65)
```

## Bootstrap Uncertainty

```python
boot = ps.semiparametric_bootstrap(fit, nboot=500, seed=0)
lo, hi = boot.parameter_ci("kt", level=0.95)
print(lo.shape)  # (1, 51) — lower bound for κ_t
```

## Residual Diagnostics

```python
ps.plot_residual_heatmap(fit)
ps.plot_residual_scatter(fit)
```
