# Forecasting

## Basic Forecast

```python
import pystmomo as ps

data = ps.load_ew_male()
fit  = ps.lc().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)

fc = ps.forecast(fit, h=20)
print(fc.rates.shape)      # (35, 20) — mortality rates by age × year
print(fc.years_f)          # [2012, 2013, ..., 2031]
```

## Choosing a kt Method

```python
# Default: Multivariate Random Walk with Drift
fc_mrwd = ps.forecast(fit, h=20, kt_method="mrwd")

# ARIMA(1,1,0) per kt index
fc_arima = ps.forecast(fit, h=20, kt_method="arima")
```

## Jump-off Choice

The jump-off affects whether the last observed rates or fitted rates are used as the baseline:

```python
fc = ps.forecast(fit, h=20, jump_choice="actual")   # last observed (default)
fc = ps.forecast(fit, h=20, jump_choice="fit")      # last fitted
```

## Confidence Intervals

Forecast confidence intervals are available from the `ForStMoMo` result:

```python
print(fc.kt_lower.shape)   # (N, h) — lower CI for period indexes
print(fc.kt_upper.shape)   # (N, h)
```

## Simulation-Based Fan Charts

For simulation-based uncertainty (recommended), use `simulate()`:

```python
sim = ps.simulate(fit, nsim=1000, h=20, seed=42)
ps.plot_fan(sim, age=65)
```
