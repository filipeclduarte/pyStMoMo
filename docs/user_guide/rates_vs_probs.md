# Rates vs Probabilities

pyStMoMo models fall into two groups depending on their link function.
Understanding the distinction is essential for comparing models and using
results in actuarial calculations.

## What does `fitted_rates` return?

```python
import pystmomo as ps
import numpy as np

data = ps.load_ew_male()

lc_fit  = ps.lc().fit(data.deaths, data.exposures,
                      ages=data.ages, years=data.years)
cbd_fit = ps.cbd().fit(data.deaths, data.exposures,
                       ages=data.ages, years=data.years)

print(lc_fit.model.link)    # "log"
print(cbd_fit.model.link)   # "logit"

# LC: central mortality rates μ_xt
print(lc_fit.fitted_rates.max())   # can be > 0.2 at oldest ages

# CBD: probabilities of death q_xt  (always in (0,1))
print(cbd_fit.fitted_rates.max())  # always < 1
```

You can always check which type a model returns:

```python
def rates_type(fit):
    return "q_xt (probability of death)" if fit.model.link == "logit" \
           else "mu_xt (central mortality rate)"

print(rates_type(lc_fit))   # mu_xt (central mortality rate)
print(rates_type(cbd_fit))  # q_xt (probability of death)
```

## Converting between μ and q

Under the Uniform Distribution of Deaths (UDD) assumption:

```python
# μ → q
mu = lc_fit.fitted_rates
q  = 1 - np.exp(-mu)

# q → μ
q_cbd = cbd_fit.fitted_rates
mu    = -np.log(1 - q_cbd)
```

## Comparing forecast outputs

When you compare forecasts from different model families, always convert to
the same scale first:

```python
fc_lc  = ps.forecast(lc_fit,  h=20)
fc_cbd = ps.forecast(cbd_fit, h=20)

# Convert LC forecast to probabilities
q_lc  = 1 - np.exp(-fc_lc.rates)   # (n_ages, 20)
q_cbd = fc_cbd.rates                # (n_ages, 20)

# Now directly comparable
diff = q_lc - q_cbd
```

## Exposure input for CBD models

CBD, M6, M7, M8 are built on the Binomial distribution, which requires the
**initial exposed-to-risk** (number of lives at the start of the year), not
the central exposure (person-years lived) used by LC/APC/RH.

If your data only contains central exposure:

```python
# Approximate initial exposure from central exposure
Ext_initial = data.exposures + 0.5 * data.deaths

# Use it when fitting CBD-family models
cbd_fit = ps.cbd().fit(data.deaths, Ext_initial,
                       ages=data.ages, years=data.years)
m7_fit  = ps.m7().fit(data.deaths, Ext_initial,
                      ages=data.ages, years=data.years)
```

The Human Mortality Database (HMD) provides both exposure types in separate
files (`Exposures_1x1.txt` for central, `Deaths_1x1.txt` for deaths).

## Using results in actuarial tables

```python
import pandas as pd

fc = ps.forecast(cbd_fit, h=20)   # q_xt directly

# Build a life table extract for age 65 over the forecast horizon
age_idx = list(data.ages).index(65)
q_65 = fc.rates[age_idx, :]   # shape (20,)

table = pd.DataFrame({
    "year": fc.years_f,
    "q_65": q_65,
    "survival_prob": np.cumprod(1 - q_65),
})
print(table)
```

For Poisson models (LC, APC, RH), convert first:

```python
fc_lc = ps.forecast(lc_fit, h=20)
q_65  = 1 - np.exp(-fc_lc.rates[age_idx, :])   # UDD conversion
```

!!! tip
    The `StMoMoData.central2initial()` helper converts a dataset's exposures
    from central to initial in one step, making it easy to switch between
    model families.
