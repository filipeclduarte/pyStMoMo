# Link Functions — μ_xt vs q_xt

## Two statistical frameworks

GAPC models split into two groups depending on the link function chosen
for the linear predictor η_xt.

---

## Log link — Central mortality rate (Poisson)

Models: **LC, APC, RH**

$$
D_{xt} \sim \text{Poisson}(E_{xt}\,\mu_{xt}), \qquad
\log(\mu_{xt}) = \eta_{xt}
$$

| Symbol | Meaning |
|--------|---------|
| $\mu_{xt}$ | Central mortality rate — expected deaths per person-year lived |
| $E_{xt}$ | Central exposed-to-risk (person-years lived in year $t$ at age $x$) |
| $D_{xt}$ | Observed death counts |

The fitted values returned by `fit.fitted_rates` and `fc.rates` are **central
mortality rates** $\mu_{xt}$.  They can exceed 1 at very old ages.

---

## Logit link — Probability of death (Binomial)

Models: **CBD, M6, M7, M8**

$$
D_{xt} \sim \text{Binomial}(E_{xt},\,q_{xt}), \qquad
\text{logit}(q_{xt}) = \eta_{xt}
$$

| Symbol | Meaning |
|--------|---------|
| $q_{xt}$ | Probability of dying between exact ages $x$ and $x+1$ |
| $E_{xt}$ | Initial exposed-to-risk (number of lives observed at age $x$ at start of year $t$) |
| $D_{xt}$ | Observed death counts |

The fitted values returned by `fit.fitted_rates` and `fc.rates` are
**probabilities of death** $q_{xt} \in (0, 1)$.

---

## Converting between μ and q

Under the **Uniform Distribution of Deaths (UDD)** assumption within each year of age:

$$
q_{xt} = 1 - e^{-\mu_{xt}}
\qquad \Longleftrightarrow \qquad
\mu_{xt} = -\log(1 - q_{xt})
$$

```python
import numpy as np
import pystmomo as ps

data = ps.load_ew_male()
lc_fit  = ps.lc().fit(data.deaths, data.exposures,
                      ages=data.ages, years=data.years)
cbd_fit = ps.cbd().fit(data.deaths, data.exposures,
                       ages=data.ages, years=data.years)

# LC gives μ → convert to q
mu  = lc_fit.fitted_rates          # central mortality rates
q   = 1 - np.exp(-mu)             # UDD conversion to probability of death

# CBD gives q directly
q_cbd = cbd_fit.fitted_rates

# Compare forecasts on the same scale
fc_lc  = ps.forecast(lc_fit,  h=20)
fc_cbd = ps.forecast(cbd_fit, h=20)

q_lc_forecast = 1 - np.exp(-fc_lc.rates)   # now comparable to fc_cbd.rates
```

---

## Exposure type matters for CBD

The Poisson group uses **central** exposure (person-years lived).
The Binomial group ideally uses **initial** exposure (lives at start of year).
The two are related by:

$$
E^o_{xt} \approx E^c_{xt} + \tfrac{1}{2}\,D_{xt}
$$

where $E^o$ is initial and $E^c$ is central.

```python
# If you only have central exposure, approximate initial exposure for CBD:
Ext_initial = data.exposures + 0.5 * data.deaths

cbd_fit = ps.cbd().fit(data.deaths, Ext_initial,
                       ages=data.ages, years=data.years)
```

In practice, for adult ages (50+), the difference is small and both exposure
types yield similar fitted values.

---

## Summary table

| Model | Link | `fitted_rates` | Exposure type |
|-------|------|----------------|---------------|
| LC    | log  | μ_xt — central rate | Central |
| APC   | log  | μ_xt — central rate | Central |
| RH    | log  | μ_xt — central rate | Central |
| CBD   | logit | q_xt — prob. of death | Initial (ideally) |
| M6    | logit | q_xt — prob. of death | Initial (ideally) |
| M7    | logit | q_xt — prob. of death | Initial (ideally) |
| M8    | logit | q_xt — prob. of death | Initial (ideally) |
