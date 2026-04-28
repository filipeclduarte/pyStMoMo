# Lee-Carter Model

## Model Specification

The Lee-Carter (1992) model:

$$\log(\mu_{xt}) = \alpha_x + \beta_x \kappa_t$$

with identifiability constraints $\sum_x \beta_x = 1$ and $\sum_t \kappa_t = 0$.

## Fitting

```python
import pystmomo as ps

data = ps.load_ew_male()
fit = ps.lc().fit(data.deaths, data.exposures,
                  ages=data.ages, years=data.years)
```

## Inspecting Parameters

```python
import numpy as np

# α_x: baseline log-mortality by age
print(fit.ax)          # shape (35,)

# β_x: age sensitivity to the period index
print(fit.bx[:, 0])   # shape (35,)

# κ_t: period (time) index
print(fit.kt[0])       # shape (51,)

# Check constraint: sum(β_x) = 1
print(np.sum(fit.bx))  # ≈ 1.0
```

## Visualising

```python
ps.plot_parameters(fit)
```

This produces a 3-panel chart: $\alpha_x$ vs age, $\beta_x$ vs age, $\kappa_t$ vs year.

## Forecasting & Simulation

```python
fc   = ps.forecast(fit, h=20)
sim  = ps.simulate(fit, nsim=1000, h=20, seed=42)
boot = ps.semiparametric_bootstrap(fit, nboot=500, seed=0)
```

## References

Lee, R.D. & Carter, L.R. (1992). Modeling and Forecasting U.S. Mortality.
*Journal of the American Statistical Association*, 87(419), 659–671.
