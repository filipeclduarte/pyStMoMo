# CBD Family

!!! info "Probability of death, not mortality rate"
    CBD-family models use the **logit link** and the **Binomial** distribution.
    `fit.fitted_rates` and `fc.rates` return the **probability of death** $q_{xt}$,
    not the central mortality rate $\mu_{xt}$.  See
    [Rates vs Probabilities](rates_vs_probs.md) for conversion formulas and
    guidance on exposure type.

## CBD (Cairns-Blake-Dowd 2006)

$$\text{logit}(q_{xt}) = \kappa_t^{(1)} + (x - \bar{x})\,\kappa_t^{(2)}$$

```python
fit = ps.cbd().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)
```

## M6 (CBD + cohort)

$$\text{logit}(q_{xt}) = \kappa_t^{(1)} + (x-\bar{x})\,\kappa_t^{(2)} + \gamma_{t-x}$$

```python
fit = ps.m6().fit(data.deaths, data.exposures,
                  ages=data.ages, years=data.years)
```

## M7 (CBD + quadratic + cohort)

Adds a quadratic age effect:

$$\text{logit}(q_{xt}) = \kappa_t^{(1)} + (x-\bar{x})\,\kappa_t^{(2)} + [(x-\bar{x})^2 - \hat\sigma^2_x]\,\kappa_t^{(3)} + \gamma_{t-x}$$

```python
fit = ps.m7().fit(data.deaths, data.exposures,
                  ages=data.ages, years=data.years)
```

## M8 (CBD + age-at-zero cohort)

Uses a centered cohort age function with pivot age $x_c$:

$$\text{logit}(q_{xt}) = \kappa_t^{(1)} + (x-\bar{x})\,\kappa_t^{(2)} + (x_c - x)\,\gamma_{t-x}$$

```python
fit = ps.m8(xc=89.5).fit(data.deaths, data.exposures,
                          ages=data.ages, years=data.years)
```

## References

Cairns, A.J.G., Blake, D. & Dowd, K. (2006). A Two-Factor Model for Stochastic
Mortality with Parameter Uncertainty. *Journal of Risk and Insurance*, 73(4), 687–718.
