# Forecasting & Uncertainty

## Period Index Forecasting

### Multivariate Random Walk with Drift (MRWD)

The default model for period indexes $\kappa_t$:

$$\Delta \kappa_t = \delta + \varepsilon_t, \quad \varepsilon_t \sim \text{MVN}(0, \Sigma)$$

Forecast mean at horizon $h$:

$$\hat{\kappa}_{T+h} = \kappa_T + h\,\hat\delta$$

The $(1-\alpha)$ confidence interval accounts for both process variance and estimation uncertainty in $\hat\delta$:

$$\text{Var}(\kappa_{T+h}) = h\,\Sigma + \frac{h^2}{T}\,\Sigma$$

### Independent ARIMA

Each $\kappa_t^{(i)}$ can be modelled with a separate ARIMA process (wrapping `statsmodels`). Default is ARIMA(0,1,0) with drift (= MRWD).

## Simulation

`simulate(fit, nsim, h, seed)` generates `nsim` mortality trajectories:

1. Simulate $\kappa_t$ paths via MRWD or ARIMA: shape $(N, h, \text{nsim})$
2. Forecast new cohort $\gamma_c$ values (for cohort models)
3. Assemble $\eta_{xt}^{(s)} = \alpha_x + \sum_i \beta_x^{(i)} \kappa_t^{(i,s)} + \beta_x^{(0)} \gamma_c^{(s)}$
4. Apply inverse link to get rates: shape $(n_\text{ages}, h, \text{nsim})$

## Bootstrap

### Semiparametric Bootstrap (Brouhns et al. 2005)

Resample death counts from their distributional assumption:

$$D_{xt}^* \sim \text{Poisson}(\hat{D}_{xt}) \quad \text{(Poisson models)}$$
$$D_{xt}^* \sim \text{Binomial}(E_{xt},\, \hat{q}_{xt}) \quad \text{(Binomial models)}$$

### Residual Bootstrap (Renshaw & Haberman 2008)

Resample deviance residuals $r_{xt}$, then invert to obtain bootstrap counts:

$$D_{xt}^* \approx \left(\sqrt{\hat{D}_{xt}} + \frac{r_{xt}^*}{2}\right)^2 \quad \text{(Poisson)}$$
