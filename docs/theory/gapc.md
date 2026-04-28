# GAPC Framework

## The General Model

Generalised Age-Period-Cohort (GAPC) mortality models (Villegas et al. 2018) share the linear predictor:

$$
\eta_{xt} = \alpha_x + \sum_{i=1}^{N} \beta_x^{(i)} \kappa_t^{(i)} + \beta_x^{(0)} \gamma_{t-x}
$$

where:

- $\alpha_x$ — static age effect (one parameter per age)
- $\beta_x^{(i)}$ — age-modulating function for the $i$-th period index
- $\kappa_t^{(i)}$ — period index (latent time trend), one per year
- $\beta_x^{(0)}$ — age-modulating function for the cohort effect
- $\gamma_c$ — cohort effect, where $c = t - x$ is the birth cohort

## Link Functions

### Poisson (log link)

$$
D_{xt} \sim \text{Poisson}(E_{xt}\, \mu_{xt}), \quad \log(\mu_{xt}) = \eta_{xt}
$$

Used by: LC, APC, RH.

### Binomial (logit link)

$$
D_{xt} \sim \text{Binomial}(E_{xt},\, q_{xt}), \quad \text{logit}(q_{xt}) = \eta_{xt}
$$

Used by: CBD, M6, M7, M8.

## Age Functions

Age modulating functions $\beta_x^{(i)}$ can be:

| Class | Formula | Models |
|---|---|---|
| `NonParametricAgeFun` | Free parameters (estimated) | LC, RH |
| `ConstantAgeFun` | $f(x) = 1$ | CBD (κ¹), APC, M6 |
| `LinearAgeFun` | $f(x) = x - \bar{x}$ | CBD (κ²) |
| `QuadraticAgeFun` | $f(x) = (x-\bar{x})^2 - \sigma^2_x$ | M7 |
| `CenteredCohortAgeFun(xc)` | $f(x) = x_c - x$ | M8 |

## Fitting

Two fitting paths are used:

**Path A — Parametric GLM** (CBD, APC, M6, M7, M8): When all $\beta_x^{(i)}$
are known functions, the model is a GLM. A sparse design matrix is built and
fitted with IRLS (via statsmodels or a hand-coded pseudoinverse IRLS for
rank-deficient designs like APC).

**Path B — Block-coordinate IRLS** (LC, RH): When $\beta_x$ are free
parameters, the bilinear structure requires iterative block updates:
1. SVD initialisation of $\hat{\beta}_x$, $\hat{\kappa}_t$
2. Newton steps cycling over $\alpha_x \to \kappa_t \to \beta_x \to \gamma_c$
3. Convergence on relative deviance change

## References

Villegas, A.M., Millossovich, P., & Kaishev, V.K. (2018). StMoMo: An R Package
for Stochastic Mortality Modelling. *Journal of Statistical Software*, 84(3), 1–38.
