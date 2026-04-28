# Identifiability Constraints

GAPC models are not identifiable without constraints because the predictor $\eta_{xt}$ is invariant under certain transformations of the parameters. pyStMoMo applies post-fit constraints that project parameters onto the identifiable subspace without changing fitted values.

## Lee-Carter

$$\sum_x \beta_x = 1, \quad \frac{1}{T}\sum_t \kappa_t = 0 \text{ (mean absorbed into } \alpha_x)$$

## APC

$$\frac{1}{T}\sum_t \kappa_t = 0, \quad \sum_c \gamma_c = 0, \quad \sum_c c \cdot \gamma_c = 0$$

The linear trend in $\gamma_c$ is redistributed into $\kappa_t$ and $\alpha_x$, then zero-mean is re-enforced.

## CBD Family (M6, M7, M8)

$$\sum_c \gamma_c = 0$$

M7 also removes the linear and quadratic trend from $\gamma_c$.

## Renshaw-Haberman

Combines the LC constraint (sum($\beta_x$)=1, mean($\kappa_t$)=0) with zero-mean cohort.
