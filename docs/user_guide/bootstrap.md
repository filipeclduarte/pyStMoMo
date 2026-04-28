# Bootstrap Uncertainty

## Semiparametric Bootstrap

Resamples death counts from their assumed distribution (Poisson or Binomial) with mean equal to the fitted deaths, then refits the model on each sample:

```python
import pystmomo as ps

data = ps.load_ew_male()
fit  = ps.lc().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)

boot = ps.semiparametric_bootstrap(fit, nboot=500, seed=0)
print(boot)
# BootStMoMo(method='semiparametric', nboot=500, n_fits_ok=500)
```

## Residual Bootstrap

Resamples deviance residuals and inverts them to obtain bootstrap death counts (Renshaw & Haberman 2008):

```python
boot_r = ps.residual_bootstrap(fit, nboot=500, seed=0)
```

## Confidence Intervals

```python
lo, hi = boot.parameter_ci("kt", level=0.95)
lo_bx, hi_bx = boot.parameter_ci("bx", level=0.95)
```

Available parameters: `"ax"`, `"bx"`, `"kt"`, `"b0x"`, `"gc"`.

## Parallel Bootstrap

```bash
pip install "pystmomo[parallel]"
```

```python
boot = ps.semiparametric_bootstrap(fit, nboot=500, n_jobs=-1, seed=0)
```

## Accessing Raw Replicate Fits

All fitted `FitStMoMo` objects are stored in `boot.fits`:

```python
deviances = [f.deviance for f in boot.fits]
```
