# External Forecasters

pyStMoMo's built-in period-index forecasters (MRWD and ARIMA) can be
replaced by any external model — LSTM, Transformer, Prophet, XGBoost,
Gaussian Process, etc. — by wrapping it in an
[`ExternalKtForecaster`][pystmomo.forecast.external.ExternalKtForecaster].

The wrapper plugs seamlessly into `forecast()` and `simulate()` via
the `kt_method` and `gc_method` parameters, while the GAPC rate
reconstruction pipeline remains unchanged.

---

## Interface required

Your external model must be expressible as two callables:

```python
def forecast_fn(h: int, *, level: float = 0.95):
    """
    Returns one of:
      - np.ndarray of shape (N, h)              — point forecast only
      - tuple (mean, lower, upper) each (N, h)  — with confidence bands
        (pass None for lower/upper to omit intervals)
    """
    ...

def simulate_fn(h: int, nsim: int, *, rng: np.random.Generator):
    """Returns np.ndarray of shape (N, h, nsim)."""
    ...
```

`N` = number of period indexes in the model (`fit.model.N`).

| Model | N | Has cohort (gc) |
|-------|---|-----------------|
| LC    | 1 | No  |
| CBD   | 2 | No  |
| APC   | 1 | Yes |
| RH    | 1 | Yes |
| M6    | 2 | Yes |
| M7    | 3 | Yes |
| M8    | 2 | Yes |

---

## Example — LC with a custom random walk

```python
import numpy as np
import pystmomo as ps
from pystmomo import ExternalKtForecaster

data = ps.load_ew_male()
fit  = ps.lc().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)

kt    = fit.kt[0]             # shape (51,) — the fitted period index
slope = np.diff(kt).mean()
sigma = np.diff(kt).std()

def forecast_fn(h, *, level=0.95):
    from scipy.stats import norm
    z    = norm.ppf((1 + level) / 2)
    mean = (kt[-1] + slope * np.arange(1, h + 1)).reshape(1, h)
    se   = sigma * np.sqrt(np.arange(1, h + 1))
    return mean, mean - z * se, mean + z * se

def simulate_fn(h, nsim, *, rng):
    innov = rng.normal(slope, sigma, (h, nsim))
    paths = kt[-1] + np.cumsum(innov, axis=0)    # (h, nsim)
    return paths[np.newaxis, :, :]                # (1, h, nsim)

ext = ExternalKtForecaster(forecast_fn, simulate_fn)
fc  = ps.forecast(fit, h=20, kt_method=ext)
sim = ps.simulate(fit, nsim=1000, h=20, kt_method=ext, seed=42)
```

---

## Example — CBD with independent models per index (N=2)

CBD has two period indexes: κ_t^(1) (level) and κ_t^(2) (slope-in-age).

```python
fit = ps.cbd().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)

kt     = fit.kt                               # shape (2, 51)
slopes = np.diff(kt, axis=1).mean(axis=1)    # (2,)
sigmas = np.diff(kt, axis=1).std(axis=1)     # (2,)

def forecast_fn(h, *, level=0.95):
    from scipy.stats import norm
    z    = norm.ppf((1 + level) / 2)
    mean = kt[:, -1:] + slopes[:, None] * np.arange(1, h + 1)   # (2, h)
    se   = sigmas[:, None] * np.sqrt(np.arange(1, h + 1))
    return mean, mean - z * se, mean + z * se

def simulate_fn(h, nsim, *, rng):
    paths = np.zeros((2, h, nsim))
    for i in range(2):
        innov      = rng.normal(slopes[i], sigmas[i], (h, nsim))
        paths[i]   = kt[i, -1] + np.cumsum(innov, axis=0)
    return paths

ext = ExternalKtForecaster(forecast_fn, simulate_fn)
fc  = ps.forecast(fit, h=20, kt_method=ext)
```

---

## Example — APC with separate kt and gc models

APC has one period index **and** a cohort effect.  Pass one
`ExternalKtForecaster` for each via `kt_method` and `gc_method`.

```python
fit = ps.apc().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)

# ── period index (N=1) ─────────────────────────────────────────────
kt      = fit.kt[0]
slope_k = np.diff(kt).mean();  sigma_k = np.diff(kt).std()

def kt_simulate(h, nsim, *, rng):
    innov = rng.normal(slope_k, sigma_k, (h, nsim))
    return (kt[-1] + np.cumsum(innov, axis=0))[np.newaxis]   # (1,h,nsim)

def kt_forecast(h, *, level=0.95):
    mean = (kt[-1] + slope_k * np.arange(1, h+1)).reshape(1, h)
    return mean, mean, mean

# ── cohort index ───────────────────────────────────────────────────
gc      = fit.gc[np.isfinite(fit.gc)]
slope_g = np.diff(gc).mean();  sigma_g = np.diff(gc).std()

def gc_simulate(h, nsim, *, rng):
    innov = rng.normal(slope_g, sigma_g, (h, nsim))
    return (gc[-1] + np.cumsum(innov, axis=0))[np.newaxis]   # (1,h,nsim)

def gc_forecast(h, *, level=0.95):
    mean = (gc[-1] + slope_g * np.arange(1, h+1)).reshape(1, h)
    return mean, mean, mean

fc = ps.forecast(fit, h=20,
                 kt_method=ExternalKtForecaster(kt_forecast, kt_simulate),
                 gc_method=ExternalKtForecaster(gc_forecast, gc_simulate))
```

!!! note "Number of new cohorts ≠ h"
    The gc forecaster will be called for **new cohorts only** (those not yet
    observed), not for all `h` future years.  The number of new cohorts is
    typically `h + len(ages) - 1`, not `h`.  Your `simulate_fn` receives the
    correct `h_new` automatically.

---

## Example — LSTM with PyTorch (LC)

```python
import torch
import torch.nn as nn
import pystmomo as ps
from pystmomo import ExternalKtForecaster

data = ps.load_ew_male()
fit  = ps.lc().fit(data.deaths, data.exposures,
                   ages=data.ages, years=data.years)
kt   = fit.kt[0]   # (51,)

# ── Normalise ──────────────────────────────────────────────────────
mu_k, sd_k = kt.mean(), kt.std()
kt_n = (kt - mu_k) / sd_k

# ── Build sliding-window dataset ───────────────────────────────────
lookback = 10
X = torch.FloatTensor(
    [kt_n[i:i+lookback] for i in range(len(kt_n) - lookback)]
).unsqueeze(-1)                         # (n, lookback, 1)
y = torch.FloatTensor(kt_n[lookback:]).unsqueeze(-1)   # (n, 1)

# ── Define and train ───────────────────────────────────────────────
class KtLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])

model = KtLSTM()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(500):
    loss = nn.MSELoss()(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()

# ── Wrap as ExternalKtForecaster ───────────────────────────────────
def lstm_forecast(h, *, level=0.95):
    model.eval()
    window = torch.FloatTensor(kt_n[-lookback:]).reshape(1, lookback, 1)
    preds  = []
    with torch.no_grad():
        for _ in range(h):
            v = model(window).item()
            preds.append(v)
            window = torch.cat([window[:, 1:], torch.tensor([[[v]]])], dim=1)
    mean = (np.array(preds) * sd_k + mu_k).reshape(1, h)
    return mean, mean, mean   # no CI — provide simulate_fn for uncertainty

def lstm_simulate(h, nsim, *, rng):
    # Simple approach: add Gaussian noise scaled to training residuals
    resid_std = float(nn.MSELoss()(model(X), y).detach().sqrt()) * sd_k
    mean, _, _ = lstm_forecast(h)
    noise = rng.normal(0, resid_std, (1, h, nsim)) * np.sqrt(np.arange(1, h+1))
    return mean[:, :, np.newaxis] + noise

ext = ExternalKtForecaster(lstm_forecast, lstm_simulate)
fc  = ps.forecast(fit, h=20, kt_method=ext)
sim = ps.simulate(fit, nsim=500, h=20, kt_method=ext, seed=42)
```

---

## Only point forecast, no simulation

If you only supply `forecast_fn` (no `simulate_fn`), `simulate()` will
repeat the point forecast across all paths — fully deterministic,
no parameter uncertainty:

```python
ext = ExternalKtForecaster(forecast_fn)   # no simulate_fn
fc  = ps.forecast(fit, h=20, kt_method=ext)    # works fine
sim = ps.simulate(fit, nsim=500, h=20, kt_method=ext)   # all nsim paths identical
```

!!! warning
    A deterministic simulate defeats the purpose of Monte Carlo — supply
    `simulate_fn` whenever uncertainty quantification matters.
