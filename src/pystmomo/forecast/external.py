"""Adapter for plugging external models into pyStMoMo's forecast/simulate pipeline.

Any model that can produce a point forecast or sample paths for κ_t (or γ_c)
can be wrapped in :class:`ExternalKtForecaster` and passed directly to
:func:`~pystmomo.forecast.forecast` or :func:`~pystmomo.simulate.simulate`
via their ``kt_method`` / ``gc_method`` parameters.

Example
-------
>>> import numpy as np
>>> from pystmomo.forecast.external import ExternalKtForecaster
>>> import pystmomo as ps
>>>
>>> data = ps.load_ew_male()
>>> fit  = ps.lc().fit(data.deaths, data.exposures,
...                    ages=data.ages, years=data.years)
>>>
>>> # Fit any external model on fit.kt[0] (the LC period index)
>>> # Here we use a trivial linear extrapolation for illustration
>>> kt = fit.kt[0]
>>> slope = np.diff(kt).mean()
>>>
>>> def my_forecast_fn(h, *, level=0.95):
...     mean = kt[-1] + slope * np.arange(1, h + 1)
...     mean = mean.reshape(1, h)          # shape must be (N, h)
...     z    = 1.96 * np.std(np.diff(kt)) * np.sqrt(np.arange(1, h + 1))
...     lo   = mean - z
...     hi   = mean + z
...     return mean, lo, hi
>>>
>>> def my_simulate_fn(h, nsim, *, rng):
...     mean = kt[-1] + slope * np.arange(1, h + 1)
...     noise = rng.standard_normal((nsim, h)) * np.std(np.diff(kt))
...     paths = mean + np.cumsum(noise, axis=1)   # (nsim, h)
...     return paths.T[np.newaxis, :, :]           # (1, h, nsim)
>>>
>>> ext = ExternalKtForecaster(my_forecast_fn, my_simulate_fn)
>>> fc  = ps.forecast(fit, h=20, kt_method=ext)
>>> sim = ps.simulate(fit, nsim=500, h=20, kt_method=ext, seed=0)
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class ExternalKtForecaster:
    """Wraps external forecast/simulate callables into pyStMoMo's forecaster interface.

    Pass an instance of this class as ``kt_method`` (or ``gc_method``) in
    :func:`~pystmomo.forecast.forecast` or :func:`~pystmomo.simulate.simulate`.

    Parameters
    ----------
    forecast_fn:
        Callable with signature ``(h, *, level=0.95) -> result`` where *result*
        is one of:

        * ``np.ndarray`` of shape *(N, h)* — point forecast only (no intervals)
        * ``tuple (mean, lower, upper)`` — each array of shape *(N, h)*.
          Pass ``None`` for *lower*/*upper* to suppress intervals.

    simulate_fn:
        Optional callable with signature ``(h, nsim, *, rng) -> np.ndarray``
        returning shape *(N, h, nsim)*.

        If omitted, :meth:`simulate` falls back to repeating the point
        forecast mean across all ``nsim`` paths (no parameter uncertainty).

    Notes
    -----
    *N* is the number of period (or cohort) indexes in the model.  For
    Lee-Carter *N = 1*; for CBD *N = 2*; for APC *N = 1* (κ) plus a
    separate gc series.

    The arrays returned by *forecast_fn* and *simulate_fn* must match the
    *N* of the fitted model — the library does not validate this at
    construction time, only when the arrays are used.
    """

    def __init__(
        self,
        forecast_fn: Callable,
        simulate_fn: Callable | None = None,
    ) -> None:
        self._forecast_fn = forecast_fn
        self._simulate_fn = simulate_fn

    # ------------------------------------------------------------------ #
    # Interface expected by forecast() and simulate()                     #
    # ------------------------------------------------------------------ #

    def forecast(
        self,
        h: int,
        *,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Call the wrapped forecast function and normalise its output.

        Returns
        -------
        mean, lower, upper:
            Arrays of shape *(N, h)*.
        """
        result = self._forecast_fn(h, level=level)

        if isinstance(result, np.ndarray):
            mean = np.atleast_2d(np.asarray(result, dtype=float))
            return mean, mean.copy(), mean.copy()

        mean, lower, upper = result
        mean = np.atleast_2d(np.asarray(mean, dtype=float))
        lower = mean.copy() if lower is None else np.atleast_2d(np.asarray(lower, dtype=float))
        upper = mean.copy() if upper is None else np.atleast_2d(np.asarray(upper, dtype=float))
        return mean, lower, upper

    def simulate(
        self,
        h: int,
        nsim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Call the wrapped simulate function and return shape *(N, h, nsim)*.

        If no *simulate_fn* was provided, returns the forecast mean broadcast
        across all paths (deterministic — no parameter uncertainty).
        """
        if self._simulate_fn is not None:
            result = np.asarray(self._simulate_fn(h, nsim, rng=rng), dtype=float)
            if result.ndim == 2:
                result = result[np.newaxis, :, :]   # (h, nsim) → (1, h, nsim)
            return result

        # Fallback: no simulate_fn → repeat mean
        mean, _, _ = self.forecast(h)
        N = mean.shape[0]
        return np.broadcast_to(mean[:, :, np.newaxis], (N, h, nsim)).copy()

    def __repr__(self) -> str:
        has_sim = self._simulate_fn is not None
        return f"ExternalKtForecaster(simulate={'yes' if has_sim else 'no (deterministic fallback)'})"
