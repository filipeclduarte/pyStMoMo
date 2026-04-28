"""Independent ARIMA wrapper for forecasting period/cohort indexes."""
from __future__ import annotations

import warnings

import numpy as np


class IndependentArima:
    """One ARIMA model per row of kt (or per scalar gc).

    Parameters
    ----------
    order:
        ARIMA order (p, d, q).
    include_constant:
        Whether to include a constant/trend term.
    models:
        List of fitted statsmodels ARIMA results, one per series.
    last_values:
        Shape (N,) — last observed values for each series.
    """

    def __init__(
        self,
        order: tuple[int, int, int],
        include_constant: bool,
        models: list,
        last_values: np.ndarray,
    ) -> None:
        self.order = order
        self.include_constant = include_constant
        self.models = models
        self.last_values = last_values

    @classmethod
    def fit(
        cls,
        kt: np.ndarray,
        order: tuple[int, int, int] = (1, 1, 0),
        include_constant: bool = True,
    ) -> "IndependentArima":
        """Fit independent ARIMA models to each row of kt.

        Parameters
        ----------
        kt:
            Shape (N, n_years) or (n_years,) for a single series.
        order:
            ARIMA (p, d, q).
        include_constant:
            Include constant/trend term.

        Returns
        -------
        IndependentArima
        """
        from statsmodels.tsa.arima.model import ARIMA

        kt = np.atleast_2d(kt)
        N = kt.shape[0]
        _, d, _ = order
        # statsmodels semantics: when d >= 1, a constant in levels corresponds
        # to a linear trend term ('t'), not 'c'.  'c' is only valid for d=0.
        if include_constant:
            trend = "t" if d >= 1 else "c"
        else:
            trend = "n"

        fitted_models = []
        for i in range(N):
            series = kt[i, :]
            # Drop NaN/Inf values that might appear from sparse cohorts
            series = series[np.isfinite(series)]
            if len(series) < max(3, sum(order) + 2):
                raise ValueError(
                    f"Series {i} has too few valid observations ({len(series)}) "
                    f"to fit ARIMA{order}."
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(series, order=order, trend=trend)
                result = model.fit()
            fitted_models.append(result)

        last_values = kt[:, -1]
        return cls(
            order=order,
            include_constant=include_constant,
            models=fitted_models,
            last_values=last_values,
        )

    def forecast(
        self,
        h: int,
        level: float = 0.95,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point forecast and confidence intervals for h steps ahead.

        Parameters
        ----------
        h:
            Forecast horizon.
        level:
            Confidence level (e.g. 0.95).

        Returns
        -------
        mean, lower, upper
            Each shape (N, h).
        """
        N = len(self.models)
        mean = np.zeros((N, h))
        lower = np.zeros((N, h))
        upper = np.zeros((N, h))
        alpha = 1.0 - level

        for i, result in enumerate(self.models):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = result.get_forecast(steps=h)
                mean[i, :] = fc.predicted_mean
                ci = fc.conf_int(alpha=alpha)
                # conf_int() may return a DataFrame or a numpy array
                # depending on statsmodels version — handle both
                if hasattr(ci, "iloc"):
                    lower[i, :] = ci.iloc[:, 0].values
                    upper[i, :] = ci.iloc[:, 1].values
                else:
                    ci = np.asarray(ci)
                    lower[i, :] = ci[:, 0]
                    upper[i, :] = ci[:, 1]

        return mean, lower, upper

    def simulate(
        self,
        h: int,
        nsim: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Simulate future trajectories from fitted ARIMA models.

        Parameters
        ----------
        h:
            Forecast horizon.
        nsim:
            Number of simulations.
        rng:
            NumPy random generator (used to set seed for statsmodels).

        Returns
        -------
        np.ndarray
            Shape (N, h, nsim).
        """
        N = len(self.models)
        paths = np.zeros((N, h, nsim))
        seed = int(rng.integers(0, 2**31))

        for i, result in enumerate(self.models):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sims = result.simulate(
                    nsimulations=h,
                    repetitions=nsim,
                    anchor="end",
                    random_state=seed + i,
                )
                sims = np.asarray(sims).squeeze()   # collapse extra dims
                if sims.ndim == 1:
                    sims = sims[:, np.newaxis].repeat(nsim, axis=1)
                if sims.shape[0] != h:
                    sims = sims.T
                paths[i, :, :] = sims

        return paths
