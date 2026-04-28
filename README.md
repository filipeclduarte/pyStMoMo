# pyStMoMo

**Stochastic Mortality Modelling in Python** — a faithful Python port of the
[StMoMo](https://github.com/amvillegas/StMoMo) R library by Villegas, Millossovich
& Kaishev.

## Overview

`pyStMoMo` implements a framework for fitting, forecasting, simulating and
validating **Generalised Age-Period-Cohort (GAPC)** stochastic mortality models,
including:

| Model | Reference |
|-------|-----------|
| Lee-Carter (LC) | Lee & Carter (1992) |
| Cairns-Blake-Dowd (CBD) | Cairns et al. (2006) |
| Age-Period-Cohort (APC) | Currie (2006) |
| Renshaw-Haberman (RH) | Renshaw & Haberman (2006) |
| M6, M7, M8 | Cairns et al. (2009) |
| Custom GAPC | — |

## Quick Start

```python
import pystmomo as ps

data = ps.load_ew_male()
fit  = ps.lc().fit(data.deaths, data.exposures, ages=data.ages, years=data.years)
fc   = ps.forecast(fit, h=50)
sim  = ps.simulate(fit, nsim=5000, h=50, seed=42)

ps.plot_parameters(fit)
ps.plot_fan(sim, age=65)
```

## Installation

```bash
pip install pystmomo
```

### From source

```bash
git clone https://github.com/filipeclduarte/pyStMoMo
cd pyStMoMo
pip install -e ".[dev]"
```

## Documentation

Full documentation at <https://filipeclduarte.github.io/pyStMoMo>.

## References

- Villegas, A.M., Millossovich, P., & Kaishev, V.K. (2018). StMoMo: An R Package
  for Stochastic Mortality Modelling. *Journal of Statistical Software*, 84(3).
- Lee, R.D., & Carter, L.R. (1992). Modeling and Forecasting U.S. Mortality.
  *JASA*, 87(419), 659–671.
- Cairns, A.J.G., Blake, D., Dowd, K., Coughlan, G.D., Epstein, D., Ong, A., &
  Balevich, I. (2009). A Quantitative Comparison of Stochastic Mortality Models
  Using Data From England and Wales and the United States. *NAAJ*, 13(1), 1–35.

## License

GPL-2.0-or-later — see [LICENSE](LICENSE).
