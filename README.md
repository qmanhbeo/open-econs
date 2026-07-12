# open-econs

[![PyPI version](https://img.shields.io/pypi/v/open-econs?color=blue)](https://pypi.org/project/open-econs/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-econs)](https://pypi.org/project/open-econs/)
[![CI](https://github.com/qmanhbeo/open-econs/actions/workflows/ci.yml/badge.svg)](https://github.com/qmanhbeo/open-econs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/qmanhbeo/open-econs)](https://github.com/qmanhbeo/open-econs/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/open-econs)](https://pypi.org/project/open-econs/)

**Python econometrics with Stata/R parity.**

open-econs brings familiar empirical economics methods from Stata and R into a
unified Python workflow. Every estimator uses a consistent API, with numerical
validation against established reference implementations.

- 142+ Stata-parity tests verify coefficients and standard errors match
  reference implementations at machine precision.
- One consistent interface across all estimators: `.summary()`, `.tidy()`,
  `.vcov()`, `.predict()`, `.export()`, `.to_latex()`.
- Immutable, named pandas outputs — no raw arrays crossing the public API.

## Installation

```bash
pip install open-econs                           # core: OLS, Oaxaca, FE, IV, Logit, Probit
pip install open-econs[plot]                      # + matplotlib for .plot()
pip install open-econs[nls]                       # + sympy for nls() (nonlinear least squares)
pip install open-econs[dev,lint]                  # + development & linting tools
pip install git+https://github.com/qmanhbeo/open-econs.git    # latest dev
```

Requires Python ≥ 3.10.

## Why open-econs?

Empirical researchers often use Python for data cleaning, Stata for estimation,
R for specialized methods, and LaTeX or custom scripts for output — a fragmented
workflow that hurts reproducibility.

open-econs keeps everything in one Python environment:

- **Familiar estimators** — OLS, fixed effects, IV/2SLS, logit, probit,
  difference-in-differences, event studies, RDD, panel models, matching
- **Consistent API** — the same `.summary()` / `.tidy()` / `.vcov()` /
  `.predict()` interface across every method
- **Named pandas outputs** — coefficients, standard errors, and diagnostics
  return as `pd.Series` or `pd.DataFrame` with clear labels
- **Immutable results** — no accidental mutation after estimation
- **Reproducible exports** — JSON, CSV, LaTeX, HTML from any result

```python
import open_econs as oe
import pandas as pd

df = pd.DataFrame({
    "income":    [30, 45, 55, 70, 85, 40, 60, 95],
    "education": [10, 12, 14, 16, 18, 11, 15, 20],
    "age":       [25, 30, 35, 40, 45, 28, 38, 50],
    "female":    [0,  0,  0,  0,  1,  1,  1,  1],
    "province":  ["A","A","B","B","C","C","A","B"],
})

r = oe.ols("income ~ education + age", data=df, cluster="province")
print(r.tidy())
```

## For Stata and R users

If you know the Stata or R command, you already know the open-econs equivalent.

### Stata

| Stata                | open-econs                  |
| -------------------- | --------------------------- |
| `regress`            | `oe.ols()`                  |
| `xtreg, fe`          | `oe.fe()`                   |
| `ivregress 2sls`     | `oe.iv()`                   |
| `logit` / `probit`   | `oe.logit()` / `oe.probit()`|
| `mlogit`             | `oe.mlogit()` (multinomial logit) |
| `oaxaca`             | `oe.oaxaca()`               |
| `xtabond2`           | `oe.abond()`                |
| `csdid`              | `oe.staggered_did()`        |
| `rdrobust`           | `oe.rdd()`                  |
| `teffects psmatch`   | `oe.psm()`                  |

### R

| R                    | open-econs                  |
| -------------------- | --------------------------- |
| `fixest` / `plm`     | `oe.fe()` / `oe.PanelContext()` |
| `AER::ivreg`         | `oe.iv()`                   |
| `did`                | `oe.staggered_did()`        |
| `MatchIt`            | `oe.psm()`                  |

See [docs/migrating_from_stata.md](docs/migrating_from_stata.md) for a
detailed migration guide.

## Quick Start

```python
import open_econs as oe
import pandas as pd

df = pd.DataFrame({
    "income":    [30, 45, 55, 70, 85, 40, 60, 95],
    "education": [10, 12, 14, 16, 18, 11, 15, 20],
    "age":       [25, 30, 35, 40, 45, 28, 38, 50],
    "female":    [0,  0,  0,  0,  1,  1,  1,  1],
    "province":  ["A","A","B","B","C","C","A","B"],
})

# --- OLS with cluster-robust SEs ---
r = oe.ols("income ~ education + age", data=df, cluster="province")
r.coefficients          # pd.Series with named index
r.tidy()                # coefficient table as DataFrame
r.predict(df.head(2))   # out-of-sample predictions
print(r.summary())      # printable summary

# --- Logit / Probit ---
r_logit = oe.logit("female ~ education + age", data=df)
r_logit.tidy()          # coef, z, P>|z| table
r_logit.margins()       # average marginal effects
r_logit.predict(proba=False)

# --- Fixed effects ---
r_fe = oe.fe("income ~ education + age", data=df, entity="province")
r_fe.tidy()

# --- IV / 2SLS ---
r_iv = oe.iv("income ~ education | age", data=df)
r_iv.tidy()             # 2SLS coefficients
r_iv.first_stage()      # first-stage F-stat

# --- Oaxaca-Blinder decomposition ---
d = oe.oaxaca("income ~ education + age + female", data=df, by="female")
d.explained             # covariate-driven gap
d.unexplained           # coefficient-driven gap
d.total_gap             # female mean - male mean

# --- VIF diagnostics ---
ctx = oe.Context(df)
ctx.vif("income ~ education + age")

# --- Context remembers the dataset ---
ctx.ols("income ~ education + age")
ctx.logit("female ~ education + age")
ctx.probit("female ~ education + age")

# --- Immutability ---
r.f_statistic = 0.0  # AttributeError: OLSResult is immutable
```

## Supported Methods

| Function | Description |
|---|---|
| `ols()` / `reg()` | OLS with HC1/robust/clustered SEs, multi-way clustering, Newey-West HAC, WLS |
| `fe()` | Fixed effects (one-way entity, two-way entity + time) |
| `iv()` | Instrumental variables / 2SLS with first-stage F-stat |
| `logit()` | Binary logit with `.margins()`, `.predict()` |
| `probit()` | Binary probit (same API as logit) |
| `mlogit()` | Multinomial logit (MNLogit) with per-outcome `.margins()` (dict), `.predict()`, robust/cluster SEs |
| `oaxaca()` | Oaxaca-Blinder decomposition (two-fold, three-fold; multiple reference types) |
| `nls()` | Nonlinear least squares (Gauss-Newton via scipy, analytic Jacobian) |
| `abond()` | Arellano-Bond dynamic panel GMM (one/two-step, Windmeijer SEs, collapsed instruments) |
| `staggered_did()` | Callaway-Sant'Anna (2021) staggered DiD (doubly-robust `dripw` or `reg`) |
| `rdd()` | Sharp / fuzzy regression discontinuity (local linear, triangular kernel) |
| `did()` / `event_study()` | Two-period DiD, event-study with pre-trend diagnostics |
| `psm()` | Propensity score matching (1:1 nearest-neighbor with replacement, AI 2012 SEs) |
| `cem()` | Coarsened exact matching (auto or explicit cutpoints, multiple binning methods) |
| `balance()` | Covariate balance diagnostics (SMD, variance ratio, weighted t-tests) |
| `rosenbaum_bounds()` | Sensitivity analysis for matched pairs |
| `synth()` | Synthetic control (Abadie-Diamond-Hainmueller) core point estimator: nested predictor-weight (V) + donor-weight (W) optimization, gap path, R/Stata parity |
| `placebo_space()` | Synthetic control placebo-in-space permutation inference (ADH): re-fits `synth()` once per donor; `post/pre` MSPE ratio per placebo; permutation p-value; optional `exclude_pre_mspe_multiple` (space-only) |
| `placebo_time()` | Synthetic control placebo-in-time permutation inference (ADH): re-fits `synth()` once per candidate pre-treatment date; `post/pre` MSPE ratio per candidate; permutation p-value |
| `PanelContext(...)` | Pooled/FE/RE/FD/Driscoll-Kraay/Hausman/ABond with remembered entity/time |
| `Context(...)` | Dataset-scoped workflow with access to all above estimators |

### Synthetic control placebo inference (ADH)

`placebo_space()` / `placebo_time()` build on a fitted `synth()` result and
reuse its validated solver — no estimator logic is duplicated. The caller only
supplies the panel `data`; the result carries the rest of the fit config.

```python
from open_econs.models.causal.synth import synth
from open_econs.models.causal.placebo import placebo_space, placebo_time

r = synth(df, "y", treated_unit="t", donor_pool=donors,
          entity="unit", time="time", pre_period=1994, post_period=1995)

ps = placebo_space(r, df)                 # permutation p-value across donors
print(ps.p_value, ps.ratios)              # fraction of donor ratios >= treated's

pt = placebo_time(r, df)                  # permutation p-value across candidate dates
print(pt.p_value)

# Opt-in, space-only pre-fit exclusion (default None: never applied silently):
ps_filtered = placebo_space(r, df, exclude_pre_mspe_multiple=10.0)
```

`SynthResult` now also stores the original `predictors` argument (the one
fit-config field it previously lacked) so the delegate can reconstruct each
placebo call. `placebo_time()` does **not** accept `exclude_pre_mspe_multiple` —
in the in-time loop "pre-MSPE" is the treated unit's own fit against itself at a
different cutoff, so the space-style exclusion concept does not carry over.

## Result API

Every estimator returns an object with:

| Method | Returns |
|---|---|
| `.summary()` | Printable string (also `__repr__`) |
| `.tidy()` | `pd.DataFrame` — coefficient or effect table |
| `.vcov()` | `pd.DataFrame` — variance-covariance matrix |
| `.predict(newdata)` | `pd.Series` — only on regression models |
| `.export(path)` | JSON / CSV serialization |
| `.plot()` | Residual diagnostics plot (requires `pip install open-econs[plot]`) |
| `.to_dict()` | `dict` — full result metadata |
| `.to_latex()` | LaTeX table string |
| `.to_html()` | HTML table string |

## Design Philosophy

### Reproducibility first
Immutable results prevent accidental mutation. All numeric artifacts are named
`pd.Series` or `pd.DataFrame` with variable-name indices. Every result can be
exported to JSON, CSV, LaTeX, or HTML.

### Familiar econometrics
Methods and defaults follow established practice from Stata and R. If you know
`regress`, `xtreg`, or `ivregress`, you already know how to use the open-econs
equivalent. Default standard errors (HC2) match modern Stata conventions.

### One Python workflow
Data processing, estimation, diagnostics, sensitivity analysis, and output
generation live in the same environment. No more stitching together separate
tools.

## Methodology Documentation

Every estimator includes a methodology specification covering:

- Mathematical formulation
- Assumptions
- Estimation procedure
- Inference (all covariance estimators with exact formulas)
- Implementation details
- Stata/R equivalents
- Academic references

See [methodology/](methodology/) for the full registry.

## Comparison

| Library | Strength |
|---|---|
| statsmodels | General statistical models |
| linearmodels | Panel and IV estimation |
| scikit-learn | Machine learning |
| **open-econs** | Empirical economics workflows with Stata/R parity |

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and development milestones.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## License

MIT
