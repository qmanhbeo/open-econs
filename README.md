# open-econs

[![PyPI version](https://img.shields.io/pypi/v/open-econs?color=blue)](https://pypi.org/project/open-econs/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-econs)](https://pypi.org/project/open-econs/)
[![CI](https://github.com/qmanhbeo/open-econs/actions/workflows/ci.yml/badge.svg)](https://github.com/qmanhbeo/open-econs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/qmanhbeo/open-econs)](https://github.com/qmanhbeo/open-econs/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/open-econs)](https://pypi.org/project/open-econs/)

**Python econometrics with Stata/R parity.**

Empirical researchers often use Python for data cleaning, Stata for estimation,
R for specialized methods, and LaTeX or custom scripts for output — a fragmented
workflow that hurts reproducibility.

open-econs brings familiar empirical economics methods from Stata and R into a
unified Python workflow. Every estimator uses a consistent API, with numerical
validation against established reference implementations.

- **Familiar estimators** — OLS, fixed effects, IV/2SLS, logit, probit,
  difference-in-differences, event studies, RDD, panel models, matching,...
- **Tested against Stata/R** 217 Stata- and R-parity tests (208 vs Stata, 9 vs R) verify coefficients and
  standard errors against reference implementations — at tolerances from `1e-6`
  (typical) up to `1e-15` for the tightest exact-equivalence cases (IV /
  Arellano-Bond / synthetic control), mostly `1e-6`–`1e-10`. They run in CI on
  every release, so a numerical-equivalence regression fails the build before it
  ships.
- **One consistent interface**  across all estimators: `.summary()`, `.tidy()`,
  `.vcov()`, `.predict()`, `.export()`, `.to_latex()`.
- **Named pandas outputs** — coefficients, standard errors, and diagnostics
  return as `pd.Series` or `pd.DataFrame` with clear labels
- **Immutable results** — no accidental mutation after estimation
- **Reproducible exports** — JSON, CSV, LaTeX, HTML from any result

## Installation & Quick Start

Requires Python ≥ 3.10.

```bash
pip install open-econs                           # core: OLS, Oaxaca, FE, IV, Logit, Probit
pip install open-econs[plot]                      # + matplotlib for .plot()
pip install open-econs[nls]                       # + sympy for nls() (nonlinear least squares)
pip install open-econs[dev,lint]                  # + development & linting tools
pip install git+https://github.com/qmanhbeo/open-econs.git    # latest dev
```

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

#     Variable       Coef    Std Err          t     P>|t|      0.025      0.975
# 0  Intercept -32.820513   0.730054 -44.956283  0.000000 -34.251392 -31.389633
# 1  education   8.974359  12.843154   0.698766  0.484698 -16.197761  34.146479
# 2        age  -1.025641   5.156134  -0.198917  0.842328 -11.131478   9.080196

print(r.summary())

#                             OLS Regression Results                            
# ======================================================================
# Dep. Variable:               income
# No. Observations:            8
# Df Residuals:                5
# Df Model:                    2
# Covariance Type:             cluster(province)
# R-squared:                   0.991941
# Adj. R-squared:              0.988718
# Condition No.:               3.31e+02
# F-statistic (cluster(province)):     645774.0223
# Prob (F-statistic):          1.548527e-06
# Log-Likelihood:              -16.392
# AIC:                         38.78
# BIC:                         39.02
# ======================================================================
#  Variable       Coef   Std Err          t    P>|t|      0.025      0.975
# Intercept -32.820513  0.730054 -44.956283 0.000000 -34.251392 -31.389633
# education   8.974359 12.843154   0.698766 0.484698 -16.197761  34.146479
#       age  -1.025641  5.156134  -0.198917 0.842328 -11.131478   9.080196
# ======================================================================
# Diagnostics:
# Jarque-Bera (chi2=0.498, p=0.7796)
# Breusch-Pagan (LM=5.296, p=0.0708)
# Durbin-Watson:                  2.1311
# Ramsey RESET (F=0.140, p=0.8731)
# ======================================================================
```



## Why open-econs?



open-econs keeps everything in one Python environment:





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
| `csdid`              | `oe.did_cs()`               |
| `rdrobust`           | `oe.rdd()`                  |
| `teffects psmatch`   | `oe.psm()`                  |

### R

| R                    | open-econs                  |
| -------------------- | --------------------------- |
| `fixest` / `plm`     | `oe.fe()` / `oe.PanelContext()` |
| `AER::ivreg`         | `oe.iv()`                   |
| `did`                | `oe.did_cs()`               |
| `MatchIt`            | `oe.psm()`                  |

See [Migrating from Stata](docs/migrating_from_stata.md) for a
detailed migration guide, or [Migrating from R](docs/migrating_from_r.md)
for the R equivalent.

Step-by-step porting walkthroughs for the causal estimators live in
[Tutorials](docs/tutorials/README.md):
- [RDD](docs/tutorials/rdd.md) (maps to `rdrobust` / `rddensity`)
- [PSM](docs/tutorials/psm.md) (maps to `MatchIt` / `teffects psmatch`)
- [Synthetic Control](docs/tutorials/synth_control.md) (maps to `Synth` / `synth`)

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

| Function | Description | Status |
|---|---|---|
| `ols()` / `reg()` | OLS with HC1/robust/clustered SEs, multi-way clustering, Newey-West HAC, WLS | Production |
| `fe()` | Fixed effects (one-way entity, two-way entity + time) | Production |
| `iv()` | Instrumental variables / 2SLS with first-stage F-stat | Production |
| `logit()` | Binary logit with `.margins()`, `.predict()` | Production |
| `probit()` | Binary probit (same API as logit) | Production |
| `mlogit()` | Multinomial logit (MNLogit) with per-outcome `.margins()` (dict), `.predict()`, robust/cluster SEs | Beta |
| `oaxaca()` | Oaxaca-Blinder decomposition (two-fold, three-fold; multiple reference types) | Production |
| `nls()` | Nonlinear least squares (Gauss-Newton via scipy, analytic Jacobian) | Beta |
| `gmm()` | General linear GMM framework (reuses `iv()` formula grammar; Hansen J size/power) | Beta |
| `abond()` | Arellano-Bond dynamic panel GMM (one/two-step, Windmeijer SEs, collapsed instruments) | Production |
| `did_cs()` | Callaway-Sant'Anna (2021) staggered DiD (doubly-robust `dripw` or `reg`) | Beta |
| `rdd()` | Sharp / fuzzy regression discontinuity (local linear, triangular kernel) | Production |
| `did()` / `event_study()` | Two-period DiD, event-study with pre-trend diagnostics | Production |
| `psm()` | Propensity score matching (1:1 nearest-neighbor with replacement, AI 2012 SEs) | Production |
| `cem()` | Coarsened exact matching (auto or explicit cutpoints, multiple binning methods) | Production |
| `balance()` | Covariate balance diagnostics (SMD, variance ratio, weighted t-tests) | Production |
| `rosenbaum_bounds()` | Sensitivity analysis for matched pairs | Beta |
| `synth()` | Synthetic control (Abadie-Diamond-Hainmueller) core point estimator: nested predictor-weight (V) + donor-weight (W) optimization, gap path, R/Stata parity | Production |
| `placebo_space()` | Synthetic control placebo-in-space permutation inference (ADH): re-fits `synth()` once per donor; `post/pre` MSPE ratio per placebo; permutation p-value; optional `exclude_pre_mspe_multiple` (space-only) | Beta |
| `placebo_time()` | Synthetic control placebo-in-time permutation inference (ADH): re-fits `synth()` once per candidate pre-treatment date; `post/pre` MSPE ratio per candidate; permutation p-value | Beta |
| `PanelContext(...)` | Pooled/FE/RE/FD/Driscoll-Kraay/Hausman/ABond with remembered entity/time | Production |
| `Context(...)` | Dataset-scoped workflow with access to all above estimators | Production |

*Status reflects validation maturity: **Production** = machine-precision parity
vs Stata/R on reference fixtures; **Beta** = validated but with narrower parity
coverage or known edge cases documented in `methodology/`. Every estimator
ships with parity tests before merge.*

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

See [Methodology](methodology/) for the full registry.

## Tutorials

Step-by-step walkthroughs for the core estimators:

- [OLS](docs/tutorials/ols.md)
- [Fixed Effects](docs/tutorials/fe.md)
- [Instrumental Variables](docs/tutorials/iv.md)
- [Difference-in-Differences](docs/tutorials/did.md)
- [RDD](docs/tutorials/rdd.md)
- [PSM](docs/tutorials/psm.md)
- [Synthetic Control](docs/tutorials/synth_control.md)

## Comparison

| Library | Strength |
|---|---|
| statsmodels | General statistical models |
| linearmodels | Panel and IV estimation |
| scikit-learn | Machine learning |
| **open-econs** | Empirical economics workflows with Stata/R parity |

## Roadmap

See [Roadmap](ROADMAP.md) for planned features and development milestones.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## License

MIT
