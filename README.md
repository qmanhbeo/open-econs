# open-econs

[![PyPI version](https://img.shields.io/pypi/v/open-econs?color=blue)](https://pypi.org/project/open-econs/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-econs)](https://pypi.org/project/open-econs/)

**The scikit-learn of empirical economics (or social sciences in general).**

A Python library that bridges the gap between traditional Stata/R econometrics
workflows and modern, production-grade Python systems.  Every estimator follows
the same interface — `summary`, `tidy`, `export` — so researchers and
AI agents never have to learn a new API.

> **Current version (v0.8.0):** 142 Stata‑parity tests across all estimators
> — 22 `.do` files with cached `.dta` fixtures, dual‑mode execution (live
> Stata or CI fallback). **All 8 ABOND flavors** (collapsed/non‑collapsed ×
> one/two‑step × robust/non‑robust) verified against `xtabond2` 3.7.2 at
> rtol=1e‑6. **Event‑study** now uses t‑distribution inference
> (`cov_kwds={"use_t": True}`), matching Stata's default. **Logit/probit
> margins** match Stata at machine precision. **Test‑suite caching**
> (`read_stata()` at module level) cuts full‑suite runtime from 235s → 94s
> (2.5× speedup). All staggered‑DID, Oaxaca, RDD, IV, panel (FE/RE/FD/DK),
> and HAC tests pass — Oaxaca at rtol=1e‑6, panel coefficients at
> rtol=1e‑6, staggered‑DID ATT at rtol=1e‑6 (SE gap deferred to CS2021
> doubly‑robust rewrite in v0.7).

## Why open-econs?

> Modern empirical research often uses Python for data engineering, Stata for estimation, LaTeX for tables, and custom scripts to glue everything together. open-econs lets you stay in one reproducible Python workflow without giving up familiar econometric methods.

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

# --- OLS with named coefficients and cluster-robust SEs ---
r = oe.ols("income ~ education + age", data=df, cluster="province")
print(r.coefficients)  # Clean pd.Series with named index 

# --- Two-fold Oaxaca-Blinder Decomposition ---
d = oe.oaxaca("income ~ education + age + female", data=df, by="female")
print(d.explained)     # 16.00  (covariate-driven gap)
print(d.unexplained)   #  4.00  (coefficient-driven gap)
print(d.total_gap)     # 20.00  (female mean - male mean)
```

## Installation

```bash
pip install open-econs                           # core: OLS, Oaxaca, FE, IV, Logit, Probit
pip install open-econs[plot]                      # + matplotlib for .plot()
pip install open-econs[dev,lint]                  # + development & linting tools
pip install git+https://github.com/qmanhbeo/open-econs.git    # latest dev
```

Requires Python ≥ 3.10.

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

# --- OLS with named coefficients and cluster-robust SEs ---
r = oe.ols("income ~ education + age", data=df, cluster="province")
r.coefficients          # pd.Series with name index
# Intercept   -32.82
# education     8.97
# age          -1.03

r.tidy()              # coefficient table as DataFrame
r.predict(df.head(2))# out-of-sample predictions
# 0    31.28
# 1    44.10

print(r.summary())    # printable summary (also __repr__)

# --- Logit / Probit ---
r_logit = oe.logit("female ~ education + age", data=df)
r_logit.tidy()        # coef, z, P>|z| table
r_logit.margins()     # average marginal effects
r_logit.predict(proba=False)  # binary class prediction

# --- Fixed effects ---
r_fe = oe.fe("income ~ education + age", data=df, entity="province")
r_fe.tidy()           # within-transformed coefficients

# --- IV / 2SLS ---
r_iv = oe.iv("income ~ education | age", data=df)
r_iv.tidy()           # 2SLS coefficients
r_iv.first_stage()    # first-stage F-stat

# --- VIF diagnostics ---
ctx = oe.Context(df)
ctx.vif("income ~ education + age")  # VIF per variable

# --- Oaxaca-Blinder decomposition ---
# NOTE: the 'by' column must also appear on the RHS of the formula
d = oe.oaxaca("income ~ education + age + female", data=df, by="female")
d.explained          # 16.00  (covariate-driven gap)
d.unexplained        #  4.00  (coefficient-driven gap)
d.total_gap          # 20.00  (female mean - male mean)

# Advanced: two-fold with different reference coefficients
d_omega = oe.oaxaca("income ~ education + age + female", data=df, by="female",
                     reference="omega")     # pooled without group dummy
d_g1    = oe.oaxaca("income ~ education + age + female", data=df, by="female",
                     reference="group1")    # Group 1 coefficients as reference
d_cust  = oe.oaxaca("income ~ education + age + female", data=df, by="female",
                     reference=0.7)         # custom weight (0–1)

# Advanced: three-fold from Group 1 perspective
d_rev = oe.oaxaca("income ~ education + age + female", data=df, by="female",
                   decomposition_type="three-fold", reverse=True)

# --- Context remembers the dataset ---
ctx.ols("income ~ education + age")            # same as oe.ols(..., data=df)
ctx.logit("female ~ education + age")
ctx.probit("female ~ education + age")
ctx.vif("education + age")

# --- Immutability ---
r.f_statistic = 0.0  # AttributeError: OLSResult is immutable
```

## Design Principles

- **Every result is immutable** once `fit()` completes.
- **All numeric artifacts are named** (`pd.Series`/`pd.DataFrame` with
  variable-name indices).  No raw `numpy.ndarray` crosses the public API.
- **Every error tells you what to fix.** Missing column → names the column,
  lists what's available. Non-binary `by` → shows the values found.
- **Consistent interface across estimators**: `summary()`, `tidy()`,
  `export()`, `predict()` (where applicable).

## Estimators

| Function | Description |
|---|---|
| `ols()` / `reg()` | OLS with HC1/robust/clustered SEs, **multi-way clustering** (`cluster=["a","b"]`), **Newey-West HAC** (`cov_type="HAC"`, `hac_adjust=True` for Stata-style df correction), WLS |
| `fe()` | Fixed effects (one-way entity, two-way entity + time) |
| `iv()` | Instrumental variables / 2SLS with first-stage F-stat |
| `logit()` | Binary logit with `.margins()`, `.predict()` |
| `probit()` | Binary probit (same API as logit) |
| `oaxaca()` | Oaxaca-Blinder decomposition (two-fold, three-fold; reference types: pooled, omega, group1, group2, custom weight; reverse three-fold) |
| `nls()` | Nonlinear least squares (Gauss-Newton via scipy). `formula="y ~ f(a, b, ...)"` with `start_values`; HC0–HC3 (`white_cov`), cluster, and Newey-West HAC (`time=`); analytic Jacobian via sympy with automatic numerical fallback; parity vs `curve_fit`, R `nls()`, Stata `nl` |
| `ctx.vif()` | Variance inflation factor / collinearity diagnostics |
| `abond()` | Arellano-Bond dynamic panel (difference GMM), one/two-step Windmeijer SEs, Hansen J + AR(1)/AR(2). Collapsed one-step non-robust now matches Stata `xtabond2` to ~1e-7 |
| `staggered_did()` | Callaway-Sant'Anna (2021) staggered / heterogeneous-timing DiD; doubly‑robust `dripw` or outcome‑regression `reg`; `covariates`, cell‑by‑cell Stata parity |
| `rdd()` | Sharp / fuzzy regression discontinuity (local linear, triangular kernel) |
| `did()` / `event_study()` / `balance()` | Two-period DiD, event-study, balance tables |
| `oe.PanelContext(...)` | `pooled/fitted/fe/re/diff/driscoll_kraay/hausman/abond` with remembered entity/time |

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

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## References

Stata reference implementations used for parity testing:

- ``xtabond2`` — David Roodman's Arellano-Bond dynamic panel GMM (`source <https://github.com/droodman/xtabond2/blob/master/xtabond2.ado>`_).
- ``csdid`` / ``drdid`` — Fernando Rios-Avila's Callaway-Sant'Anna staggered DiD (`csdid <https://github.com/friosavila/stpackages/tree/main/csdid>`_, `drdid <https://github.com/friosavila/stpackages/tree/main/drdid>`_).
- ``oaxaca`` — Ben Jann's Oaxaca-Blinder decomposition (`source <https://github.com/benjann/oaxaca>`_).

Methodology:

- Callaway, B., & Sant'Anna, P. (2021). Difference-in-Differences with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200–230. https://doi.org/10.1016/j.jeconom.2020.12.001
- Sant'Anna, P. C., & Zhao, J. (2020). Doubly Robust Difference-in-Differences Estimators. *Journal of Econometrics*, 219(1), 101–122. https://doi.org/10.1016/j.jeconom.2020.06.003
- ``DRDID`` (R) — Pedro H. C. Sant'Anna's doubly-robust DiD implementation (`github.com/pedrohcgs/DRDID <https://github.com/pedrohcgs/DRDID>`_, ``R/drdid_panel.R``). Basis for ``staggered_did()``'s ``dripw`` influence function (the ``trad`` method, ``csdid``'s default ``dripw``).

