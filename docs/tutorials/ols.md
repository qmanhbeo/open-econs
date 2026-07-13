# Tutorial: OLS regression

This tutorial fits an OLS model, inspects the output, and compares it with
Stata / R / statsmodels. It is fully runnable.

## 1. Setup and data

```python
import numpy as np
import pandas as pd
import open_econs as oe

rng = np.random.default_rng(42)
n = 500
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
eps = rng.normal(size=n)
y = 1.5 * x1 - 0.8 * x2 + eps

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
```

## 2. Fit

```python
res = oe.ols("y ~ x1 + x2", data=df)
res.summary()
```

`oe.ols` is also exposed as `oe.reg`, mirroring Stata's `reg`.

## 3. Read the result

Every open-econs estimator returns an immutable result object with a uniform
interface:

```python
res.tidy()        # R-broom-style coefficient table (DataFrame)
res.vcov()        # variance-covariance matrix (DataFrame)
res.coefficients  # named Series
res.std_errors
res.p_values
res.r_squared
res.n_obs
```

## 4. Robust and clustered standard errors

```python
res_hc2 = oe.ols("y ~ x1 + x2", data=df, cov_type="HC2")          # heteroskedasticity-robust
res_clu = oe.ols("y ~ x1 + x2", data=df, cluster="firm")          # one-way clustering
res_2w  = oe.ols("y ~ x1 + x2", data=df, cluster=["firm", "year"])  # multi-way clustering
```

For panel/timeseries data with autocorrelation, use Newey-West HAC:

```python
res_hac = oe.ols("y ~ x1 + x2", data=df, cov_type="HAC", lags=2, time="year")
```

`cov_type="HAC"` (or lowercase `"hac"`) is accepted only where a `lags`
argument drives Newey-West; it is a validated canonical estimator for `ols()`,
`fe()`, `nls()`, and `PanelContext.driscoll_kraay()`.

## 5. Export

```python
res.export("ols_fit.json")   # full payload
res.export("ols_fit.csv")    # tidy() table
res.to_latex(caption="OLS results", label="tab:ols")
res.to_html(caption="OLS results")
```

## 6. Parity note

Coefficients and (non-robust) standard errors match Stata's `reg y x1 x2`,
R's `lm(y ~ x1 + x2)`, and statsmodels' `OLS`. `cov_type="HC2"` matches Stata's
`reg, robust` / R's `sandwich::vcovHC(type="HC2")`. Clustered SEs match Stata's
`vce(cluster ...)`. These are covered by gated CI parity tests that run when the
optional `[stata]` / `[r]` fixtures are available; see `docs/api_stability.md`.
