# Tutorial: Instrumental variables (2SLS)

This tutorial fits a two-stage least-squares IV model with `oe.iv()`.

## 1. Setup with endogeneity

```python
import numpy as np
import pandas as pd
import open_econs as oe

rng = np.random.default_rng(1)
n = 600
z = rng.normal(size=n)                       # valid instrument
u = rng.normal(size=n)
x = 0.7 * z + 0.5 * u + rng.normal(size=n)   # endogenous regressor (correlated w/ u)
y = 2.0 * x + u

df = pd.DataFrame({"y": y, "x": x, "z": z})
```

## 2. Fit 2SLS

`oe.iv` uses R-style syntax with **three parts**: `y ~ exog | endog ~ instruments`.
Variables left of the first `|` are exogenous regressors, between the bars are
endogenous regressors, and right of the second `|` are the excluded instruments.

```python
res = oe.iv("y ~ 1 | x ~ z", data=df)   # intercept + endogenous x, instrumented by z
res.summary()
```

You can include exogenous controls and multiple instruments:

```python
res2 = oe.iv("y ~ 1 + w | x ~ z1 + z2", data=df)   # w exogenous, x endogenous, z1/z2 instruments
```

> Note: the legacy two-part syntax `y ~ x | z` (treating every RHS variable as
> endogenous) still runs but emits a `FutureWarning`; prefer the explicit three-part
> form above.

`oe.iv` shares `ols()`'s formula grammar. With exactly-identified systems the
estimate equals `oe.ols("y ~ x", data=df).coefficients` on the instrumented `x`.

## 3. Inference and diagnostics

```python
res.tidy()
res.vcov()
res.stage1_results      # first-stage estimates (if exposed)
```

`oe.iv` supports `cov_type` (e.g. `"HC2"`, `"robust"`) the same way as `ols()`,
and `cov_type="HAC"` (validated canonical Newey-West) for panel/timeseries data
with a `time` column and `lags`. One-way cluster-robust standard errors are
available via the separate `cluster="<col>"` argument (mirroring `ols()`), which
takes precedence over `cov_type`.

## 4. Overidentification

When instruments are overidentified, the Hansen J / Sargan test of
overidentifying restrictions is reported (consistent with `ivreg2`'s `J`). Use
it to sanity-check instrument validity.

## 5. Parity note

Coefficients and SEs match Stata's `ivregress 2sls y (x = z)`, R's
`AER::ivreg(y ~ x | z)`, and statsmodels' `IV2SLS`. `cov_type="HAC"` matches
`ivregress, vce(hac ...)`.
