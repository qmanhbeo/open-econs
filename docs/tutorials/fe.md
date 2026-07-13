# Tutorial: Fixed-effects (panel) regression

This tutorial fits a panel fixed-effects model with `PanelContext.fe()`.

## 1. Setup and panel data

```python
import numpy as np
import pandas as pd
import open_econs as oe

rng = np.random.default_rng(0)
n_firm, n_year = 100, 10
firm = np.repeat(np.arange(n_firm), n_year)
year = np.tile(np.arange(n_year), n_firm)
alpha = rng.normal(scale=2.0, size=n_firm)            # firm fixed effect
x = rng.normal(size=n_firm * n_year)
eps = rng.normal(size=n_firm * n_year)
y = 1.2 * x + alpha[firm] + eps

df = pd.DataFrame({"firm": firm, "year": year, "x": x, "y": y})
```

## 2. Fit fixed effects

`PanelContext` takes the data plus the entity and time identifiers:

```python
ctx = oe.PanelContext(df, entity="firm", time="year")
res = ctx.fe("y ~ x")
res.summary()
```

`fe()` absorbs the entity effect via the within (demeaning) transform. Add a
time fixed effect with `ctx.fe("y ~ x", cov_type="HC1")` plus a time term in the
formula, or use the two-way within transform through `fe()` after encoding time
dummies in the formula.

## 3. Standard errors

```python
res_clu = ctx.fe("y ~ x", cluster="firm")                 # cluster by entity
res_clu2 = ctx.fe("y ~ x", cluster=["firm", "year"])      # two-way
res_dk  = ctx.driscoll_kraay("y ~ x", lags=2)             # Driscoll-Kraay
```

`driscoll_kraay()` accepts `cov_type="HAC"` as a preferred alias for the
historical `cov_type="kernel"` (both are the same period-aggregation Bartlett
Newey-West estimator).

## 4. Other panel estimators via the same context

```python
ctx.re("y ~ x")                  # random effects (FGLS)
ctx.diff("y ~ x")                # first differences
ctx.hausman(fe_res, re_res)      # Hausman test (FE vs RE)
ctx.abond("y ~ x", lags=1)       # Arellano-Bond dynamic GMM
ctx.gmm("y ~ x | z1 z2")         # linear GMM
```

## 5. Parity note

`fe()` matches Stata's `xtset firm year; xtreg y x, fe` and R's `plm(y ~ x,
model="within")`. Driscoll-Kraay SEs match `xtreg ... , vce(dk)` / `plm`
`vcovDriscollKraay`. Cluster-robust SEs match `xtreg, vce(cluster firm)`.
