# Tutorial: Difference-in-Differences

open-econs covers three DiD flavors: static (two-way) DiD, event-study
(leads/lags), and staggered adoption. All share the immutable result interface.

## 1. Setup

```python
import numpy as np
import pandas as pd
import open_econs as oe

rng = np.random.default_rng(7)
n = 400
firm = np.repeat(np.arange(100), 4)
year = np.tile([2019, 2020, 2021, 2022], 100)
treat = (firm % 2 == 0).astype(int)
post = (year >= 2021).astype(int)
tau = 1.5
y = 1.0 + 0.5 * post + 0.3 * treat + tau * treat * post + rng.normal(size=n)

df = pd.DataFrame({"firm": firm, "year": year, "treat": treat,
                   "post": post, "y": y})
```

## 2. Static (two-way) DiD

```python
res = oe.did("y ~ treat*post", data=df,
             treatment="treat", post="post", cluster="firm")
res.tidy()            # the treat:post interaction is the ATT
```

## 3. Event study (leads / lags)

`event_study()` needs a relative-time column named `{treatment}_event_time`
(e.g. `-2, -1, 0, 1, 2`) in the data; it builds the relative-time dummies
automatically. `omitted_period=-1` (the period just before adoption) is the
normalization baseline.

```python
df3["treat_event_time"] = df3["year"] - 2021     # years 2019..2022 -> -2..1
es = oe.event_study("y ~ 1", data=df3,
                    treatment="treat", post="post",
                    omitted_period=-1, cluster="firm")
es.tidy()            # one coefficient per relative-time period
```

## 4. Staggered adoption

For rolling treatment timing, use `did_cs()` (csdid-style aggregated
influence function). It takes **column names** (not a formula), and the
`treatment` column must be a *time-varying* adoption indicator (0 before
adoption, 1 from adoption onward) so that different entities form different
cohorts.

```python
periods = [2015, 2016, 2017, 2018, 2019, 2020]
firm = np.repeat(np.arange(80), 6)
time = np.tile(periods, 80)
# staggered adoption: entities adopt in different years
adopt = np.where(firm % 4 == 0, 2017,
         np.where(firm % 4 == 1, 2018,
         np.where(firm % 4 == 2, 2019, 9999)))   # 9999 = never treated
treat = (time >= adopt).astype(int)
y = 1.0 + 0.4 * time + 1.5 * treat + rng.normal(size=len(firm))

sdf = pd.DataFrame({"firm": firm, "time": time, "treat": treat, "y": y})

res_s = oe.did_cs(
    sdf, y="y", entity="firm", time="time", treatment="treat",
    cov_type="cluster",          # default; preferred for publication
)
res_s.summary()         # aggregated ATT + per-lead event-study table
res_s.att               # overall pooled ATT
res_s.att_group_time    # per-cohort x lead table
```

### did_cs HAC is experimental

`did_cs(..., cov_type="HAC", lags=L)` is a **project convention**, not an
externally validated estimator: it applies a Newey-West Bartlett correction for
common-time shocks to the aggregated influence function. It is **not** symmetric
with `iv()`/`gmm()`/`did()`/`event_study()` HAC (those rest on canonical
Newey-West). A `UserWarning` is raised on use, and `lags` is required. At
`lags=0` it reduces exactly to the cluster-robust SE. Prefer `cov_type="cluster"`
for publication.

## 5. Parity note

Static DiD matches Stata's `did` (two-period) and R's `fixest::feols(y ~ treat*post)`.
Staggered DiD matches `csdid`'s aggregated ATT; the HAC convention does **not**
have a Stata/R reference and is not claimed to.
