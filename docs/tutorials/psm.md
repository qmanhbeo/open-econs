# Tutorial: Propensity Score Matching (PSM)

open-econs provides a full matching + balance + sensitivity workflow:

- `oe.psm` — nearest-neighbour 1:1 propensity-score matching **with replacement**
  (Abadie–Imbens robust SE), returning an ATE.
- `oe.cem` — Coarsened Exact Matching, returning strata and ATT weights for a
  downstream weighted regression.
- `oe.balance` — covariate balance table (standardized mean differences, variance
  ratios, weighted t-tests).
- `oe.rosenbaum_bounds` — Rosenbaum sensitivity analysis for hidden bias in
  matched pairs.

These map to R's `MatchIt` / `Matching` and Stata's `teffects psmatch` /
`rbounds`. `PSMResult` and `CEMResult` also expose a convenience `.balance()`;
`PSMResult` additionally exposes `.sensitivity()` which delegates to
`oe.rosenbaum_bounds` on the stored match pairs.

## 1. Simulate a treated/control sample

```python
import numpy as np
import pandas as pd
import open_econs as oe

rng = np.random.default_rng(3)
n = 1000
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
ps = 1.0 / (1.0 + np.exp(-(0.5 * x1 + 0.5 * x2)))
treat = (rng.uniform(size=n) < ps).astype(float)
tau = 1.5                                   # true treatment effect
y = 2.0 + x1 + x2 + tau * treat + rng.normal(0.0, 0.5, n)

df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})
```

## 2. Propensity-score matching (1:1 NN, with replacement)

```python
res = oe.psm(df, treatment="treat", covariates=["x1", "x2"])
res.tidy()
```

```
   term   coef  std_err       z  P>|z|   0.025   0.975
0   ATE  1.5*      ...    ...    ...     ...     ...
```

```python
print(res.summary())
```

`PSMResult` exposes `.effect` (ATE), `.se`, `.z_stat`, `.p_value`,
`.conf_int_lower` / `.conf_int_upper`, `.n_matched`, `.n_treated`,
`.n_control`, and `.caliper`. The `caliper=1.0` default matches Stata's
`teffects psmatch, ate caliper(1.0)`; tighter calipers are supported but are
**not** independently validated. Matching is **with replacement** (each control
may be reused), matching `teffects psmatch`.

## 3. Covariate balance

Compare covariate means between treated and control **on the matched sample**,
using the PSM match-frequency weights:

```python
bal = res.balance(covariates=["x1", "x2"])
bal
```

The balance table reports treated/control means, the difference, the
standardized mean difference (`SMD`), the variance ratio, and a weighted
t-test per covariate. After good matching, `|SMD|` should be small (convention:
< 0.1) and the t-tests should be non-significant. Unmatched balance can be
checked with the raw data:

```python
oe.balance(df, treatment="treat", covariates=["x1", "x2"])
```

## 4. Coarsened Exact Matching (CEM)

CEM is preprocessing only — it returns strata and ATT weights, not an effect.
Estimate the ATT with a weighted OLS on the matched subset (use `HC3` robust
SEs, **not** cluster-by-stratum, because CEM strata have no paired structure):

```python
cem_res = oe.cem(df, treatment="treat", covariates=["x1", "x2"])
cem_res.summary()

m = cem_res.matched.values.astype(bool)
md = df.loc[m].copy()
md["cem_w"] = cem_res.weights.loc[m].values
att = oe.ols("y ~ treat", data=md, weights="cem_w", cov_type="HC3")
att.tidy()                      # the `treat` coefficient is the ATT

cem_res.balance(covariates=["x1", "x2"])   # balance on the CEM-matched sample
```

`CEMResult` exposes `.weights` (ATT weights), `.matched` (boolean mask),
`.strata`, `.n_matched_strata`, plus `.tidy()` / `.summary()`.

## 5. Rosenbaum sensitivity bounds

`PSMResult.sensitivity()` extracts the within-pair outcome differences and runs
Rosenbaum bounds for hidden bias. (Equivalently call `oe.rosenbaum_bounds`
directly on a vector of pair differences.)

```python
rb = res.sensitivity(outcome="y")
print(rb.summary())
rb.critical_gamma          # smallest Gamma where the upper-bound p > 0.05
```

`RosenbaumBoundsResult` reports `Gamma`, the lower/upper bound p-values per
`Gamma` (`.tidy()`), and `.critical_gamma` — the smallest `Γ` at which the
upper-bound p-value rises above 0.05. A larger `critical_gamma` means the
finding is more robust to unobserved confounding. `oe.rosenbaum_bounds`
implements the Wilcoxon signed-rank bounds (Rosenbaum 2002), validated against
Stata `rbounds`.

## 6. Parity note

- **PSM ATE** is validated against Stata `teffects psmatch, ate caliper(1.0)`;
  the Abadie–Imbens (2006, 2012) robust SE with the PS-estimation adjustment
  matches Stata's `vce(robust, nn(#))`.
- **CEM** weights and strata are validated against Stata's `cem` SSC package
  (default Sturges coarsening; `autocuts` also supports `fd` / `scott` / `ss`).
- **Rosenbaum bounds** follow Stata `rbounds` (zero-difference handling and
  tie-breaking conventions).
- **Honest limitations**: matching is with-replacement (no without-replacement
  mode); only the ATE estimand is implemented (`estimand="att"` is not);
  calipers other than 1.0 are not independently validated; CEM produces weights,
  not SEs — use HC3 weighted OLS for inference. No Stata/R numerical parity is
  re-asserted here; the relevant checks live in `tests/test_psm.py`,
  `tests/test_cem.py`, and `tests/test_coverage_gaps.py`.
