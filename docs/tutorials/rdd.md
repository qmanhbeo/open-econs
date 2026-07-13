# Tutorial: Regression Discontinuity (RDD)

open-econs implements both **sharp** and **fuzzy** RDD via triangular-kernel
local linear regression (`oe.rdd`), plus the McCrary /
Cattaneo–Jansson–Ma density (manipulation) test (`oe.density_test`). The
estimator mirrors R's `rdrobust` and Stata's `rddensity`.

This walkthrough runs with **no optional dependencies**. By default `oe.rdd`
uses the `cct` bandwidth selector, which delegates to the `rdrobust` package;
when that is not installed (or you pass `bandwidth_select="ik"`), open-econs
falls back to the from-source Imbens–Kalyanaraman (IK) bandwidth with the
nearest-neighbour (NN) or Eicker–Huber–White (EHW) variance — both implemented
natively. Likewise `oe.density_test` falls back to a from-source CJM estimator
when the `rddensity` package is absent (`backend="builtin"`). Installing the
optional `rd` extra (`pip install open-econs[rd]`) enables the reference R/Stata
backends.

## 1. Simulate a sharp RDD

A running variable `x` with a cutoff at 0; the outcome jumps by `tau = 2.0`
above the cutoff.

```python
import numpy as np
import pandas as pd
import open_econs as oe

rng = np.random.default_rng(7)
n = 2000
x = rng.uniform(-1.0, 1.0, n)
tau = 2.0                                  # true discontinuity
treat = (x >= 0.0).astype(float)           # treatment = being above the cutoff
eps = rng.normal(0.0, 0.5, n)
y = 1.0 + 0.5 * x + tau * treat + eps

df = pd.DataFrame({"y": y, "x": x, "treat": treat})
```

## 2. Estimate the discontinuity (sharp RDD)

Use the from-source IK bandwidth + EHW variance so the example needs no
optional packages.

```python
res = oe.rdd(
    df, y="y", running="x", cutoff=0.0,
    bandwidth_select="ik", vce="ehw",
)
res.tidy()
```

```
        term  coef  std_err       z  P>|z|
0  discontinuity  2.0*  ...      ...    ...
```

```python
print(res.summary())
```

```
            Sharp Regression Discontinuity Results
==================================================================
  Running variable : x
  Outcome          : y
  Cutoff           : 0.0
  Bandwidth (h)    : ...
  Observations     : ... left / ... right
  Discontinuity    : 2.0... (se ..., p ...)
```

`RDResult` exposes: `.effect` (the discontinuity), `.se`, `.z_stat`,
`.p_value`, `.bandwidth`, `.n_left` / `.n_right` (observations inside the
bandwidth on each side), plus `.tidy()` / `.summary()`.

## 3. Density / manipulation test

The density test checks whether units sorted around the cutoff (evidence of
manipulation that threatens the design). It uses only the running variable,
not the outcome.

```python
dt = oe.density_test(df, "x", 0.0, backend="builtin")
dt.tidy()
print(dt.summary())
```

`DensityTestResult` reports `theta = fhat_right - fhat_left` (the density
discontinuity at the cutoff), its `se`, the `z` statistic and `P>|z|`, the
one-sided density estimates `fhat_left` / `fhat_right`, and the chosen
bandwidths `h_left` / `h_right`. For a cleanly simulated running variable
`theta` is near zero and `P>|z|` is large (no rejection). A positive,
significant `theta` signals bunching just above the cutoff.

You can also call the density test off the fitted `RDResult`:

```python
res.density_test(df, backend="builtin")
```

## 4. Fuzzy RDD

With imperfect compliance, the cutoff's jump does not equal the treatment jump.
Supply the actual `treatment` column and set `fuzzy=True`; `oe.rdd` estimates
the local-linear-IV ratio (reduced-form ÷ first-stage).

```python
rng2 = np.random.default_rng(11)
x2 = rng2.uniform(-1.0, 1.0, n)
# treatment probability jumps at the cutoff (0.2 -> 0.8) but is not deterministic
p = 0.2 + 0.6 * (x2 >= 0.0)
treat_f = (rng2.uniform(size=n) < p).astype(float)
y_f = 1.0 + 0.5 * x2 + tau * treat_f + rng2.normal(0.0, 0.5, n)
df_f = pd.DataFrame({"y": y_f, "x": x2, "treat": treat_f})

res_f = oe.rdd(
    df_f, y="y", running="x", cutoff=0.0,
    treatment="treat", fuzzy=True,
    bandwidth_select="ik", vce="ehw",
)
res_f.tidy()           # recovers tau = 2.0 (not the 0.6 first-stage jump)
```

## 5. Parity note

- **Sharp RDD** matches R `rdrobust` (CCT bandwidth, separate-side local linear
  regression, NN cluster-robust variance) when the `rd` extra is installed; the
  built-in IK + NN/EHW path is the no-dependency fallback and is validated
  against the same design in the unit tests (`tests/test_rdd.py`).
- **Density test**: `backend="rddensity"` wraps the authors' reference
  implementation and matches Stata `rddensity` to machine precision when given
  the same bandwidth; `backend="builtin"` is the from-source reproduction.
- **Honest limitations**: the default (`cct`) bandwidth requires `rdrobust`;
  without it open-econs silently falls back to IK (so a run that *looks* like
  CCT may actually be IK — pass `bandwidth_select` explicitly if it matters).
  Only the triangular kernel is supported. Fuzzy RDD uses the ratio-of-sharp
  estimator, not a full local-IV polynomial specification. No Stata/R numerical
  parity is re-asserted here; the gated parity tests live in
  `tests/test_rdd.py` / `tests/test_density_test.py`.
