# Tutorial: Synthetic Control

open-econs implements the Abadie–Diamond–Hainmueller synthetic control **core
point estimator** (`oe.synth`) plus ADH permutation inference
(`placebo_space` / `placebo_time`). The estimator mirrors R's `Synth` (primary
reference) and Stata's `synth` (secondary reference).

> **API note.** `oe.synth` is a top-level export. The two inference helpers
> `placebo_space` and `placebo_time` currently live in the submodule and are
> imported as `from open_econs.models.causal.placebo import placebo_space,
> placebo_time` (a top-level `oe.placebo_space` / `oe.placebo_time` export is a
> planned v1.1 tidy-up). This tutorial uses the real, current import path so it
> runs as written.

```python
import numpy as np
import pandas as pd
import open_econs as oe
from open_econs.models.causal.placebo import placebo_space, placebo_time
```

## 1. Simulate a balanced panel

One treated unit `A` plus a donor pool, with `A` an exact convex combination of
three donors and a post-treatment shift of `3.0`.

```python
rng = np.random.default_rng(0)
N, T = 6, 16
times = list(range(1990, 1990 + T))          # 1990..2005
donors = [f"d{i}" for i in range(N)]
treated = "A"
units = [treated] + donors

base = rng.normal(size=(N, T)).cumsum(axis=1)
y = pd.DataFrame(index=pd.Index(units, name="unit"), columns=times, dtype=float)
y.loc[donors] = base

w_true = np.array([0.4, 0.35, 0.25])
y.loc[treated] = w_true @ base[:3, :]          # A is a known combo of d0,d1,d2
post = [t for t in times if t >= 2000]
y.loc[treated, post] = y.loc[treated, post] + 3.0

df = (y.reset_index().melt(id_vars="unit")
        .rename(columns={"variable": "year", "value": "y"}))
```

## 2. Fit the synthetic control

`oe.synth` takes the panel, outcome, treated unit, and donor pool as
**positionals**. `pre_period` is the last pre-treatment period; `post_period`
is the first post-treatment period.

```python
res = oe.synth(
    df, "y", treated, donors,
    entity="unit", time="year",
    pre_period=1999, post_period=2000,
)
res.tidy()                 # donor weights W
```

```
  Donor  Weight
0    d0    0.40
1    d1    0.35
2    d2    0.25
3    d3    0.00
   ...
```

```python
print(res.summary())
```

`SynthResult` exposes: `.weights` (donor weights `W`, a `pd.Series`),
`.predictor_weights` (`V`), `.pre_mspe` / `.post_mspe`, and `.gap_path` — a
`DataFrame` of `treated`, `synthetic`, and `gap` over the analysis window.

## 3. Read the gap path

The post-treatment gap (`treated − synthetic`) is the estimated effect.

```python
res.gap_path.tail()
res.gap_path.loc[2000, "gap"]     # ≈ 3.0 (the injected shift)
```

A good pre-treatment fit means the pre-window gap is near zero; the post-window
gap is the synthetic-counterfactual estimate.

## 4. Permutation inference (ADH)

`placebo_space` re-fits `synth` once per donor (each temporarily treated, the
rest as its donor pool); `placebo_time` re-fits once per candidate pre-treatment
date. The ADH p-value is the fraction of placebo post/pre-MSPE ratios that are
at least as large as the treated unit's own ratio.

```python
ps = placebo_space(res, df)       # optional: exclude_pre_mspe_multiple=10
print(ps.summary())
ps.p_value

pt = placebo_time(res, df)
print(pt.summary())
pt.p_value
```

`PlaceboSpaceResult` / `PlaceboTimeResult` expose `.p_value`, `.tidy()` (per
unit / per date MSPE and ratio table), `.ratios`, and `.gap_paths`.

## 5. What is intentionally out of scope

`SynthResult.predict()` and `SynthResult.plot()` raise `NotImplementedError` by
design in this pass — they are a separate, later-scoped task. Use `.gap_path`
directly for inspection.

## 6. Parity note

- `oe.synth` mirrors R `Synth` (predictor standardization by `1/sd`, two-start
  outer `V` optimization, inner `W` QP solved by SLSQP) and is checked against
  Stata `synth`. Because the inner `W` problem is a unique convex QP, **`W`,
  pre-MSPE, and the gap path agree closely** with R; the predictor-weight vector
  `V` may land on a different local optimum (nonconvex `V` landscape) and is
  reported honestly rather than forced to match.
- `placebo_space` / `placebo_time` are validated against R `Synth` via a gated
  parity test (p-value matches; per-donor ratio agreement for the
  well-determined majority). The **donor-exclusion** `exclude_pre_mspe_multiple`
  is an opt-in, space-only parameter (default `None`: never applied silently);
  `placebo_time` does **not** accept it (in the in-time loop "pre-MSPE" is the
  treated unit's own fit against itself, so the space-style exclusion does not
  apply).
- **Honest limitations**: this is a point estimator plus permutation inference
  only — no `predict()`/`plot()`, no conformal/synthesis variants. Parity is
  established by the *gated* R/Stata fixture tests (`tests/test_synth.py`,
  `tests/test_synth_placebo.py`), which are not re-run here; local runnability,
  not Stata/R numerical parity, is the validation bar for this tutorial.
