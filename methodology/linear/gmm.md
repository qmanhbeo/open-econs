---
method: gmm
aliases:
  - linear_gmm
  - two-step_gmm
category: linear
api:
  - oe.gmm()
context_api: []
panel_api: []
problem: Efficient linear-in-parameters GMM estimation with one-step / two-step steps, robust (sandwich) VCE, and the Hansen J overidentification test.
estimator: General two-step GMM on arbitrary moment conditions Z, powered by the shared core in `open_econs/models/_gmm_core.py` (also the engine behind `abond`).
stata_equivalent:
  - gmm (Stata 17)
  - xtabond / xtdpd (dynamic panel; apply Windmeijer)
r_equivalent:
  - gmm::gmm (R package gmm, v1.9-1)
status: mature
tier: 1
references:
  - Hansen1982
  - Windmeijer2005
  - NeweyWest1987
---

# Linear GMM — Two-Step Efficient GMM with Robust VCE

> **Estimator summary**: Linear-in-parameters GMM (textbook, not panel-difference) with one-/two-step estimation, robust sandwich VCE, and the Hansen J test.

## Overview

`oe.gmm()` wraps the shared solver `estimate_gmm()` with the standard IV formula grammar. It is generic GMM (identity one-step weighting = plain 2SLS; two-step uses the efficient `S⁻¹` weighting), **not** Arellano-Bond. The AB-specific conventions (`sig2_scale`, `small_sample_correction`) are kept inside `abond()` and never exposed here.

The shared core also powers `abond()`, so every convention below applies there too — `abond()` simply inherits `windmeijer=True, robust_meat="one-step"` (correct for `xtabond`/`xtdpd`).

## Mathematical Formulation

### Estimator

One-step: `b1 = (Z'WZ)⁻¹ Z'W · (Z'X)⁻¹` (W = identity by default → 2SLS).
Two-step: `b2 = (Z'X' S⁻¹ Z'X)⁻¹ Z'X' S⁻¹ Z'Y`, with `S = Σᵢ (Zᵢ'e1ᵢ)(Zᵢ'e1ᵢ)'` (per-entity, one-step residuals).

### Two-step robust VCE — the critical convention split

The two-step robust VCE is a **full sandwich**:

\[
V = (G' S_1^{-1} G)^{-1}\; (G' S_1^{-1} S_2 S_1^{-1} G)\; (G' S_1^{-1} G)^{-1},
\qquad G = Z'X
\]

- **S1** = moment cov from the **one-step** residuals e1 — the *efficient-weight bread*.
- **S2** = moment cov from the **two-step** residuals e2 — the *robust meat*.

The literature and R's `gmm` (`vcov="MDS"`) set S2 = S1 (use e1 for both) → collapses to `V2 = (G' S1⁻¹ G)⁻¹`.
**Stata's `gmm` uses e2 for S2** (the meat only; bread stays e1). This is the single source of the two-step robust SE divergence.

## Assumptions

1. **Exogeneity of instruments**: E[Z'ε] = 0 (consistency).
2. **Identification**: L ≥ p (exact or over-identified); J has dof = L − p.
3. **No perfect collinearity** in Z'X.

## Inference

### Covariance estimators

| Estimator | Formula | Use case | Reference |
|-----------|---------|----------|-----------|
| One-step non-robust | `V1 = σ² (Z'Z)⁻¹`, `σ² = e1'e1 / N` | Homoskedastic | Hansen (1982) |
| One-step robust | `V1robust = V1 Z'X A1 S A1' Z'X V1` | Heteroskedastic | White-style sandwich |
| Two-step naive (`windmeijer=False`) | full sandwich with S2 from e2 | **Stata `gmm` parity** | this page |
| Two-step Windmeijer (`windmeijer=True`) | `V2 + D V1robust D' + 2 D V2` | literature / `xtabond` default | Windmeijer (2005) |

### Default Behavior

- `windmeijer=True`, `robust_meat="one-step"` → matches **R's `gmm`** (`vcov="MDS"`, `centeredVcov=FALSE`) to machine epsilon. This is the recommended literature default.
- For **exact Stata `gmm` parity**, use **BOTH** `windmeijer=False` and `robust_meat="two-step"`. Setting only one yields a hybrid matching neither R/literature nor Stata.

### Technical Deviations from External Software

- **Windmeijer**: OE default applies it; Stata `gmm` does **NOT** (confirmed in `gmm.ado` — no Windmeijer code, no WC-robust label). Stata `xtabond`/`xtdpd` DO apply it.
- **Robust meat residuals**: Stata `gmm` uses e2 for S2; OE/R use e1. Toggle `robust_meat="two-step"` reproduces Stata.
- **One-step J**: Stata uses model-based `A1 = (Z'Z)⁻¹/σ²`; OE uses robust S when `cov_type="robust"`. R's `specTest(tsls)` matches OE's model-based form. Both valid.
- **Two-step J**: always uses `A2 = S⁻¹` (efficient weighting) → matches machine epsilon when coefficients match.
- **HAC two-step (KNOWN DIVERGENCE vs R)**: R applies the Bartlett kernel to BOTH the weighting matrix and the VCE; OE applies it to the VCE only. R two-step HAC SEs therefore diverge from OE. Documented, not asserted in parity tests. **NOTE:** OE HAC now matches **Stata** `gmm ..., wmatrix(hac bartlett L) vce(hac bartlett L)` to ≤1e-6 (coefficients AND SEs) under `windmeijer=False, robust_meat="two-step"`. See below.
- **Intercept as instrument**: OE always includes `_cons` as its own instrument in Z; Stata `gmm` single-equation `instruments(...)` does this automatically. Fixtures match this.

## Implementation Details

### Formula Interface

```
y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5     # endog x1,x2 instrumented by z1..z5; _cons auto-included
```

### Result Object

`GMMResult` (immutable): `.coefficients`, `.std_errors`, `.z_stats`, `.p_values`, `.conf_int`, `.hansen_j`, `.hansen_j_pvalue`, `.hansen_j_dof`, `.vcov()`, `.tidy()`, `.summary()`.

### Numerical Checks

Per-entity clustering of S via `eq_entity` (default: each observation its own group for iid). `pinv` fallback on singular S / G.

## Stata / R Equivalents

### Stata (`gmm`, Stata/MP 17)

| open-econs | Stata | Notes |
|------------|-------|-------|
| `oe.gmm("y ~ x1 + x2 \| z1+z2+z3+z4+z5", df, step="two-step", cov_type="robust")` | `gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(z1 z2 z3 z4 z5)` | default = Windmeijer; ~15% larger SEs than Stata `gmm` |
| `... , windmeijer=False, robust_meat="two-step"` | same Stata `gmm` | **matches to ≤1e-6** (max gap 2.06e-08) |
| `... , step="two-step", cov_type="cluster", cluster="cluster"` (default) | `gmm (...), instruments(z1..z5) winitial(unadjusted) twostep vce(cluster cluster)` | default = Windmeijer; coef matches Stata, SEs differ |
| `... , cov_type="cluster", cluster="cluster", windmeijer=False, robust_meat="two-step"` | same Stata `gmm vce(cluster cluster)` | **matches to ≤1e-6** (coef `[0.9148,1.9888,1.6213]`, SE `[0.1232,0.0814,0.8269]`) |
| `... , cov_type="HAC", lags=3, time="t", cluster="cluster"` (default) | `tsset cluster t; gmm (...), instruments(z1..z5) winitial(unadjusted) twostep wmatrix(hac bartlett 3) vce(hac bartlett 3)` | default = Windmeijer; coef matches Stata, SEs differ |
| `... , cov_type="HAC", lags=3, time="t", cluster="cluster", windmeijer=False, robust_meat="two-step"` | same Stata `gmm wmatrix(hac ...) vce(hac ...)` | **matches to ≤1e-6** (coef `[0.8916,2.0166,1.5701]`, SE `[0.1287,0.0944,0.7973]`) |
| `... , step="two-step", cov_type="robust"` (over-ID) | `gmm (...), instruments(z1..z5) winitial(unadjusted)` | coefficients match ≤1e-7 |

Over-identified fixtures use `instruments()` + `winitial(unadjusted)` (standard 2SLS one-step). Coefficients match OE ≤1e-7 in all cases.

### R (`gmm::gmm`, v1.9-1)

| open-econs | R | Notes |
|------------|---|-------|
| `oe.gmm(..., step="two-step", cov_type="robust")` | `gmm(g, type="twoStep", wmatrix="optimal", vcov="MDS", centeredVcov=FALSE)` | matches to machine epsilon |
| `oe.gmm(..., robust_meat="two-step")` | — (R has no e2-meat option) | no clean R anchor; not asserted |
| `oe.gmm(..., cov_type="HAC", lags=3, time="t")` | `gmm(g, vcov="HAC", kernel="Bartlett", bw=4, prewhite=0, centeredVcov=FALSE)` | **coefficients match** (R kernel collapses to iid weight for the coef → equals OE robust coef `[0.870,2.027,1.464]`); **SEs diverge** (R kernel in weight+VCE) |
| `oe.gmm(..., cov_type="cluster", cluster="cluster")` | `gmm(g, vcov="iid", cluster=df$cluster)` | **DISTINCT convention** (see below); R coef `[0.850,2.012,1.354]` ≠ OE/Stata cluster coef. Flagged gap — needs a `weight` toggle to reproduce R. |

R source evidence (`.weightFct`): `MDS` weight = `crossprod(gt)/n` from one-step residuals `gt`. This confirms R uses e1 for both bread and meat (on OE's side, not Stata's). R's `cluster=` adds a clustered sandwich on top of the iid-efficient GMM — a convention **distinct from both** Stata's cluster-efficient-weight `gmm` and OE's default.

### Three-way cluster/HAC efficient-weight convention (source-confirmed 2026-07-17)

The two-step coefficient changes across covariance types because the efficient weight `A2 = S⁻¹` uses a *different* `S` per convention:

| Tool / setting | Efficient-weight bread `A2` | VCE meat `S2` | Resulting two-step `b` (df_gmm) |
|----------------|------------------------------|----------------|----------------------------------|
| **OE/Stata `gmm`, robust** | iid `S₁` (per-obs) | iid `S₂` (e2 if `robust_meat="two-step"`) | `[0.870, 2.027, 1.464]` |
| **OE/Stata `gmm`, cluster** | **cluster** `S` | cluster `S` (e2 if `robust_meat="two-step"`) | `[0.915, 1.989, 1.621]` |
| **OE/Stata `gmm`, HAC** | **HAC** `S` | HAC `S` (e2 if `robust_meat="two-step"`) | `[0.892, 2.017, 1.570]` |
| **R `gmm`, cluster (`vcov="iid", cluster=`)** | iid `S₁` | cluster `S` | `[0.850, 2.012, 1.354]` |
| **R `gmm`, HAC** | iid `S₁` (kernel-averaged) | HAC `S` | `[0.870, 2.027, 1.464]` (= OE robust) |
| **OE `gmm`, `weight="iid"` (any cov_type)** | iid `S₁` (per-obs) | cov-structure `S` (e2 if `robust_meat="two-step"`) | `[0.870, 2.027, 1.464]` (= OE robust coef) |

**KEY (corrected):** Stata's `vce(cluster c)` and `wmatrix(hac ...) vce(hac ...)` build the efficient weight from the **same** covariance structure used for the VCE (cluster S / HAC S) — NOT an iid bread. An earlier hypothesis (iid bread + cluster meat for Stata cluster) was **WRONG**: it reproduced Stata's *SE* but forced the cluster coefficient to equal the robust coefficient, breaking the b-match. The correct reconstruction is bread = `S` (the covariance structure), meat = `S` from e2 (`robust_meat="two-step"`), no Windmeijer.

The **HAC VCE meat must be the HAC S from e2**, not the clustered S. A bug was fixed where the `robust_meat="two-step"` block built `S2` via the per-entity clustered loop for all cov_types; for HAC it now calls `_hac_S(Z, e2, ...)` so the meat matches the HAC bread structure (this is what made the HAC SE match Stata to ≤1e-6).

## Root-Cause Knowledge (do NOT re-trace)

**The 2.7% two-step robust SE gap was resolved as follows.** Stata's two-step `S` (the robust meat) is built from e2, not e1. Stata's `e(S)` is extractable from a live run and equals `(1/N)·Σᵢ(Zᵢ·e2ᵢ)(Zᵢ·e2ᵢ)'` to machine epsilon; feeding Stata's OWN extracted `e(S)` into the full-sandwich formula reproduces Stata's `e(V)` to ~2e-08. Stata's two-step S computation lives inside the **compiled Mata binary `_gmm_wrk()`** (no `.mata` source shipped), so it was confirmed numerically, not from source — but the result is definitive.

**Reference values** (300-obs `df_gmm.csv`, `y ~ x1+x2 | z1..z5`, two-step robust):
- Stata `gmm` SE: `[0.1260902, 0.0986776, 0.7745471]`
- OE `windmeijer=False, robust_meat="two-step"`: same (gap 2.06e-08) ✅
- OE default (`windmeijer=True, robust_meat="one-step"`) = R: `[0.14527, 0.10322, 0.82625]`

**FOOTGUN (rule 18):** `robust_meat="two-step"` switches ONLY the robust meat S2 to e2; the efficient-weight bread S1 stays at e1 (for robust). For cluster/HAC the bread is the *cluster/HAC* S (Stata-style), NOT iid. Two separate traps: (a) forcing an iid bread for cluster/HAC breaks the Stata coefficient match; (b) building the `robust_meat="two-step"` meat via the clustered loop for HAC (instead of `_hac_S` from e2) breaks the Stata HAC SE match. Both are guarded by `TestGmmOverIdentifiedTwoStepCluster` / `...TwoStepHAC` in `tests/stata/tests/test_stata_gmm.py` (atol=1e-6). The `TestGmmOverIdentifiedTwoStepRobust.test_standard_errors_stata_parity` guard also covers the robust path.

## API Examples

```python
import open_econs as oe

# Default (matches R / literature; Windmeijer on, e1-meat)
r = oe.gmm("y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", df, step="two-step", cov_type="robust")
print(r.std_errors)   # [0.14527, 0.10322, 0.82625]

# Exact Stata gmm two-step robust parity
r = oe.gmm("y ~ x1 + x2 | z1 + z2 + z3 + z4 + z5", df, step="two-step",
           cov_type="robust", windmeijer=False, robust_meat="two-step")
print(r.std_errors)   # [0.1260902, 0.0986776, 0.7745471]
```

## Fixture regeneration

- Stata: `tests/stata/generate-fixtures/gmm.do` → `tests/stata/fixtures/expected/gmm.dta` (read via `read_stata("gmm")`). Regenerate only with `OE_REGENERATE_FIXTURES=1` + StataMP.
- R: `tests/r/generate-fixtures/gmm.R` → `tests/r/fixtures/expected/gmm.json` (read via `read_r("gmm")`).
- Diagnostic `.do`/`.dta` from the root-cause investigation live in `tests/stata/generate-fixtures/archive/`.

## Limitations

1. No nonlinear or substitutable-expression moment conditions.
2. HAC two-step diverges from R (kernel applied to weight+VCE in R, VCE only in OE) — R *coefficient* still matches (kerneled weight collapses to iid optimal), only SEs diverge. Documented, SEs not asserted.
3. `small_sample_correction` not exposed via `gmm()` (only via `abond()`).
4. No one-step J asserted against Stata (model-based vs robust weighting divergence).
5. **R cluster convention not reproduced** (flagged gap, rule 3/15): R `gmm(..., vcov="iid", cluster=)` yields a distinct coefficient `[0.850,2.012,1.354]`. OE now exposes a `weight` toggle: `weight="stata"` (default, cov-structure bread) and `weight="iid"` (iid efficient-weight bread, meat stays cov-structure). `weight="iid"` reproduces the textbook iid-weighted two-step GMM coefficient `[0.870,2.027,1.464]` (self-consistency tested in `TestGmmWeightToggleIidBread`), but this is **NOT** R's cluster coefficient — R's `cluster=` argument affects the two-step weighting beyond a plain iid bread. So R cluster remains open; `TestGmmROverIdentifiedClusterTwoStep` pins the divergence (fails loudly if silently "fixed" without explicit R-parity assertion). See FUTURE_WORK GMM-RCLUSTER for the remaining reverse-engineering path.

## References

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*.
- Windmeijer, F. (2005). A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators. *Journal of Econometrics*.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*.
