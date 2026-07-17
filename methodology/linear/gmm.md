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
- **HAC two-step (KNOWN DIVERGENCE)**: R applies the Bartlett kernel to BOTH the weighting matrix and the VCE; OE applies it to the VCE only. R two-step HAC SEs therefore diverge from OE. Documented, not asserted in parity tests.
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
| `... , step="two-step", cov_type="robust"` (over-ID) | `gmm (...), instruments(z1..z5) winitial(unadjusted)` | coefficients match ≤1e-7 |

Over-identified fixtures use `instruments()` + `winitial(unadjusted)` (standard 2SLS one-step). Coefficients match OE ≤1e-7 in all cases.

### R (`gmm::gmm`, v1.9-1)

| open-econs | R | Notes |
|------------|---|-------|
| `oe.gmm(..., step="two-step", cov_type="robust")` | `gmm(g, type="twoStep", wmatrix="optimal", vcov="MDS", centeredVcov=FALSE)` | matches to machine epsilon |
| `oe.gmm(..., robust_meat="two-step")` | — (R has no e2-meat option) | no clean R anchor; not asserted |

R source evidence (`.weightFct`): `MDS` weight = `crossprod(gt)/n` from one-step residuals `gt`. This confirms R uses e1 for both bread and meat (on OE's side, not Stata's).

## Root-Cause Knowledge (do NOT re-trace)

**The 2.7% two-step robust SE gap was resolved as follows.** Stata's two-step `S` (the robust meat) is built from e2, not e1. Stata's `e(S)` is extractable from a live run and equals `(1/N)·Σᵢ(Zᵢ·e2ᵢ)(Zᵢ·e2ᵢ)'` to machine epsilon; feeding Stata's OWN extracted `e(S)` into the full-sandwich formula reproduces Stata's `e(V)` to ~2e-08. Stata's two-step S computation lives inside the **compiled Mata binary `_gmm_wrk()`** (no `.mata` source shipped), so it was confirmed numerically, not from source — but the result is definitive.

**Reference values** (300-obs `df_gmm.csv`, `y ~ x1+x2 | z1..z5`, two-step robust):
- Stata `gmm` SE: `[0.1260902, 0.0986776, 0.7745471]`
- OE `windmeijer=False, robust_meat="two-step"`: same (gap 2.06e-08) ✅
- OE default (`windmeijer=True, robust_meat="one-step"`) = R: `[0.14527, 0.10322, 0.82625]`

**FOOTGUN (rule 18):** `robust_meat="two-step"` switches ONLY the robust meat S2 to e2; the efficient-weight bread S1 stays at e1. A "global S swap" (replacing S1 with e2 everywhere) regresses parity to **0.00013**. The atol=1e-6 test `TestGmmOverIdentifiedTwoStepRobust.test_standard_errors_stata_parity` guards this — any such regression fails CI.

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
2. HAC two-step diverges from R (kernel applied to weight+VCE in R, VCE only in OE) — documented, not asserted.
3. `small_sample_correction` not exposed via `gmm()` (only via `abond()`).
4. No one-step J asserted against Stata (model-based vs robust weighting divergence).

## References

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*.
- Windmeijer, F. (2005). A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators. *Journal of Econometrics*.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*.
