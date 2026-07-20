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
- **HAC two-step — TWO conventions (source-confirmed 2026-07-17):**
  1. **Per-entity HAC (OE default + Stata):** HAC S is Newey-West *within each panel entity*, accumulated. Matches Stata `gmm ..., wmatrix(hac bartlett L) vce(hac bartlett L)` to ≤1e-6 (coef AND SE) under `windmeijer=False, robust_meat="two-step"`. Coef `[0.892, 2.017, 1.570]`.
  2. **Pooled-sample HAC (R):** R's `gmm(vcov="HAC")` applies the Bartlett kernel to BOTH weighting matrix AND VCE over the **full sample** (each obs its own entity). Selected in OE via `hac_weighting=True`. Coefficient matches R `[0.885, 2.018, 1.534]` to ≤1e-6; SE matches R to within ~6e-4.
   - **~6e-4 SE gap — root cause (source-confirmed 2026-07-17):** R is *internally inconsistent* between its coefficient and its reported VCE. The two-step **coefficient** is optimized with a HAC weight `W=S⁻¹` built from the **first-stage 2SLS residuals** (`.weightFct(z$coefficient=res1$par, ...)` in `momentEstim.baseGmm.twoStep.formula`, `res1$par` = 2SLS theta). The **reported** `vcov` (`FinRes.baseGmm.res`) builds `v = .weightFct(z$coefficient=final coef, "HAC")` from the **two-step (final) residuals** and uses it for both `z$w` and `z$vcov`. Empirically: e1-HAC (2SLS-residual) bread → R coefficient to 2e-15; e2-HAC (two-step-residual) bread → R SE to 6 decimals `[0.127674, 0.096713, 0.802281]` but a DIFFERENT coefficient `[0.888, 2.016, 1.510]`. So R's `coef(g)` and `vcov(g)` derive from two different S matrices. OE uses ONE consistent S (e1-HAC bread, also in the meat sandwich) → exact coefficient + ~6e-4 on the (inconsistent) R SE. **Do not "fix" to 1e-6** by switching bread to e2-HAC: that replicates R's inconsistency, breaks OE's coef↔SE consistency, and breaks the exact coef match. SE test stays at atol=1e-3. See FUTURE_WORK GMM-HAC.
   - **Fixture fix (2026-07-17):** `oid_hac_2s` previously stored the plain optimal coef `b2` as "R HAC" — wrong. Now stores R's actual HAC coef `[0.885,...]` / SE from `coef(g_hac_oid)` / `vcov(g_hac_oid)`.
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
| `oe.gmm(..., weight="iid", windmeijer=False, robust_meat="two-step")` | `gmm(g, vcov="iid", centeredVcov=FALSE)` ← `cluster=` is a NO-OP | **matches R `vcov="iid"`** coef `[0.850,2.012,1.354]` + SE `[0.132,0.102,0.805]` to ≤1e-6 |

R source evidence (`.weightFct`): `MDS` weight = `crossprod(gt)/n` from one-step residuals `gt`. This confirms R uses e1 for both bread and meat (on OE's side, not Stata's). **R's `cluster=` argument is NOT a real parameter** — it falls through `...` and is never consumed in `gmm()` / `FinRes.baseGmm.res` / `.weightFct`. R has **NO cluster VCE**; the historical "R cluster" fixture is simply R's plain `vcov="iid"` two-step GMM. The `vcov="iid"` weight is the *homoskedastic* `S = Z_iid' Z_iid / n` (Z_iid = intercept + explicit instruments), NOT the per-obs EHW S.

### Three-way cluster/HAC efficient-weight convention (source-confirmed 2026-07-17)

The two-step coefficient changes across covariance types because the efficient weight `A2 = S⁻¹` uses a *different* `S` per convention:

| Tool / setting | Efficient-weight bread `A2` | VCE meat `S2` | Resulting two-step `b` (df_gmm) |
|----------------|------------------------------|----------------|----------------------------------|
| **OE/Stata `gmm`, robust** | iid `S₁` (per-obs EHW) | iid `S₂` (e2 if `robust_meat="two-step"`) | `[0.870, 2.027, 1.464]` |
| **OE/Stata `gmm`, cluster** | **cluster** `S` | cluster `S` (e2 if `robust_meat="two-step"`) | `[0.915, 1.989, 1.621]` |
| **OE/Stata `gmm`, HAC** | **HAC** `S` | HAC `S` (e2 if `robust_meat="two-step"`) | `[0.892, 2.017, 1.570]` |
| **R `gmm`, `vcov="iid"`** (homoskedastic; `cluster=` ignored) | **homoskedastic** `S = Z_iid' Z_iid / n` (Z_iid = intercept + explicit instruments) | homoskedastic, scaled by `sig2` | `[0.850, 2.012, 1.354]` |
| **R `gmm`, HAC (`vcov="HAC"`)** | **pooled-sample HAC** `S` from 2SLS (first-step) residuals — used for the *coefficient* | pooled-sample HAC `S` from **two-step (final) residuals** — used for the *reported VCE* (internally inconsistent with the coef) | `[0.885, 2.018, 1.534]` (coef) / SE `[0.128, 0.097, 0.802]` |
| **OE `gmm`, `hac_weighting=True`, HAC** | pooled-sample HAC `S` (full sample, e1/2SLS) | pooled-sample HAC `S` (e2, same bread) | `[0.885, 2.018, 1.534]` (= R HAC coef ≤1e-6; SE within ~6e-4 — see e1/e2 note above) |
| **OE `gmm`, `weight="iid"` (robust cov_type)** | homoskedastic `S = Z_iid' Z_iid / n` | homoskedastic, scaled by `sig2` | `[0.850, 2.012, 1.354]` (= R `vcov="iid"`, coef+SE ≤1e-6) |

**KEY (corrected):** Stata's `vce(cluster c)` and `wmatrix(hac ...) vce(hac ...)` build the efficient weight from the **same** covariance structure used for the VCE (cluster S / HAC S) — NOT an iid bread. An earlier hypothesis (iid bread + cluster meat for Stata cluster) was **WRONG**: it reproduced Stata's *SE* but forced the cluster coefficient to equal the robust coefficient, breaking the b-match. The correct reconstruction is bread = `S` (the covariance structure), meat = `S` from e2 (`robust_meat="two-step"`), no Windmeijer.

The **HAC VCE meat must be the HAC S from e2**, not the clustered S. A bug was fixed where the `robust_meat="two-step"` block built `S2` via the per-entity clustered loop for all cov_types; for HAC it now calls `_hac_S(Z, e2, ...)` so the meat matches the HAC bread structure (this is what made the HAC SE match Stata to ≤1e-6).

## Root-Cause Knowledge (do NOT re-trace)

**The 2.7% two-step robust SE gap was resolved as follows.** Stata's two-step `S` (the robust meat) is built from e2, not e1. Stata's `e(S)` is extractable from a live run and equals `(1/N)·Σᵢ(Zᵢ·e2ᵢ)(Zᵢ·e2ᵢ)'` to machine epsilon; feeding Stata's OWN extracted `e(S)` into the full-sandwich formula reproduces Stata's `e(V)` to ~2e-08. Stata's two-step S computation lives inside the **compiled Mata binary `_gmm_wrk()`** (no `.mata` source shipped), so it was confirmed numerically, not from source — but the result is definitive.

**Reference values** (300-obs `df_gmm.csv`, `y ~ x1+x2 | z1..z5`, two-step robust):
- Stata `gmm` SE: `[0.1260902, 0.0986776, 0.7745471]`
- OE `windmeijer=False, robust_meat="two-step"`: same (gap 2.06e-08) ✅
- OE default (`windmeijer=True, robust_meat="one-step"`) = R: `[0.14527, 0.10322, 0.82625]`

**FOOTGUN (rule 18):** `robust_meat="two-step"` switches ONLY the robust meat S2 to e2; the efficient-weight bread S1 stays at e1 (for robust). For cluster/HAC the bread is the *cluster/HAC* S (Stata-style), NOT iid. `weight="iid"` is an explicit convention switch that uses the *homoskedastic* S (`Z_iid' Z_iid / n`, Z_iid = intercept + explicit instruments) for BOTH bread and meat — this reproduces R's `gmm(..., vcov="iid")` (homoskedastic GMM), NOT the per-obs EHW S. Two separate traps: (a) forcing an iid bread into cluster/HAC (Stata-style) breaks the Stata coefficient match; (b) building the `robust_meat="two-step"` meat via the clustered loop for HAC (instead of `_hac_S` from e2) breaks the Stata HAC SE match. Both are guarded by `TestGmmOverIdentifiedTwoStepCluster` / `...TwoStepHAC` in `tests/stata/tests/test_stata_gmm.py` (atol=1e-6). R `vcov="iid"` parity is guarded by `TestGmmROverIdentifiedIidTwoStep` / `TestGmmWeightToggleIidBread` (atol=1e-6).

**J-statistic convention (`/sig2`) — CLOSED, two valid conventions (rule 14).** OE's one-step J uses the model-based S: `J = g'(Z'Z)^{-1}g / sig2` (`_gmm_core.py` line ~158). Stata's `e(J)` uses the robust sandwich `Ŝ = (1/N)Σᵢ gᵢgᵢ'` — `gmm.ado` line 1358 sets `e(J) = Q·N`. R's `specTest()` does NOT divide by `sig2`. On the corrected fixture (single-equation + `winitial(unadjusted)`), Stata one-step J=3.7702 (model-based weight), OE one-step J=4.085 (robust S under `cov_type="robust"`); **two-step J matches to machine epsilon** (both use the efficient S⁻¹). OE's `/sig2` convention is kept as-is and documented in the `_gmm_core.py` module docstring + inline at line 158 + the `gmm()` docstring. Commit `a941114`. This is why no one-step J is asserted cross-tool (Limitations #4). Do NOT "reconcile" the one-step J — it is a genuine convention split.

**Expression-form weighting matrix (the 6.8% "non-convergence" that wasn't) — CLOSED.** An earlier claim that "Stata's Gauss-Newton doesn't converge to 2SLS" was **wrong**: for linear models the GMM objective is quadratic and any Newton solver converges in one step (`e(converged)=1` in every configuration; tightening tolerances has zero effect). The 6.8% coefficient gap was a **specification difference** — Stata's *multi-equation expression form* with `winitial(identity)` minimizes `(Y−Xb)'ZZ'(Y−Xb)` (the ZZ'-weighted objective), NOT the standard 2SLS objective `g'(Z'Z)⁻¹g`. These coincide only when exactly identified (L==p). Resolution: the Stata fixture (`gmm.do`/`gmm.dta`) was regenerated using single-equation form with `instruments()` + `winitial(unadjusted)`, giving standard 2SLS; all overidentified coefficients then match OE ≤1e-7 (Stata single-eq b2=1.354058 = OE 1.354058 = Python exact 2SLS 1.354058). Rationale is in the `gmm.do` header. Do NOT switch the fixture back to multi-equation expression form.

**`_hac_S` vectorization — bit-identical (provenance).** The inner per-lag/per-t `np.outer` accumulation in `_hac_S` (`open_econs/models/_gmm_core.py`) was replaced by a single batched `np.einsum("ti,tj->ij", moments[lag:], moments[:-lag])` per entity (the ragged per-entity loop is preserved). **Bit-identical** (atol=0) to the scalar loop — `sum(axis=0)` over the (T−lag, L, L) tensor equals the sequential `Gamma += np.outer(...)`; verified by `TestHacSVectorization` in `tests/non_stata_nor_r/test_gmm_core.py` (parametrized over seeds/lags, adjust flag, pooled `hac_weighting` path, no-time pooled path). Commit `897c31a`. Do NOT re-touch unless a new HAC edge case breaks bit-identicality.

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
2. HAC two-step: R applies the Bartlett kernel to BOTH weight and VCE over the **pooled sample** (`hac_weighting=True` reproduces R's HAC coefficient `[0.885,...]` ≤1e-6; SE within ~6e-4). OE default HAC is per-entity (Stata-style). Both conventions covered by tests.
3. `small_sample_correction` not exposed via `gmm()` (only via `abond()`).
4. No one-step J asserted against Stata (model-based vs robust weighting divergence; `/sig2` convention split — see Root-Cause Knowledge, commit `a941114`).
5. **R `cluster=` is a NO-OP — RESOLVED as R `vcov="iid"` (2026-07-17).** R's `gmm` `cluster=` argument is silently ignored (not a real parameter; falls through `...`). R has NO cluster VCE. The "R cluster" fixture is R's plain `gmm(..., vcov="iid")` two-step GMM (homoskedastic efficient weight), reproduced exactly by OE `weight="iid"` (`TestGmmROverIdentifiedIidTwoStep`, `TestGmmWeightToggleIidBread`, coef+SE ≤1e-6). The earlier "gc$w distinct aggregation" finding was chasing an ignored argument. See FUTURE_WORK GMM-RCLUSTER (now RESOLVED).

## System GMM (Blundell-Bond) — `oe.abond(..., system=True)`

> **Status (2026-07-19):** RESOLVED. All four collapsed system-GMM flavors (one/two-step × robust/non-robust) match Stata `xtabond2` 3.7.2 to <1e-7 on coefficients, SEs, Hansen J, and AR(1)/AR(2) tests. The weight-matrix and `sig2` conventions below are the confirmed root causes. Do NOT re-trace the W structure.

### What system GMM stacks
System GMM (Blundell-Bond 1998) stacks **two** equation blocks over the SAME parameter vector `β = (L.y, x, z, _cons)`:
- **Difference equation** `Δy_it = ...` — instrumented by lagged **levels** of y (GMM) + differenced exogenous vars (IV). This is exactly what `abond()` already estimates (the proven `xtabond2` difference-GMM path).
- **Level equation** `y_it = ...` — instrumented by lagged **differences** of y (GMM) + level exogenous vars (IV) + `_cons`. No fixed-effect column; the level instruments identify the unit effect out.

### Weight matrix is COUPLED, NOT block-diagonal (RESOLVED root cause)
**This is the single most important fact.** Stata `xtabond2` (and `xtdpdsys`) build the one-step weight matrix as the **coupled** operator

\[
H = \begin{bmatrix} M'M & M' \\ M & I \end{bmatrix}
\]

where `M` is the first-difference operator (tridiagonal, diag 2 / off −1) and the off-diagonal blocks `M'` / `M` couple each difference-equation residual at time `t` to the level-equation residual at time `t` (and `t−1`). **A block-diagonal `blkdiag(W_diff, W_level)` DROPS these cross-blocks and is WRONG** — it produces coefficients off by orders of magnitude (e.g. `b_Ly ≈ 0.33` vs target `0.0095`).

- `W_diff` block = `M'M` (our `_build_h` output: diag 2, off −1, zeroed at entity boundaries), over the difference rows.
- `W_level` block = `I` (identity) over the level rows.
- Cross blocks = `M'` (diff→level) and `M` (level→diff), built from the same first-difference operator aligned by time index `t`.

Construction: for stacked residuals `e = [e_diff; e_level]`, `H` couples `e_diff[t]` with `e_level[t]` and `e_level[t−1]`. Build `H` as a sparse (T-block) matrix per entity and `scipy.linalg.block_diag` across entities. **Do NOT use `block_diag(W_diff, I)`** without the off-diagonal `M`/`M'` blocks.

### Instrument depths (RESOLVED)
For system GMM with `gmm(L.y, lag(2 4) collapse)` (diff) and `gmm(L.y, lag(1 1) collapse)` (level):
- **Diff-eq GMM depths = 2, 3, 4** (instruments `y_{t-3}, y_{t-4}, y_{t-5}`). The depth-4 column is ALL-ZERO for T=5 panels but **xtabond2 KEEPS it** (still counts toward `zrank`). Do NOT apply the degenerate-depth filter to the diff-eq GMM block in system mode.
- **Level-eq GMM = 2 instruments**: `Δy_{t-1} = y_{t-1}−y_{t-2}` and `Δy_{t-2} = y_{t-2}−y_{t-3}` (i.e. `gmm(L.y, lag(1 1))` in the level equation collapses to two lagged-difference instruments). The single-`Δy_{t-1}` variant gets `b_Ly` close (0.0149) but `b_z` off; the two-instrument variant gets `b_x`, `b_z` to ~1e-3 but `b_Ly` stuck (~0.33). Earlier single-instrument attempts left b_Ly stuck (~0.33 vs target 0.0095); the two-instrument form is the correct Stata convention and now matches to <1e-7.
- Total instrument columns `L = 11` for the controlled set: 3 diff-GMM + `D.x` + `D.z` + 2 level-GMM + `x` + `z` + `_cons`. Stata's on-screen "Number of instruments = 10" and Hansen `dof = 6` reflect that `e(j0)=11` counts the constant (Hansen dof uses 10). **Treat the open-econs `zrank` field with care** (see footgun).

### `sig2` normalization (RESOLVED convention)
`xtabond2` computes the level-error variance from the **level-equation residuals only**, divided by 2 (ado line 662: `sig2 = e'e/2` for the level block). The single `sig2_scale` scalar in `_estimate_gmm` cannot encode both blocks, so bake the structure into `H` (the coupled `M'M`/`I`/`M'`/`M`) and pass `sig2_scale` consistent with how the core computes `sig2 = sig2_scale · e'e / wttot`. Do NOT use `sig2_scale=0.5` over the full stacked residual (wrong). The implemented `sig2` must match Stata `e(sig2)` (≈ 0.248590 two-step / 0.350040 one-step on the fixture).

### Fixture (anchor = `xtabond2`, NOT `xtdpd`)
`tests/stata/fixtures/expected/sysgmm.dta` (generated with `xtabond2`, not `xtdpd` — `xtdpd`'s diff-eq normalization differs from `xtabond2` and `abond()` already matches `xtabond2` to machine precision). Canonical generating line:
```
xtabond2 y L.y x z, gmm(L.y, lag(2 4) collapse) iv(x z, eq(diff)) ///
                     gmm(L.y, lag(1 1) collapse) iv(x z, eq(level)) [twostep] [robust] small
```
Targets (`read_stata("sysgmm")` keys `c_2s_nr` etc.):
- `c_2s_nr`: `b_Ly=0.009464`, `se_Ly=0.065036`, `b_x=1.134976`, `b_z=−0.442064`, `b_cons=0.090758`, `se_cons=0.239373`, `N=120`, `zrank=11`, `ar1=−2.654426`, `ar2=0.910692`, `hansen_j=7.262579`, `hansen_p=0.297245`, `sig2=0.248590`.
- `c_2s_r`: same coef, `se_Ly=0.113738`, `ar1=−2.174560`, `ar2=0.890461`, `sig2=0.248590`.
- `c_1s_nr`: `b_Ly=0.110421`, `se_Ly=0.051559`, `b_z=−0.603776`, `ar1=−3.757261`, `ar2=1.157888`, `hansen_j=NaN` (one-step non-robust), `sig2=0.350040`.
- `c_1s_r`: `b_Ly=0.110421`, `se_Ly=0.121040`, `hansen_j=7.262579`, `sig2=0.350040`, `ar1=−2.565394`, `ar2=0.961705`.
- **One-step AR is NON-empty** (assert the values; do NOT force NaN). `hansen_j=NaN` only for `c_1s_nr` (assert `np.isnan`).
- `xtdpdsys y L.y x z, lags(1) twostep vce(robust)` is a SEPARATE 13-instrument auto-set (`b_Ly=−0.154518`) — documented only, NOT the parity target.

### Open item (RESOLVED 2026-07-19)
All four collapsed system-GMM flavors now match Stata `xtabond2` 3.7.2 to <1e-7 on coefficients, SEs, Hansen J, and AR(1)/AR(2). The coupled `H` + 2-instrument level GMM + level-residual `sig2/2` convention (below) together close the `b_Ly` gap that earlier attempts had stuck at ~0.33. The winning combination: the level-eq GMM second instrument (`Δy_{t-2}`) plus the post-small `sig2` in the AR denominator and the raw/raw `pV_ar` ratio for one-step non-robust (see `arellano_bond.md` §"1s_nr sig2 convention"). Non-collapsed system GMM remains `NotImplementedError` (footgun below).

### Footguns (rule 18)
- **Coupled `H`, not block-diagonal** — the #1 trap; block-diag gives wildly wrong coefs.
- **Diff-eq GMM depths = 2,3,4** (not 0..maxL, not including depth 1). Keep the all-zero depth-4 column.
- **`zrank` ambiguity:** Stata `e(j0)=11` counts the constant; Hansen dof = 10. Decide whether `ArellanoBondResult.zrank` reports 11 (Stata `e(j0)`) or 10 (true instrument count) and flag it. Currently set to `Z.shape[1]` (= 11).
- **`sig2` from level residuals / 2**, not the stacked residual.
- **Non-collapsed system GMM is deferred** (`raise NotImplementedError` when `system=True and collapse=False`) — `xtabond2`/`xtdpdsys` only support the collapsed form here.
- **`system=False` must be byte-identical** to the committed `abond()` (regression-guarded by `tests/stata/tests/test_stata_abond.py`).
- **R parity deferred** (broken upstream in `plm::pgmm`); no R subprocess in parity tests.

## References

- Hansen, L. P. (1982). Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*.
- Windmeijer, F. (2005). A Finite Sample Correction for the Variance of Linear Efficient Two-Step GMM Estimators. *Journal of Econometrics*.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*.
- Blundell, R., & Bond, S. (1998). Initial Conditions and Moment Restrictions in Dynamic Panel Data Models. *Journal of Econometrics*.

