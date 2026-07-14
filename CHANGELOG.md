# Changelog

## [1.1.0] - 2026-07-12

> **⚠ SUPERSEDES ORIGINAL SPIKE REPORT §5** — The original eval report
> claimed that pyfixest applies `√(n/(n−k))` to HC2, creating a
> discrepancy with Stata. Source verification during the ols() phase
> (2026-07-13) found this claim is **incorrect**. See [D1 Correction]
> below. The original report's §5 table should be read with this
> correction in mind.

### Breaking Changes

#### `fe()`: HC2/HC3 on fixed-effects models now raises `VcovTypeNotSupportedError`

`fe()` no longer accepts `cov_type="HC2"` or `cov_type="HC3"` on models with
absorbed fixed effects. This is a **correctness fix**, not a preference change.

HC2 and HC3 apply leverage-based corrections (dividing by `1 - h_ii`) that are
statistically invalid once fixed effects have been absorbed: the hat-matrix
diagonal `h_ii` no longer reflects the true projection because the FE projection
removes degrees of freedom in a way that HC2/HC3's pointwise leverage
adjustment does not account for. Continuing to compute these would silently
return wrong standard errors under a library brand promise of Stata/R numerical
parity.

**Migration:**
- Replace `cov_type="HC2"` (or `"HC3"`) with `cov_type="HC1"` in `fe()` calls.
  HC1 applies the simpler `N/(N-K)` correction and is valid post-absorption.
- For clustered standard errors, use `cluster=...` instead (which routes to
  `CRV1` via `pyfixest`).

The error message will tell you exactly what to pass:
```
VcovTypeNotSupportedError: cov_type='HC2' is not supported for models with
absorbed fixed effects. ... Use cov_type='HC1' instead (or cov_type='CRV1' /
cluster=... for clustered SEs). See CHANGELOG for details.
```

This error is raised as `oe.VcovTypeNotSupportedError` (exported at the top
level) so you can catch it specifically if needed.

#### `fe()`: Default `cov_type` changed from `"HC2"` to `"HC1"`

The default `cov_type` for `fe()` changed from `"HC2"` to `"HC1"`. This is
a breaking change for any code that relied on the old default without
explicitly passing `cov_type=`. HC1 matches Stata's `xtreg, fe` and is the
statistically correct default for FE models.

#### `fe()`: Standard errors may differ slightly (pyfixest dof scaling)

`fe()` now uses `pyfixest` as its compute backend for non-HAC covariance
types. The standard errors, t-statistics, p-values, and confidence intervals
may differ slightly from previous versions due to the adoption of `pyfixest`'s
small-sample corrections (fixest-standard leverage-adjusted dof scaling). These
differences are typically on the order of `1e-4` to `1e-6` and represent the
correct finite-sample adjustment.

### Added

- `fixed_effects=` kwarg on `fe()` for arbitrary N-way fixed effects. Pass a
  list of column names (e.g. `fixed_effects=["firm", "year", "industry"]`).
  Takes precedence over `entity=`/`time=`; passing both raises `ValueError`.
  Backward-compatible: existing `entity=`/`time=` callers are unaffected.
- `VcovTypeNotSupportedError` exported at top level (`oe.VcovTypeNotSupportedError`).
- `fe()` now supports multi-way cluster-robust SEs: pass `cluster=["c1", "c2"]`.
- `reghdfe` multiway-cluster fixture (`tests/stata/generate-fixtures/panel_fe_multiway_cluster.do`)
  added for D3 verification.

### Internal

- `fe()` compute backend replaced with `pyfixest.feols` for HC0/HC1/nonrobust/CRV1/CRV3.
- HAC path retains original statsmodels implementation (pyfixest does not support HAC with absorbed FE).
- `pyfixest==0.60.0` pinned in `pyproject.toml`.
- Added `_count_absorbed_dof()` helper for correct N-way FE degrees-of-freedom calculation.

### IV-Cluster Backend Exception (Source-Confirmed)

`iv()` now routes cluster-robust IV standard errors through `linearmodels`
with `debiased=False` instead of `pyfixest`'s CRV1. This is a targeted,
documented backend exception — not a convention compromise.

**Root cause:** Stata's `ivregress 2sls, cluster()` (default, no `small`
option) applies **no small-sample correction** to the cluster-robust VCE
(`ivregress.ado` lines 637-714). pyfixest's CRV1 applies
`(N-1)/(N-K) * G/(G-1)` unconditionally, which matches Stata's `small`
variant, not the default. `linearmodels(debiased=False)` matches Stata's
default exactly.

**Scope:** This exception applies only to `iv()` with `cluster=` (no FE).
For `fe()` with `cluster=`, pyfixest's CRV1 is used (matching Stata's
`xtreg, fe, cluster()` which applies the correction by default).

**Future consideration:** A `small=` parameter on `iv()` could toggle
between Stata's default (no SSC) and `small` variant (with SSC). Not
implemented now — noted for `ols()` phase.

### ols() Covariance Decisions (Resolved)

#### Default `cov_type` changed to `HC2`

`ols()` defaults to `cov_type="HC2"` rather than Stata's bare `nonrobust`
default. This is a deliberate product decision, not a parity deviation:
defaulting to non-robust standard errors is widely considered poor applied
practice — it silently understates uncertainty under heteroskedasticity,
the single most common applied-econometrics footgun. Users who want exact
Stata-default parity should pass `cov_type='nonrobust'` explicitly. This
preserves both the parity promise (the option to exactly match Stata is one
keyword away) and sound econometric defaults.

#### HAC `adjust=False` (matches NW1987, R, statsmodels)

OE's `ols()` HAC path defaults to uncorrected Newey-West (`adjust=False`),
matching the original Newey & West (1987) specification, R's
`sandwich::NeweyWest(adjust=FALSE)`, and statsmodels. This deviates from
Stata's `newey` command (which always applies `N/(N-K)`). The `adjust=True`
option is available for users who need Stata's correction.

#### Category B: F-stat Convention (Source-Confirmed, Correctness Fix)

**Stata's `regress, robust` does not report the ANOVA F-statistic.** The
Stata manual (`regress.pdf`, Methods and Formulas section) states:
*"When vce(robust) is specified, the ANOVA test is not valid, and the
statistic corresponds to a Wald test based on the robustly estimated
variance matrix."*

This is a genuine source-confirmed correctness finding, not a cosmetic
change. OE's `_compute_f_stat()` was previously using the ANOVA form
(residual sum of squares decomposition), which is only valid under
homoskedasticity. It now uses the Wald form `F = (Rβ̂)'(RVR')⁻¹(Rβ̂)/q`
where V is the actual cov_type-dependent VCE from the fit. This matches
Stata's behaviour: ANOVA F under `vce(ols)`, Wald F under `vce(robust)`.

The pyfixest backend's `_f_statistic` was also verified (tests ALL
coefficients, q=k, not slopes-only q=k-1) and is not used — OE computes
its own Wald F to ensure the correct reference distribution.

### D1 Correction: HC2/HC3 dof Scaling — Spike Report Superseded

**The original spike/eval report (§5) claimed that pyfixest applies
`√(n/(n−k))` to HC2, creating a discrepancy with Stata. This claim is
incorrect and has been superseded.**

All four independently verified sources converge on the same HC2/HC3
formula with **no extra dof factor**:

| Source | HC2 formula | HC3 formula |
|---|---|---|
| Stata `regress.sthlp` lines 193-217 | `u²/(1−h)` | `u²/(1−h)²` |
| statsmodels `linear_model.py:1997-2016` | `e²/(1−h)` | `e²/(1−h)²` |
| pyfixest `feols_.py:858-879` | `scores/√(1−h)` → `bread @ (e²/(1−h)·xx') @ bread` | `scores/(1−h)` → `bread @ (e²/(1−h)²·xx') @ bread` |
| OE `white_cov()` `cov.py:129-209` | `e²/(1−h)` | `e²/(1−h)²` |

**D1 is closed as a non-issue: no HC2/HC3 correction is needed anywhere.**

---

*Future work items (HC4/HC4m/HC5, non-Bartlett HAC kernels, pyfixest
nid/CRV3, SSC toggle) are logged in `FUTURE_WORK.md`.*
