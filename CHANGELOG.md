# Changelog

## [1.1.0] - 2026-07-12

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
- `reghdfe` multiway-cluster fixture (`tests/stata/do/panel_fe_multiway_cluster.do`)
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

### ols() HAC Default Flag (Noted, Not Fixed)

OE's `ols()` HAC path defaults to uncorrected Newey-West (`adjust=False`),
which deviates from Stata's `newey` command (always applies `N/(N-K)`).
This is independent of the pyfixest integration and should be the first
thing investigated when the `ols()` phase kicks off.

### D1 Correction: HC2/HC3 dof Scaling — Spike Report Superseded

**The original spike/eval report (§5) claimed that pyfixest applies
`√(n/(n−k))` to HC2, creating a discrepancy with Stata. Source verification
during the ols() phase (2026-07-13) found this claim is incorrect.**

All four independently verified sources converge on the same HC2/HC3 formula
with **no extra dof factor**:

| Source | HC2 formula | HC3 formula |
|---|---|---|
| Stata `regress.sthlp` lines 193-217 | `u²/(1−h)` | `u²/(1−h)²` |
| statsmodels `linear_model.py:1997-2016` | `e²/(1−h)` | `e²/(1−h)²` |
| pyfixest `feols_.py:858-879` | `scores/√(1−h)` → `bread @ (e²/(1−h)·xx') @ bread` | `scores/(1−h)` → `bread @ (e²/(1−h)²·xx') @ bread` |
| OE `white_cov()` `cov.py:129-209` | `e²/(1−h)` | `e²/(1−h)²` |

**D1 is closed as a non-issue: no HC2/HC3 correction is needed anywhere.**
The `√(n/(n−k))` claim did not survive contact with actual source and has
been superseded. The original eval report's §5 table should be read with
this correction in mind.
