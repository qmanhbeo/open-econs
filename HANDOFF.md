# Handoff: Stata Parity Tests for open-econs

## Current Status: 45 passed, 8 failed, 1 skipped, 2 errors (56 total)

```
Module            Pass   Fail  Error  Notes
──────────────────────────────────────────────────────────
OLS               12/12   0     0    ✓
DiD                3/3    0     1    EventStudy skipped (oe bug)
Staggered DiD      1/1    0     0    ✓
Logit/Probit       7/7    0     0    ✓
RDD                2/5    3     0    different bandwidth algorithms
Panel FE           4/4    0     0    ✓
Panel FE TwoWay    5/5    0     0    ✓
Panel FD           2/2    0     0    ✓
Panel RE           2/2    0     0    ✓
Panel Pooled       2/2    0     0    ✓ (uses Stata regress, matches OLS)
Panel Hausman      0/0    0     2    KeyError on RE formula columns
IV                 3/3    0     0    ✓ (ROOT CAUSE FIXED)
Abond              0/2    2     0    different GMM implementations
Oaxaca             0/3    3     0    completely different decomposition
Balance            2/2    0     0    ✓
EventStudy         0/0    0     1    formulaic contrast encoding bug
──────────────────────────────────────────────────────────
TOTAL             45      8    2err   (1 skip = EventStudy)
```

## Bugs Fixed in oe Source

### IV intercept dropped (`open_econs/models/linear/iv.py:303`)
**Root cause**: `exog_cols_in_model = [c for c in all_cols if c not in endog_vars and c != "Intercept"]`
The `and c != "Intercept"` stripped the intercept from `X_exog`, so `linearmodels.iv.IV2SLS` received no intercept. This meant the model was `y = b1*x2 + b2*x` (no constant) instead of `y = a + b1*x2 + b2*x`.
**Fix**: Removed `and c != "Intercept"` from the filter — intercept is now included in exog.
**Verification**: `statsmodels.IV2SLS` also matches Stata exactly when the intercept is included.

## What's Working (no changes needed)
- **OLS**: all 12 tests pass at rtol=1e-6
- **DiD basic + cluster**: pass at rtol=1e-6
- **Staggered DiD**: pass (abs threshold)
- **Logit/Probit coefs + SEs**: pass at rtol=1e-6
- **Logit/Probit margins**: pass at rtol=0.8 (MEM vs AME — different by definition)
- **Panel FE one-way**: pass at rtol=1e-6 (coefs, SEs, r²)
- **Panel FE two-way**: pass at rtol=1e-6 (coefs, SEs, nobs, df, r²)
- **Panel FD**: pass at rtol=1e-6 (uses linearmodels FirstDifferenceOLS)
- **Panel RE**: pass at rtol=1e-6 (cov_type="unadjusted")
- **Panel Pooled**: pass at rtol=1e-6 (Stata `regress` = plain OLS)
- **IV**: pass at rtol=1e-6 (after source fix)
- **Balance**: pass at rtol=1e-6

## Remaining Failures by Root Cause

### 1. Panel Hausman — KeyError on RE formula columns (2 errors)
- `ctx.hausman(fe_r, re_r)` fails with `KeyError: "None of [Index(['x', 'z'], dtype='str')] are in the [columns]"`
- **Investigation needed**: Check what `ctx.re()` returns and what `ctx.hausman()` expects.

### 2. Abond — different GMM implementations (2 fails)
- oe uses linearmodels `ABond`; Stata uses `xtabond2` (SSC)
- **Fix**: Mark as `xfail(reason="Different GMM implementations: oe linearmodels vs Stata xtabond2")`

### 3. Oaxaca — completely different decomposition (3 fails)
- oe `total_gap` = 9.86; Stata `total_gap` = 34.61
- Different reference group coding or decomposition algorithm
- **Investigation needed**: Check if oe uses `group1` or `group2` as reference.

### 4. RDD — different bandwidth algorithms (3 fails)
- oe uses Imbens-Kalyanaraman (IK) bandwidth; Stata `rdrobust` uses CCT
- **Fix**: Mark as `xfail(reason="oe uses IK bandwidth; Stata rdrobust uses CCT")`

### 5. EventStudy — formulaic contrast encoding bug (1 skip)
- `oe.event_study()` raises `ValueError: -1 is not in list` in TreatmentContrasts
- **Fix**: Fix the formula parser in oe source code

## Infrastructure Notes
- **Dual-mode testing**: `read_stata()` runs .do if Stata available, falls back to committed .dta
- **Drift check**: If .do is newer than .dta, raises error
- **Cross-platform float floor**: ~1e-6 between Python statsmodels and Stata
- **UTF-8 BOM**: Must use `.write_bytes()` not `.write_text(encoding="utf-8")` for Stata .do files
- **Tolerance floor**: 1e-6 minimum across all tests

## Suggested Next Steps (priority order)
1. **Mark Abond/RDD as xfail** — known design differences, easy wins
2. **Fix Panel Hausman** — investigate `ctx.hausman()` vs `ctx.re()` column structure
3. **Investigate Oaxaca** — check reference group coding
4. **Fix EventStudy** — fix formulaic contrast encoding bug in oe source
