# Cross-estimator `cov_type` validation unification

> Moved from `ROADMAP.md` during the 2026-07-13 trim. Technical detail behind
> the v0.9 `cov_type` unified validation helper.

Every estimator that accepts `cov_type` now routes it through one shared
helper, `open_econs.core.cov_type.validate_cov_type`, so an invalid value
raises a single consistent open-econs-native `ValueError` (naming the estimator
and the accepted set) instead of an opaque statsmodels/linearmodels error or a
silent fallback to a default.

- The lowercase `"hac"` (any mixed case) is accepted as an alias for `"HAC"`
  **only** where `"HAC"` was already a valid option (`ols()`, `fe()`, `nls()`,
  `PanelContext.driscoll_kraay()`); everywhere else `"hac"` (and `"HAC"`) raise
  a clear error.
- This is a deliberate narrow exception — no other `cov_type` string is
  case-normalized (e.g. `"hc2"` is still rejected), so the fix does not quietly
  change behavior for any other value.
- Any other spelling normalization (`did()`/`event_study()` historically accept
  `"robust"` -> `"HC2"`) is passed explicitly via the helper's `aliases` map so
  the precedent is visible at the call site.
- Estimators covered: `ols()`/`reg()`, `fe()`, `nls()`, `iv()`, `gmm()`,
  `mlogit()`, `logit()`, `probit()`, `did()`, `event_study()`,
  `PanelContext.pooled()`/`fe()`/`re()`/`driscoll_kraay()`. `oaxaca()`, `cem()`,
  `staggered_did()` have **no** `cov_type` parameter (out of scope); defaults
  unchanged; no HAC support added where it did not already exist
  (`PanelContext.pooled()` intentionally excludes `"HAC"` because it has no
  `lags` param to drive Newey-West).
- Regression tests in `tests/test_cov_type_validation.py` (36 cases: invalid ->
  clear error, `"hac"` alias identical to `"HAC"`, previously-valid values
  unchanged).
