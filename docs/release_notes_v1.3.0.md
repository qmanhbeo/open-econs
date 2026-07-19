# open-econs v1.3.0 — Release Notes

**Status: PUBLISHED.** Version bumped to `1.3.0` in `pyproject.toml` and
`open_econs/_version.py`; CHANGELOG, ROADMAP, FUTURE_WORK, and this document
updated. Tagged `v1.3.0`; the GitHub Release `published` event triggers the
trusted-publisher PyPI upload (`publish.yml`), and the release parity gate
(`ci-parity.yml`) runs on the same event.

## Highlights

- **Version `1.3.0`** — the **OLS diagnostics battery** release. First-class
  post-estimation diagnostics for OLS results, each source-verified against
  Stata/R under the same parity discipline as the rest of the library.
- New `open_econs/core/diagnostics.py` module and a unified diagnostics surface
  on `OLSResult` (`open_econs/core/results.py`), covering: Breusch–Godfrey
  serial-correlation test, White heteroskedasticity test, Ljung–Box
  portmanteau test, Cook's distance, leverage (`hat`/`h`), and DFBETAS
  influence measures.
- All diagnostics match Stata/R to 1e-6 except where noted below.

## What's new

### Diagnostic methods

- `bg_test(lags)` — Breusch–Godfrey test for higher-order serial correlation
  (regress auxiliary on lagged residuals + original regressors). Parity vs
  Stata `estat bgodfrey` and R `lmtest::bgtest`.
- `white_test()` — White's general heteroskedasticity test (full cross-products
  of regressors + squares). Parity vs Stata `estat imtest, white` and R
  `lmtest::white.test`.
- `ljung_box(lags)` — Ljung–Box portmanteau statistic on residuals. Parity vs
  Stata `wntestq` (Stata's version of Ljung–Box) and R `Box.test(type="Ljung")`.
- `cooks_distance()` — Cook's D influence statistic. Parity vs Stata
  `predict, cooksd` / `ols_p` and R `cooks.distance`.
- `leverage()` — diagonal of the hat matrix (leverage / `h`). Parity vs Stata
  `predict, leverage` and R `hatvalues`.
- `dfbetas()` — standardized DFBETAS (scaled by SE of the coefficient estimate
  with the observation removed). Parity vs Stata `dfbeta` / `ols_p` and R
  `dfbetas`.

### Reference anchors

- `bg_test` → Stata `estat bgodfrey` + R `lmtest::bgtest`.
- `white_test` → Stata `estat imtest, white` + R `lmtest::white.test`.
- `ljung_box` → Stata `wntestq` + R `Box.test(type="Ljung")`.
- `cooks_distance` / `leverage` / `dfbetas` → Stata `predict` family +
  R `cooks.distance` / `hatvalues` / `dfbetas`.

## Parity discipline

- Every new diagnostic ships with a parity test against Stata and/or R at
  ≤1e-6, run in CI on every release. No tolerance was loosened for this
  release (rule 2).
- **Documented tolerance divergence (rule 15):** Stata's `dfbeta`/`ols_p`
  Cook's-D and DFBETAS accumulate each observation's leave-one-out quantities
  through independent float accumulation, so on the full-sample fixtures they
  match open-econs at **rtol=1e-4** rather than 1e-6. All other quantities
  (BG, White, Ljung–Box statistics, leverage, and the R-anchored paths) match
  at **1e-6**. This divergence is asserted via `xfail(strict=True)` where it
  sits above the 1e-6 bar, and via a dedicated 1e-4 rtol test where it is the
  expected Stata convention — never papered over.
- Full suite green before release: **1282 passed, 23 xfailed, 30 deselected**
  (excluding `synth_placebo`), ruff + mypy clean on `open_econs/`.

## Scope

v1.3.0 is a **feature (OLS diagnostics)** release. No breaking changes. The
next method additions follow the roadmap: `heckman()` and `feglm` binomial FE
absorption remain **deferred** (NOT in v1.3).

## Upgrade

```bash
pip install -U open-econs
```

No breaking changes. Deprecation shims from v1.0.2 (`staggered_did` → `did_cs`,
`did_sun_abraham` → `did_sa`) remain in place and will be removed in v2.0.0.
