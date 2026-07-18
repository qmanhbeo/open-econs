# open-econs v1.1.0 — Release Notes

**Status: PENDING.** Version bumped to `1.1.0` in `pyproject.toml`; CHANGELOG,
ROADMAP, README, and this document updated. Tag `v1.1.0` + GitHub Release + PyPI
upload to follow (the GitHub Release `published` event triggers the trusted-
publisher PyPI workflow).

## Highlights

- **Version `1.1.0`** — the first **time-series econometrics** release.
- **New `open_econs/models/timeseries/` module**, wrapping `statsmodels.tsa` and
  `arch` under the same source-verified Stata/R parity discipline as the rest of
  the library. *Wrapping is an implementation strategy, not a parity exemption*
  — every wrapped estimator is validated against Stata `.ado`/Mata or R package
  source (AGENTS.md rule 1), not just against reference output.
- **One public tag bundles three internal sub-milestones** (unit-root/ARIMA/
  GARCH, VAR/VECM/Johansen, ARDL/UECM). They were developed incrementally but
  never released separately, so they ship together as `1.1.0`.

## What's new

### Unit-root / stationarity tests

`adf()`, `pp()`, `kpss()`, `dfgls()`, `zivot_andrews()` over the `arch.unitroot`
backend. Parity vs Stata `dfuller` / `pperron` / `kpss` and R `urca`. Documented
tolerance exceptions where the reference implementations themselves diverge (see
the test suite and `methodology/timeseries/`).

### ARIMA / ARMA and GARCH

- `arima()` / `arma()` — Box-Jenkins mean models via `statsmodels.tsa`, parity vs
  Stata `arima` and R `forecast`.
- `garch()` — full GARCH family via `arch.arch_model`, parity vs Stata `arch` and
  R `rugarch`. The backcast convention is matched to Stata/R.

### VAR / VECM / Johansen cointegration

- `var()` — vector autoregression with `.irf()`, `.fevd()`,
  `.test_causality()` (Granger + instantaneous), and lag-order selection with
  dual information-criterion conventions (Stata's `K*ln(2π)+K` offset and
  Lütkepohl). Parity vs Stata `var` / `vargranger` / `fcast` and R `vars`.
- `vecm()` + Johansen `coint_johansen` — trace and max-eigenvalue rank tests
  across all 5 deterministic cases via Osterwald-Lenum critical-value tables.
  Parity vs Stata `vecrank` and R `urca::ca.jo`.

### ARDL / UECM + PSS(2001) bounds test

- `ardl_fit()` / `uecm_fit()` wrapping `statsmodels.tsa.ardl` (`ARDL` / `UECM`),
  exposed at the top level.
- `.bounds_test(case)` on the fitted UECM: Pesaran-Shin-Smith (2001) **F**- and
  **t**-bounds tests for a level relationship across all 5 deterministic cases,
  returning the F/t statistics, lower/upper critical-value bounds per
  significance level (10% / 5% / 1% by default; 2.5% when `signif` includes
  `0.025`), the long-run multipliers, and the error-correction term.
- Toggles (rule 15): `cv_vintage="pss2001"` (default, published-table critical
  values — the cross-tool 1e-6 anchor) vs `"statsmodels"` (Monte-Carlo simulated,
  a documented divergence); `lr_sign="stata"` (default `−θ/ρ`).
- **Parity:** 1e-6 vs Stata SSC `ardl` (14 tests) and R `ARDL` (10 tests) on the
  canonical Pesaran denmark example (`LRM ~ LRY + IBO + IDE`, case 3) — F-stat,
  t-stat, EC term, long-run multipliers, and all critical values (incl. 2.5%)
  match. Conventions source-verified against `ardl.ado` / `ardlbounds.ado` and
  the R `ARDL` package source.
- **Footgun fixed & recorded (rules 16, 18):** an apparent ~1e-5 Stata-only gap
  was root-caused to `import delimited` reading numeric columns as
  single-precision `float`; fixed with `set type double` in the fixture
  generator. OE itself was always correct. See `methodology/timeseries/ardl.md`.

## Parity discipline

- Every new time-series estimator ships with a parity test against Stata and/or R
  at ≤1e-6, run in CI on every release. No tolerance was loosened for this
  release; genuine reference divergences are exposed as toggles or documented as
  evidenced exceptions, never papered over.
- Full suite green: **1099 passed** (excluding `synth_placebo`), ruff + mypy
  clean on `open_econs/`.

## Scope

v1.1.0 is a **feature (time-series)** release. The next method additions follow
the roadmap: `v1.2` count / limited-dependent-variable models (`poisson()`
FE-backed, `nbreg()`, `tobit()`, `heckman()`, ordered logit/probit).

## Upgrade

```bash
pip install -U open-econs
```

No breaking changes. Deprecation shims from v1.0.2 (`staggered_did` → `did_cs`,
`did_sun_abraham` → `did_sa`) remain in place and will be removed in v2.0.0.
