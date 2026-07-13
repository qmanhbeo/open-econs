# v0.6.x — numerical parity & fixture detail (rolled up)

> Moved from `ROADMAP.md` during the 2026-07-13 trim. This file holds the
> patch-level numeric parity / fixture detail for v0.6.1–v0.6.9 so the roadmap
> can stay a concise narrative. Feature bullets remain in `ROADMAP.md`.

## Arellano-Bond / `xtabond2` numerical parity (v0.6.5)
- **`abond()` collapsed one-step non-robust matches Stata `xtabond2`** to ~1e-7
  (coefficients *and* VCV), on the 30×5 `df_panel` fixture.
- **GMM instrument lag fixed** — Stata `gmm(L.y, lag(a b))` lags `L.y`, not `y`;
  prior construction silently used the initial observation as an instrument (the
  dominant VCV-gap source).
- **Weighting-matrix `H` corrected** to the true unnormalized `M'M`.
- Ground-truth extraction tooling (`abond_gt_*.csv`) for element-by-element
  validation; parity test added.

## Non-collapsed / full GMM instruments (v0.6.6)
- **Non-collapsed `abond()`** — full GMM expansion matching Stata
  `_MakeGMMinsts`/`_Explode`; block builder isolated, `n_gmm_i` formula corrected
  for lag offset.
- **All four flavors** (collapsed/non-collapsed × one/two-step) validated vs
  Stata at machine precision (~1e-8).
- **40 Stata-parity tests**; collapsed path untouched.

## Oaxaca Stata parity & advanced options (v0.6.7, v0.6.8.3)
- **Stata parity fix** — `.do` files mis-extracted `e(b)` columns; regenerated
  `.dta`. OE matches Stata `pooled`/`threefold` to machine precision.
- **`reference`** (two-fold: pooled/omega/group weights) and **`reverse`**
  (three-fold) parameters.
- **21 new tests** at 1e-12 (v0.6.7); **20 new parity tests** vs Stata `oaxaca`
  v4.1.1 with Stata↔OE terminology documented (v0.6.8.3). 96 Oaxaca tests total.
- Expanded fixtures: two-fold (4 reference variants) + three-fold (default +
  reverse) components extracted.

## RDD rdrobust backend (v0.6.8)
- **RDD rdrobust backend** — CCT bandwidth, separate-side LLR, NN cluster-robust
  variance (`pip install open-econs[rd]`, rdrobust >= 2.0).
- **RDD built-in fallback** — IK bandwidth, NN/EHW variance (no rdrobust
  dependency).
- **event_study() fix** — falls back to first available period when
  `omitted_period` absent.
- Removed faulty Stata event-study test (DiD vs event-study parameterization
  mismatch).

## Logit/Probit AME correction (v0.6.8.1)
- **`logit()/probit().margins()` now AME** (`at="overall"`), matching Stata
  `margins, dydx(*)`.
- **Fixed logit margins fixture** (`_b[x1]` → `r(b)[1,1]`); tolerances tightened
  to 1e-6.

## CS2021 DR staggered-DiD cell-by-cell parity (v0.6.8.2)
- **`staggered_did()` now the full Callaway & Sant'Anna (2021) DR group-time
  estimator** (was simplified OLS approx): `dripw` + `reg`.
- **18 cell-by-cell Stata-parity tests**; ATT(g,t) coefficients match at 1e-6.
- **Unbalanced-cohort fixture** added; weight-formula audit (cohort-proportional,
  not uniform).
- (Full IF-rewrite detail: see `docs/cs2021-dr-recon.md`.)

## Newey-West `hac_adjust` + Panel FD fixture (v0.6.8.4)
- **`ols(hac_adjust=True)` / `newey_west_cov(adjust=True)`** — opt-in Stata-style
  `N/(N−K)` HAC df correction; parity tests at 1e-7.
- **Panel FD fixture corrected** (`regress dy dx dz, noconstant`); tolerances
  tightened to 1e-6.

## Full Stata-parity coverage & caching (v0.6.9)
- **Event-study Stata parity** (synthetic fixture, t-dist inference) at 1e-6.
- **All 8 ABOND flavors live-verified** via `read_stata()` (40 tests at 1e-6);
  non-collapsed ~1e-16, collapsed ~3e-9.
- **Module-level `read_stata()` caching** — 22 Stata calls; suite 235s→94s
  (2.5×).
- All 149 Stata-parity tests pass.
- **staggered-DiD live `read_stata()` conversion deferred** to the v0.7 DR
  rewrite.
