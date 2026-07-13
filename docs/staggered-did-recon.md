# `staggered_did()` — sample-alignment, SE-gap root cause, and HAC caveat

> Moved from `ROADMAP.md` during the 2026-07-13 trim. Technical detail behind
> the v0.7 sample-alignment fix and the v0.9 HAC rollout for `staggered_did()`.

## Sample-alignment & SE-gap root cause (v0.7.0)

- **`.do` sample-alignment** — drops never-treated gvar=5 entities to match
  Python filters; regenerated `.dta` (`e(N)` 150→100 balanced, 150→115
  unbalanced).
- **SE-gap root-caused** to `makerif2` full-sample IF rescaling vs OE per-cell
  IF (sample mismatch ruled out); test constants synced.

## HAC rollout caveat (v0.9)

`staggered_did()` HAC is a **project convention** — a Newey-West temporal
correction on the aggregated influence function — and is **not externally
validated** (staggered-DiD HAC is a contested area with no Stata/R reference).

- It raises a `UserWarning` and reduces exactly to the cluster-robust SE at
  `lags=0`.
- The other four estimators with HAC (`iv()`, `gmm()`, `did()`, `event_study()`)
  rest on canonical literature (validated against `statsmodels`/`R` `sandwich`).
- See the estimator docstring for the full caveat. **Do not present HAC coverage
  as symmetric across estimators.**
