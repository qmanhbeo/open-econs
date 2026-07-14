# CS2021 doubly-robust (DR) staggered-DiD estimator — IF rewrite notes

> Moved from `ROADMAP.md` during the 2026-07-13 trim. Technical detail behind
> the v0.7 `did_cs()` CS2021 DR estimator rewrite.

- **`_cell_dripw` reimplemented from R `DRDID` `drdid_panel`** (csdid default
  `dripw`); validated to machine precision vs csdid's saved per-entity RIF
  (per-cell corr 1.0; SE exact: cell (3,3)=0.4652265, (3,4)=0.4941999).
- **Full-sample weighted-RIF cluster SE** → aggregated **0.41781627**
  (balanced) / **0.62720813** (unbalanced), replacing the wrong
  `sqrt(mean se²)`.
  - This is validated against **csdid's own influence-function aggregation**,
    i.e. `csdid y x z, saverif(rif)` + `csdid_stats simple`, and the `did` R
    package's `aggte(type="simple")` (`getSE` = `sqrt(mean(if²)/n)` =
    `sqrt(Σ if_i²/N²)`). All three agree to machine precision.
  - **IMPORTANT — `csdid_estat simple` is NOT the reference.** In the installed
    csdid version (v1.6/v1.58) `csdid_estat simple` is buggy: it posts the raw
    per-(g,t) VCoV and prints element [1,1] (the *first, pre-treatment* cell's
    SE: 0.7479047 balanced, 0.47824472 unbalanced) as the "simple" ATT SE —
    which is not an aggregation SE at all. Do not compare OE's aggregated SE
    against `csdid_estat simple`; use `csdid_stats` (or the `did` R package)
    instead.
- Parity now holds at **rtol=1e-6** (was 0.2/0.6); all 18 staggered-DiD tests
  pass. References: see README.
