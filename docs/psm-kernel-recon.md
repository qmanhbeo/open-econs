# PSM kernel / smooth-weight matching — investigation (deferred)

> Moved from `ROADMAP.md` during the 2026-07-13 trim. This is the source of
> truth for the deferred `psm()` kernel-matching feature. Status: **investigated
> 2026-07-11, deliberately NOT built in v0.8** (deferred to its own scoped,
> parity-validated pass). Classification: (c) investigated, deliberately
> deferred.

- **Reference implementation:** `psmatch2` (Leuven & Sianesi, SSC) is the
  classic/applied-standard kernel matcher and directly supports `kernel` on the
  PS (`psmatch2 treat covariates, kernel kerneltype(epan) bw(#) outcome(y) ate`
  — confirmed live, ATT = −31.07, S.E. = 22.53 on cattaneo2). `kmatch` (Ben
  Jann, SSC) is the modern alternative (richer options, better-documented
  variance). **Both are viable parity references for the point estimate**;
  `psmatch2` is the more "expected" one in applied micro. Confirmed live that
  `teffects psmatch` does **not** accept `kernel` (r(198)), and neither does
  `teffects nnmatch` in this Stata 17 — so `teffects` (the command `psm()` is
  validated against for NN) cannot be the kernel reference. Installed `kmatch`
  is 1.1.5 (only `kmatch md ..., kernel()`; `psmatch2` runs out of the box).
- **Variance (read from source) — and why the SE is the hard part:** all three
  Stata commands compute the *same* kernel-on-PS point estimate but **diverge
  on the SE**, and none matches OE's `psm()` standard:
  - `psmatch2, kernel` → Abadie–Imbens (2002) influence function, **explicitly
    without** the PS-estimation correction (its own output note: "S.E. does not
    take into account that the propensity score is estimated") — known to
    *understate* SEs.
  - `kmatch` → influence function with weights assumed *fixed* (Jann 2019);
    `kmatch.sthlp` states analytic SEs are generally *conservative* and
    recommends `teffects`/bootstrap for consistent SEs.
  - `teffects psmatch` (discrete NN, what `psm()` is validated against) →
    **full AI-2012**, including the `c'_τ V_γ c_τ` PS-estimation adjustment
    (`psm.py:384,428,437`).
  - **Consequence:** since `teffects` can't do kernel, ANY kernel reference
    (psmatch2 *or* kmatch) leaves a gap — to stay consistent with OE's
    teffects-equivalent standard the kernel variance needs *both* the
    continuous-weight `K/K'` generalization *and* the PS-estimation adjustment,
    which neither user command provides by default. The point estimate is
    reusable; the variance is a genuine build.
- **The continuous-weight trap (the reason it is not a bolt-on):** OE's `psm()`
  variance (`psm.py`) is built on the discrete NN with-replacement count
  structure — `K_m(i)` (times matched) and `K'_m(i) = Σ_{j: i∈Ω(j)} 1/|Ω(j)|²`
  (the `K² + 2K − K'` term, `psm.py:382`). For kernel weights these counts are
  replaced by **weight-based aggregates** (total kernel weight supplied/received,
  and a normalized-squared-weight term). A naive "weighted average" plug-in is
  wrong — the `K'` normalization is what carries the AI-2012 variance over to
  continuous weights. This needs its own validated implementation, not an
  extension of the discrete `K/K'`.
- **Why deferred (not (a) reuse, not (b) built now):** it is a distinct
  estimator from the discrete-NN `psm()` variance, not a reuse; and bolting an
  un-validated continuous-weight variance onto the v0.8 close-out would violate
  the "built-and-validated or explicitly deferred" + "CI-green" standard. It
  deserves a dedicated pass with parity fixtures vs `kmatch` (and a decision on
  analytic-fixed-weights vs bootstrap).
