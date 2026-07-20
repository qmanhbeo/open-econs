# Future Work — Open & Deferred Items

This file tracks **open capability gaps and accepted deferrals** only. Each
item names a specific implementation path, not a vague exploration (rule 11),
and every parity claim must hold to ≤1e-6 (rule 2) with synth placebo excluded
from full runs (rule 5).

**Where the finished work went (rule 13/16):** closed convention decisions and
root-cause traces live in `methodology/<area>/<model>.md` — check there before
re-investigating "why X differs." Delivered features are recorded in
`CHANGELOG.md` and the git log. This file intentionally no longer duplicates
them. Completed performance vectorizations (psm Candidate C `cdb15be`, `_hac_S`
Candidate D `897c31a`) are recorded in `methodology/performance-conventions.md`
and `methodology/linear/gmm.md`; the GPU decline is in
`methodology/performance-conventions.md`.

---

## HC Estimators Beyond HC0–HC3

OE currently supports HC0–HC3. Three higher-order variants from R's
`sandwich::vcovHC` type parameter are not yet implemented. These are
OE-specific opt-in extensions beyond Stata parity — Stata does not offer
HC4/HC4m/HC5 natively.

### HC4 — Discounted Leverage (Cribari-Neto & da Mota 2017)

- **What:** HC4 applies a *discount factor* to the leverage adjustment,
  weighting the hat-matrix diagonal by `d_ii = h_ii / max(h_ii)` rather
  than using `1 - h_ii` directly. This downweights observations with
  moderate leverage more gently than HC3, which can be overconservative.
- **Reference implementation:** R `sandwich::vcovHC(model, type = "HC4")`.
- **Where in OE:** New function `hc4_cov()` in `open_econs/core/cov.py`.
  Signature: `hc4_cov(residuals, X, hat_matrix) -> np.ndarray`. Same
  interface as existing `white_cov()`.
- **Formula:** `Ω_HC4 = (1/(n-k)) Σ d_ii · e_i² · x_i x_i'` where
  `d_ii = min(1, h_ii / max(h_ii))`.
- **Parity anchor:** R `sandwich::vcovHC` with `type = "HC4"` on a known
  dataset. No Stata equivalent.

### HC4m — Modified HC4 (Cribari-Neto & da Mota 2017)

- **What:** Variant of HC4 that replaces the discount factor with
  `d_ii = h_ii / h̄` (where `h̄` is the average hat diagonal), capped
  at some threshold. More conservative than HC4 when leverage is
  concentrated in a few observations.
- **Reference implementation:** R `sandwich::vcovHC(model, type = "HC4m")`.
- **Where in OE:** Add `hc4m_cov()` to `open_econs/core/cov.py`.
- **Formula:** `Ω_HC4m = (1/(n-k)) Σ d_ii · e_i² · x_i x_i'` where
  `d_ii = min(h_ii / h̄, m)` for some tuning constant `m`.
- **Parity anchor:** R `sandwich::vcovHC` with `type = "HC4m"`.

### HC5 — Robust to Leverage (Cribari-Neto & da Mota 2017)

- **What:** Extends HC4 by using a different discount weighting that is
  robust to the presence of high-leverage observations even when the
  design matrix is nearly singular.
- **Reference implementation:** R `sandwich::vcovHC(model, type = "HC5")`.
- **Where in OE:** Add `hc5_cov()` to `open_econs/core/cov.py`.
- **Parity anchor:** R `sandwich::vcovHC` with `type = "HC5"`.

---

## HAC Kernel Variants

OE's Newey-West implementation currently uses the Bartlett (triangular)
kernel, which is the default in Stata and R. Four other kernels are
available in R's `sandwich::NeweyWest` but not in OE.

### Parzen Kernel

- **What:** Replace Bartlett weights with Parzen weights:
  `w(j) = 1 - 6(j/L)² + 6|j/L|³` for `|j/L| ≤ 0.5`,
  `w(j) = 2(1 - |j/L|)³` for `0.5 < |j/L| ≤ 1`.
- **Reference:** R `sandwich::NeweyWest` with `kernel = "Parzen"`.
- **Where in OE:** Add `kernel=` parameter to `newey_west_cov()` in
  `open_econs/core/cov.py`.
- **Parity anchor:** R `sandwich::NeweyWest(kernel = "Parzen")`.

### Quadratic Spectral (QS) Kernel

- **What:** Replace Bartlett with QS kernel:
  `w(j) = (25/(12π²(j/L)²)) · (sin(6π(j/L)/(5))/(6π(j/L)/(5)) -
  cos(6π(j/L)/(5)))`. Andrews (1991) optimal kernel.
- **Reference:** R `sandwich::NeweyWest` with `kernel = "QS"`.
- **Parity anchor:** R `sandwich::NeweyWest(kernel = "QS")`.

### Truncated Kernel

- **What:** Simple truncation: `w(j) = 1` if `|j| ≤ L`, else `0`.
- **Reference:** R `sandwich::NeweyWest` with `kernel = "Truncated"`.
- **Parity anchor:** R `sandwich::NeweyWest(kernel = "Truncated")`.

### Tukey-Hanning Kernel

- **What:** `w(j) = (1 + cos(π j/L)) / 2`.
- **Reference:** R `sandwich::NeweyWest` with `kernel = "TukeyHanning"`.
- **Parity anchor:** R `sandwich::NeweyWest(kernel = "TukeyHanning")`.

---

## pyfixest Backend Extensions

### pyfixest `nid` (Non-IID) VCE

- **What:** pyfixest supports `vcov = {"CRV1": cluster}` and
  `vcov = "iid"` / `"hetero"` natively, plus a `"nid"` option for
  non-iid inference that is not available in OE's current pyfixest
  adapter paths.
- **Reference:** pyfixest `feols()` `vcov` parameter documentation.
- **Where in OE:** Add `cov_type="NID"` or similar to the pyfixest
  adapter path in `ols.py` / `fe.py`.
- **Decision required:** Whether "nid" is a distinct cov_type or a
  modifier on existing types. Needs investigation of pyfixest's actual
  implementation to determine the right API surface.

### pyfixest CRV3 (HC3-Style Clustered SEs)

- **What:** pyfixest supports `vcov = {"CRV3": cluster}` which applies
  HC3-style leverage correction within clusters. OE currently only routes
  CRV1 through pyfixest; CRV3 is available on the pyfixest side but
  OE's adapter does not expose it.
- **Reference:** pyfixest `feols()` with `vcov = {"CRV3": cluster}`.
- **Where in OE:** Add CRV3 as an option in the cluster path of
  `fe.py` and `ols.py` adapters.
- **Parity anchor:** Stata does not have CRV3 directly; this is a
  pyfixest/R-specific estimator. Cross-check against R `fixest` with
  `vcov = ~cluster`.

### SSC (Small-Sample Correction) Toggle

- **What:** OE's pyfixest adapters currently let pyfixest apply its
  default SSC unconditionally. There is no user-facing toggle to disable
  the `(N-1)/(N-K)` factor or the `G/(G-1)` cluster correction.
  This matters for users who want to match specific Stata variants
  (e.g., `ivregress, cluster` without `small` applies no SSC).
- **Reference:** Stata `ivregress` default vs `small` variant;
  pyfixest `feols()` default behavior.
- **Where in OE:** Add `small_sample_correction: bool = True` parameter
  to `ols()`, `fe()`, and `iv()`.
- **Decision required:** How this interacts with the existing IV cluster
  backend exception (linearmodels `debiased=False`). May need a
  cleaner API than a boolean toggle.

---

## Related Items (Not Covariance, But Identified During Integration)

### iv() Cluster Backend Reconciliation

- **What:** The current `iv()` adapter routes cluster-robust IV through
  linearmodels (`debiased=False`) instead of pyfixest's CRV1, because
  pyfixest's SSC does not match Stata's `ivregress` default. If the SSC
  toggle (above) is implemented, this backend exception could potentially
  be eliminated by using pyfixest with `small=False`.
- **Status:** Blocked on SSC toggle implementation and validation.

### iv() Option-Coverage Audit (rule 15) — PARTIAL 2026-07-17

- **Scope:** `iv()` Stata/R parity across `cov_type` (nonrobust, robust/HC1,
  HC0/HC2/HC3, HAC), `cluster`, `lags`/`time`/`hac_adjust`, FE
  (`entity`/`time_fe`/`fixed_effects`).
- **New fixtures (2026-07-17):** `df_iv_panel.csv` (500 obs, 50 id x 10 t,
  overidentified 2 instruments z1/z2, exog w, id/time) enables robust/cluster/
  HAC/FE parity that the old just-identified 1-instrument fixtures could not
  exercise.  Stata fixtures `iv_robust.dta` (vce(robust)) and
  `iv_cluster_panel.dta` (vce(cluster id)) committed; their `.do` generators
  in `tests/stata/generate-fixtures/`.
- **Stata parity — ACHIEVED (≤1e-6, tested in test_stata_iv.py):**
  - `nonrobust` (just-id): TestIVBasic.
  - `robust`/HC1 (over-id): TestIVRobustOveridentified — matches
    `ivregress 2sls, vce(robust)` coef+SE exactly.
  - `cluster` (just-id + over-id): TestIVCluster + TestIVClusterOveridentified
    — match `ivregress 2sls, vce(cluster ...)` coef+SE exactly.
- **Stata parity — GAPS (not testable via `ivregress`):**
  - **HAC:** Stata `ivregress 2sls, vce(hac bartlett L)` returns rc=111 (not
    supported).  HAC IV in Stata must go through `newey`/a user command;
    `ivregress` has no HAC VCE.  So iv() HAC parity is R-reference only
    (linearmodels/sandwich), NOT Stata.  Documented as a convention boundary.
  - **FE:** `ivregress` has no `absorb`; FE IV is `xtivreg, fe`.  Coefficients
    match `xtivreg, fe` exactly (within-transform drops _cons).  **SEs:** RESOLVED
    (2026-07-17).  Stata `vce(robust)` is cluster-robust by entity id, not an HC
    estimator; OE routes FE-robust to pyfixest `CRV1` by entity via `fe_robust="xtivreg"`
    (default) — matches Stata to ≤1e-6 for both `nonrobust` and `robust`.  See
    `iv() FE SE DOF` below (RESOLVED).
- **R parity — ACHIEVED (≤1e-6, tested in tests/r/tests/test_r_iv.py, 2026-07-17):**
  - `AER` installed (R 4.6.1).  `ivreg2` is **UNAVAILABLE** for R 4.6.1
    (removed from CRAN) — parity established against `AER::ivreg` +
    `sandwich` (`vcovHC`/`vcovCL`) instead, which is the standard R IV
    reference.  Fixture `iv.json` generated by `tests/r/generate-fixtures/iv.R`.
  - **Stata-vs-R SE divergence (source-confirmed):** Stata `ivregress` uses
    `s2 = SSR/N` (nonrobust + robust) and **no** cluster SSC; R `AER`/`sandwich`
    uses `s2 = SSR/(N-K)` and the `G/(G-1)` cluster SSC.  Both are legitimately
    "HC1"/"robust" but differ by √(N/(N-K)) / √(G/(G-1)).  OE exposes this as
    the **`debiased`** toggle (`iv()`): default `False` = Stata (preserves Stata
    parity), `debiased=True` = R.  The R test suite passes `debiased=True` and
    asserts coef + SE ≤1e-6 (residual ~2.5e-7 is AER/sandwich vs linearmodels
    impl noise, within rule 2).  Coefficient matches R exactly for every
    cov_type (no toggle needed).
- **HC0/HC2/HC3:** RESOLVED (2026-07-17).  `iv()` now hand-rolls the
  MacKinnon-White IV sandwich (`_iv_hc_sandwich` in `iv.py`) and matches R
  `sandwich::vcovHC(type="HC0"/"HC2"/"HC3")` to **machine precision**
  (≤1e-16) on the `iv_input.csv` fixture, with parity tests
  `TestIvRHc0/TestIvRHc2/TestIvRHc3`.  Root cause of the prior 1.95e-6 HC3
  gap: two formula errors, now fixed (see methodology/linear/iv_2sls.md):
  (a) the meat must use the **projected regressors** `Xp = P_Z X`, not raw X;
  (b) the leverage `h` must be `diag(X (Xp'Xp)^{-1} X' Z (Z'Z)^{-1} Z')`
  (R `AER::hatvalues.ivreg`), and must **NOT be clipped at 0** — R's leverage
  can be slightly negative (min ≈ -1.2e-3) and clipping it broke the
  `1/(1-h)^2` HC3 hardening.  Stata `ivregress` has no HC0/HC2/HC3 VCE, so
  parity is R-reference only (rule 14/15).

> **iv() FE SE DOF reconciliation — RESOLVED** (2026-07-17); full write-up in
> `methodology/linear/iv_2sls.md` (Stata `xtivreg, fe vce(robust)` is
> cluster-robust by entity id; OE `fe_robust="xtivreg"` toggle matches ≤1e-6;
> diagnostic `.do` archived). Moved out of the open queue.

---

## did() Phase — Backend and Extension Items

### D10: did()/event_study() pyfixest Backend Swap (Deprioritized)

- **What:** Route `did()` and `event_study()` through pyfixest for
  maintenance consistency. Both currently use statsmodels OLS, which
  produces correct results with full Stata parity. A pyfixest swap here
  is cosmetic (same OLS math, different library) with real regression
  risk for no user-facing benefit.
- **Status:** Approved as deprioritized. Do not pursue unless the user
  explicitly wants to unify all estimators on pyfixest. The benefit is
  marginal; the risk is real.
- **Decision:** D10 (2026-07-14).

> **Delivered DID extensions moved out (see CHANGELOG + git + methodology):**
> D11 `did_cs().aggte()` (dynamic/group/calendar), D12 `did_gardner()`
> (Gardner 2022 DID2S, `methodology/causal_inference/did_gardner.md`), and D13
> `did_sa()` (Sun & Abraham 2021, `methodology/causal_inference/did_sa.md`) are
> all delivered with R parity ≤1e-6. The only remaining D13 open question — a
> Stata anchor for Sun-Abraham (`csdid` w/ `aggte(type="simple")` or
> `eventstudyinteract`) — is folded into the did_sa methodology note as a
> deferred decision.

---

## Time-Series Backend (v1.1.0) — Flagged Gaps

Two parity gaps in `open_econs/models/timeseries/` were identified during the
v1.1.0 finish-line but intentionally **not** resolved this session (judgment
calls 1–3 from the handoff). They are documented here, not papered over with
relaxed tolerances (standing rule 2).

### TS-1: Legacy CV-Table Parity (Fuller / ERS / ZA small-N tables)

- **What:** OE's unit-root / stationarity tests surface the backend's *native*
  critical-value tables (arch: MacKinnon 2010 for ADF/PP/DF-GLS, Hobijn et al.
  2004 for KPSS, arch 100k-rep Monte Carlo for ZA). Stata prints Fuller (1976)
  for ADF/PP and ERS (1996) / Cheung-Lai (1995) for DF-GLS; R `urca` prints
  Kwiatkowski et al. (1992) for KPSS and Zivot-Andrews (1992) for ZA. In small
  samples these tables diverge, so the CV vintage is *labelled* (never hidden)
  in every `summary()` (decision 1).
- **Why flagged:** replicating the exact small-N tables is a finite-sample
  table-reproduction exercise, not a numerical-method fix. The asymptotic test
  is identical across tools; only the tabulated quantiles differ.
- **Implementation path:** port the explicit tabulated quantiles for
  Fuller (1976) ADF, ERS (1996) DF-GLS, and ZA (1992) so OE can optionally
  emit the Stata/R-vintage CVs alongside (or instead of) the MacKinnon/Hobijn
  tables. Keep the `cv_vintage` label as the source of truth.
- **Parity anchor today:** the **MacKinnon (1994) p-value** is the one
  cross-tool point of genuine agreement and is asserted against Stata in
  `tests/stata/tests/test_timeseries_adf_pp.py`; the statistic is asserted
  against R (`urca`) in `tests/r/tests/test_timeseries_urca.py`.
- **XFAIL anchors (rule 22):** the CV divergence itself is now pinned with
  `xfail(strict=True)` tests asserting OE's 5% ADF critical value against the
  Stata Fuller CV (`test_timeseries_adf_pp.py::TestADFStata::`
  `test_cv5_matches_stata_fuller`) and the R `urca` CV
  (`test_timeseries_urca.py::TestADFCriticalValueVintageGap`). They flip to
  xpass when the Fuller/ERS vintage tables are ported. See
  `methodology/timeseries/unitroot_cv_and_dfgls.md`.

### TS-2: DFGLS Lag-Selection Port (Ng-Perron vs arch AIC)

- **What:** OE's `dfgls()` uses arch's default AIC lag selection on the
  GLS-detrended series. Stata `dfgls` uses Ng-Perron sequential-t / SIC / MAIC.
  The GLS detrending (ERS cbar) and the default max-lag *ceiling* match
  exactly, but the lag-selection *method* differs by design.
- **Why flagged:** porting Ng-Perron sequential testing is a non-trivial
  method implementation, out of scope for the v1.1.0 finish-line. The reported
  DF-GLS statistic therefore cannot be asserted equal to Stata's, and the
  current tests deliberately do **not** cross-check the statistic against
  Stata (no relaxed tolerance).
- **Implementation path:** implement Ng-Perron sequential-t / SIC / MAIC
  inside `dfgls()` (new `method="ng-perron"` option), gated behind an explicit
  argument so the arch-AIC default is preserved. The GLS-detrending and
  max-lag-ceiling code is shared with arch and already verified.
- **Parity anchor today:** `tests/non_stata_nor_r/test_timeseries_dfgls.py`
  asserts an exact OE-vs-arch **backend identity** (same args → identical
  statistic) plus a same-backend CV-vintage regression guard. The Stata gap is
  tracked here until the port lands.
- **XFAIL anchors (rule 22):** the same file's `TestDFGLSStataLagSelectionGap`
  now pins the Stata gap with `xfail(strict=True)` tests asserting OE's
  AIC-selected lag / statistic against Stata's Ng-Perron SIC/MAIC reference
  (lag 1, DF-GLS mu = -1.1362432, source-verified via `ur_dfgls_c.do`). They
  flip to xpass when `method="ng-perron"` lands. Footgun: Stata's *seq-t* rule
  coincidentally picks lag 0 (= AIC) on this series, so the assertion targets
  SIC/MAIC, not seq-t — see `methodology/timeseries/unitroot_cv_and_dfgls.md`.

---

## GMM Convention Differences (Documented, Not Bugs)

> **Closed convention decisions moved to `methodology/linear/gmm.md`**
> (Root-Cause Knowledge section, rule 16): the J-statistic `/sig2` split
> (GMM-J), the HAC kernel-scope convention (GMM-HAC), the R `cluster=` NO-OP
> (GMM-RCLUSTER), the Windmeijer + `robust_meat` conventions (GMM-WC), and the
> Stata expression-form weighting matrix (GMM-GN). Only the open IVGMM
> wrap-candidate remains below.

### GMM-IVGMM: linearmodels.IVGMM Wrap Candidate (Rule 14)

- **What:** `linearmodels>=6.0` is already a core dependency (v7.0 installed).
  `linearmodels.iv.model.IVGMM` exists with `centered=True` default.  Zero
  current references in the codebase — OE uses only `IV2SLS`.  Candidate for
  future evaluation of whether to wrap `IVGMM` instead of hand-rolling
  `_gmm_core.py`.
- **Status:** Checked 2026-07-17 per rule 14.  No formula/citation evidence
  found there resolving the J-convention question.  Not actioned now.
- **No action required** unless `_gmm_core.py` needs replacement.

---

## ABOND R-Parity (DEFERRED — plm `pgmm` broken; Stata parity complete, accepted as sufficient)

- **Decision (2026-07-17):** R `plm::pgmm` parity for `oe.abond()` is
  **intentionally DEFERRED / accepted-incomplete**.  Stata `xtabond2` parity is
  fully covered (52 tests: 40 difference-GMM + 12 system-GMM, all green) and is
  the primary anchor for abond's users.  R parity adds little given low abond
  usage and is currently impossible to verify (plm `pgmm` is broken upstream).
  This is a documented acceptance, NOT an open "todo" — do not treat it as
  pending work unless plm is fixed.

- **What (rule 15 gap):** `oe.abond()` has full Stata `xtabond2` parity
  (`tests/stata/tests/test_stata_abond.py`: 8 difference-GMM flavors × 5
  cross-checks = 40 tests, plus 4 system-GMM flavors × 3 cross-checks = 12
  tests; all green) but **no R parity**.  The canonical R anchor for
  Arellano-Bond difference GMM is `plm::pgmm`.
- **Blocker (verified 2026-07-17):** `plm::pgmm` is **broken in this environment
  on BOTH installed R versions** — R 4.6.1 AND R 4.5.2 (plm freshly installed
  into the 4.5.2 user library).  It errors inside plm itself, NOT in our code,
  and fails on the **canonical plm `EmplUK` example** for `effect="twoways"` and
  `effect="individual"` alike.  Tested across **plm 2.6.3, 2.6.4, and 2.6.7** —
  all fail identically.  A third R install, **R 2.5.0** (2007), was also checked
  and is **non-viable**: CRAN no longer serves packages for it (no
  `bin/windows/contrib/2.5`, and its network stack cannot fetch the modern
  `src/contrib` index), and any `plm` that ran on R 2.5.0 predates the modern
  `pgmm` two-part-formula API.  Base `plm` (e.g. `plm(..., model="within")`) works, so
  the breakage is specific to `pgmm`.  **This is a plm library bug, not an
  R-version or OE issue** — installing R 4.5.2 does NOT help.
- **Exact source-level diagnosis (from `deparse(pgmm)`, plm 2.6.4):**
  - `effect="twoways", transformation="d"`: builds `V1 <- diff(diag(1, T-TL1+1))[, -1]`
    which is `(T-TL1)×(T-TL1)`, then does `yX1[[i]] <- cbind(yX1[[i]], V1)` where
    `yX1[[i]]` has **T** rows → "number of rows of matrices must match".  The
    time-demeaning matrix is dimensioned for the *transformed* series but cbind'd
    onto the *untransformed* entity model matrix.  Long-standing plm bug.
  - `effect="individual"`: fails earlier at `W1[[i]] <- cbind(W1[[i]], Z1[[i]])`
    (the `normal.instruments` branch) with the same row-mismatch, and a separate
    `seq_len(TL1-1)` NA error when all instruments are GMM.  `TL1` (time-lag-1)
    is computed as `NA` in this code path.
  - Conclusion: `pgmm` is non-functional in this plm/R combination for any
    standard Arellano-Bond call.
- **Impact:** Cannot generate a committed R `abond.json` fixture via pgmm.
  Therefore no `tests/r/tests/test_r_abond.py` can be added until `pgmm` is fixed
  upstream or a working R environment exists.
- **Remediation options (rule 3 — flag, do not silently skip):**
  1. **Upstream plm fix** — wait for / contribute a plm patch (the `cbind` row
     mismatch is the concrete bug to fix), then regenerate on R 4.5.2+.
  2. **CI on a plm-supported R** — even R 4.5.2 fails here, so this needs a R
     version where plm's `pgmm` was last known-good (pre-2.6.x, or a patched
     build).  R 4.5.2 alone is NOT sufficient.
  3. **Do NOT hand-roll `pgmm` as the R anchor** — a hand-rolled R script would
     re-implement OE's own `_gmm_core.py`+`abond.py` math, which is NOT an
     independent reference implementation (violates rule 1's "verify against
     source" intent).  Only use a hand-roll as a last resort and label it
     explicitly as a self-consistency check, not parity.
- **Status:** DEFERRED (accepted).  Stata `xtabond2` parity is complete and
  authoritative (difference GMM **and** Blundell-Bond system GMM, all four
  flavors matching AR tests to < 1e-7); R parity is intentionally not covered.
  The simulation-only
  `tests/non_stata_nor_r/test_abond.py` stays as the only non-Stata abond
  coverage; it is NOT a parity anchor (rule 7: it belongs in the deferred-
  migration bucket, not in `tests/r/`).
- **Next agent:** do NOT attempt to add R abond parity unless plm's `pgmm` is
  fixed upstream AND a known-good plm/R combo is available.  It is an explicit
  accepted deferral, not pending work.  If revisited, mirror the Stata
  fixture's 8 flavors (collapsed/non-collapsed × one/two-step × robust/
  non-robust) on the same `df_panel.csv`, using `pgmm(..., effect="twoways",
  transformation="d")` with a
  two-part formula `y | lag(y,-1)+lag(x,0)+lag(z,0) ~ lag(y,-2:-4)+lag(x,0)+lag(z,0)`
  to match Stata's `gmm(L.y, lag(2 4)) iv(x z)`.  Also update `r_runner.R_EXE`/
  docs if a second R is added.

---

## Candidate A — synth analytic gradient for SLSQP: DEFERRED (blocked)

- **What:** `synth.py` `_optimize_v` runs SLSQP on `_fn_v`; ~60% of the cProfile
  time is `scipy`'s numeric `approx_derivative` of the V objective. An analytic
  gradient would speed up the inner V-optim substantially (compounds across the
  permutation loops in `placebo.py`).  See the 2026-07-17 inspection plan.
- **Blocker:** Supplying an analytic gradient changes the SLSQP convergence
  path and **risks diverging from R `Synth`'s `optimx` local optimum** — `synth`
  V-optim is multi-modal (known hazard for d0/d1/d5/d7 fixtures). Requires a
  **full R `Synth` re-parity pass** (all permutation fixtures) before merge.
  That budget was not authorized in the Candidate-B session.
- **Status:** DEFERRED. Do NOT implement until a dedicated R re-parity session
  is scoped. Flagged per rule 3/15 (disparity = optimizer path; expose only
  behind a toggle if ever implemented, and cover both settings in parity tests).
- **Next agent:** treat as blocked, not pending, unless the project lead budgets
  the R `Synth` re-parity pass.

---

## Open follow-up — audit Stata `.do` generators for `set type double` (rule 18)

- **What:** The ARDL parity work uncovered that Stata `import delimited` reads
  numeric columns as single-precision `float` by default, which silently
  truncated near-collinear inputs and produced a spurious ~1e-5 parity gap.
  Fixed for `ardl.do` with `set type double`. **Other `.do` generators that
  `import delimited` a CSV and then run a regression may carry the same latent
  footgun** — it only bites when inputs are ill-conditioned enough that float
  truncation exceeds 1e-6, so existing passing fixtures are not necessarily
  safe under future data changes.
- **Scope:** grep `tests/stata/generate-fixtures/*.do` for `import delimited`
  without a preceding `set type double`; add it defensively and re-run the drift
  check. Low risk, mechanical; good subagent task (rule 9).
- **Status:** CLOSED (2026-07-20). All 63 `*.do` generators that `import
  delimited` now carry `set type double` immediately before the import (5 were
  already correct: `ardl.do`, `nbreg.do`, `ordered.do`, `poisson.do`,
  `tobit.do`; 58 were fixed). The drift check was re-run and no committed
  `.dta` fixture changed (current inputs are well-conditioned; the footgun only
  bites ill-conditioned data, per rule 6 any future drift would be investigated,
  not absorbed).
- **Context:** root cause fully written up in `methodology/timeseries/ardl.md`
  and the inline comment block of `tests/stata/generate-fixtures/ardl.do`.

---

*Last updated: 2026-07-18. Reorganized: closed/resolved root causes moved to
`methodology/<area>/<model>.md` (rule 16) and delivered features dropped (see
git + CHANGELOG); this file now tracks OPEN + accepted-deferred work only.
Latest release: v1.4.2 — patch release closing ROBUST-REG-STATA (quantile
regression `quantile_reg` + outlier-robust regression `robust_reg` shipped in
v1.4.0; ROBUST-REG-STATA Stata `rreg` coef+SE parity closed to <3e-10 in
v1.4.2). ARDL write-up in
`methodology/timeseries/ardl.md`; follow-up `set type double` audit queued above.*

---

## v1.2 poisson: ppmlhdfe non-clustered robust SE (RESOLVED — 2026-07-20)

- **What:** `oe.poisson(..., vcov_backend="stata", cluster=None)` (Stata
  `ppmlhdfe` non-clustered default) does **not** reproduce ppmlhdfe's SE to
  1e-6 via fixest/pyfixest. ppmlhdfe reports a *robust (sandwich) SE*, not an
  OIM iid SE, and its robust factor uses the Correia-Guimaraes-Zylkin (2019)
  nonlinear Poisson adjustment. fixest/pyfixest `ssc(k_adj=False, G_adj=True,
  k_fixef="none")` reproduces it only to ~4e-4 (x2: OE 0.03967 vs ppmlhdfe
  0.04183), even via R `fixest::fepois` directly.
- **Resolution (option a):** wrapped ppmlhdfe's exact robust bread in
  `open_econs/models/limited/poisson.py::_ppmlhdfe_robust_vcov`. For
  `vcov_backend="stata"` with no `cluster=`, the SE is now
  `sqrt((N-1)/(N-K)) * sqrt(diag( bread @ meat @ bread ))` with
  `meat = Σ_i (y_i − μ_i)^2 / μ_i · x_i x_i'` and `bread = (X'WX)^{-1}` (`X`
  = FE-residualized regressors, `W = diag(μ)`). This matches ppmlhdfe's
  non-clustered SE to ≤2e-7 absolute on the `poisson` fixture
  (x1 0.038951 / x2 0.041833). The `k_adj = (N-1)/(N-K)` factor is what
  ppmlhdfe applies to its *non-clustered* robust SE (it does NOT apply it to
  the cluster-robust SE, which keeps `G_adj` only).
- **In scope / passing:** the **cluster-robust** SE — the headline PPML use
  case — matches ppmlhdfe to 1e-6 (`vcov_backend="stata"` + `cluster=...`).
  Verified: x1 0.041178 / x2 0.047180 (= ppmlhdfe). Point estimates, deviance,
  and log-pseudolikelihood match to 1e-6 across Stata/R/pyfixest.
- **Where OE lives:** `open_econs/models/limited/poisson.py`; root cause written
  up in `methodology/limited/poisson.md` §2.4.
- **Status:** RESOLVED. `tests/stata/tests/test_stata_poisson.py::
  TestStataPoissonIidSE` now passes to 1e-6 (xfail removed, real pass). The
  fixest backend (default) is untouched.

---

## v1.2 ologit/oprobit: Stata-R coefficient divergence & robust-SE gap (OPEN parity gaps)

- **What (a):** Stata ologit/oprobit and R MASS::polr MLEs differ by
  ~1e-5 on **coefficients and cutpoints** (independent optimizer convergence;
  both report the *same* log-likelihood to 1e-8 and the *same* OIM SEs to 1e-7,
  so it is precision, not a formula difference). OE anchors to Stata (the
  project's primary reference) and matches it to 1e-6 after an L-BFGS-B polish
  pass on the statsmodels log-likelihood (gtol=1e-12, ftol=1e-14). R
  coef/cutpoint assertions are skip-ped at 1e-6 with the exact magnitude
  recorded (rule 15). R log-likelihood and OIM SE match OE to 1e-6 and ARE
  asserted.
- **What (b):** oe.ologit(..., cov_type="HC1") (and HC0/HC2/HC3) robust SEs
  diverge from Stata ologit, vce(robust) by ~4e-4 ? same root cause as the
  poisson non-clustered robust gap: numerical-score bread vs Stata's exact OIM
  bread + Stata's small-sample factor. OIM (nonrobust) SE matches Stata to
  1e-6 (the validated deliverable). Robust SE assertions were skip-ped in
  	ests/stata/tests/test_stata_ordered.py::TestStataOrderedRobustSE.
- **Cutpoint sign:** task brief assumed polr/OrderedModel negates Stata's
  cutpoints ? **FALSE** (source-verified). All three store cumulative,
  increasing thresholds with P(Y<=j)=F(c_j - x'?) and the SAME sign. OE stores
  Stata convention; no negation. See methodology/limited/ordered.md (rule 18
  footgun).
- **Where OE lives:** open_econs/models/limited/ordered.py; OrderedResult
  in open_econs/core/results.py; root cause in methodology/limited/ordered.md.
- **Status:** (b) RESOLVED 2026-07-20. The robust bread now uses Stata's exact
  OIM bread `inv(-H)` with the EXACT analytical observation scores over the
  full `(?, cut1..)` vector in Stata's cumulative-cutpoint parameterization
  (Jacobian-transformed from statsmodels' incremental-exponential threshold
  params), and Stata's `n/(n-1)` small-sample factor for HC1. `oe.ologit("y ~
  x1+x2+x3", cov_type="HC1")` now matches Stata `ologit, vce(robust)` to
  <=1e-6 (x1 5.8e-9, x2 2.8e-9, x3 1.3e-8). The `xfail(strict=True)` is
  removed and `TestStataOrderedRobustSE` is a real passing test. OIM (nonrobust)
  SE is unchanged and still matches to 1e-6. (a) remains OPEN (Stata-anchored
  engine-difference, accepted).
- **Next agent:** (b) is closed; do not regress it. Stata coef/cutpoint/OIM-SE
  parity remains the shipped deliverable.

---

## v1.2 tobit: robust/cluster SE open gap (RESOLVED — 2026-07-20)

- **What:** oe.tobit(..., cov_type="HC0"|"HC1"|"HC2"|"HC3") and
  cluster= robust SEs are computed by a numerical-score sandwich (per-obs
  score via central numeric diff of the (beta, ln sigma) log-likelihood). They
  diverge from Stata 	obit's *exact* OIM-robust bread by ~1e-4 (same class of
  issue as poisson/ologit). The **OIM (nonrobust)** SE � the validated
  deliverable � matches Stata/R AER::tobit to 1e-6.
- **In scope / passing:** OIM SE, point estimates, sigma (cross-checked against
  Stata's ar(e.y)=sigma^2 5th e(b) element and R's summary()),
  Log(scale)=ln sigma, and log-likelihood all match Stata & R to ~1e-9.
- **Where OE lives:** open_econs/models/limited/tobit.py; OIM via
  inv(approx_hess(nll)) on (beta, sigma); robust via _sandwich_cov.
  Root cause in methodology/limited/tobit.md A3.
- **Status:** RESOLVED 2026-07-20. `_sandwich_cov` now uses the exact analytic
  Tobit score contributions (incl. censored regions) on `(beta, sigma)` with the
  OIM `inv(Hessian)` bread. Stata `vce(robust)` = `(n/(n-1)) · OIM_bread ·
  OPG_meat · OIM_bread`; `vce(cluster <id>)` = `(G/(G-1)) · OIM_bread ·
  (Σ_g g_s g_s') · OIM_bread` (both verified to 1e-10). OE exposes
  `vce(robust)` as `cov_type="HC1"`. Robust/cluster SEs asserted at 1e-6 in
  `tests/stata/tests/test_stata_tobit.py::TestStataTobitRobustCluster`;
  `tobit.do` now emits `rse_*`/`cse_*` and `tobit_input.csv` has a 40-cluster
  `id` column. Next agent: no further work required.

---

## v1.2 nbreg: Stata `dispersion(constant)` MLE & non-clustered SE gaps (RESOLVED 2026-07-20)

- **What (a) — Stata `dispersion(constant)` is a Stata-specific NB2 MLE:**
  `oe.nbreg(dispersion="const")` implements the **textbook NB2 gamma mixture**
  (Var = μ + α·μ²), matching R `MASS::glm.nb` and R `fixest::fenegbin` to 1e-6
  on coefs / alpha=1/theta / log-likelihood. Stata's `nbreg, dispersion(constant)`
  fits a **different** MLE (Var = μ·(1+δ), δ = exp(lndelta)). This is now
  reproduced exactly via `oe.nbreg(dispersion="const_stata")`: coef x1 =
  0.414535, delta = 1.263565, LL = -842.203 (matches Stata to machine precision;
  source `nbreg_al.ado`). `const_stata` is **pooled-only**.
- **What (b) — non-clustered (OIM) SEs:** Stata `nbreg` non-clustered SEs use the
  inverse of the **full observed-information Hessian** (beta + aux jointly). This
  is now reproduced for `dispersion in ("const", "const_stata")` via
  `vcov_backend="stata"` (`_nb_oim_bread`). se_x2 matches Stata 0.059624 to
  ~3e-8. The cluster-robust (CRV1) SE — the standard NB use case — is also matched
  via this toggle.
- **In scope / passing:** NB2 pooled & FE coefficients, alpha/theta, and
  log-likelihood match R `glm.nb` + `fixest::fenegbin` AND Stata
  `nbreg, dispersion(mean)` to 1e-6. NB1 (`dispersion="mean"`) implemented per
  Hilbe. `vcov_backend` toggle (fixest / stata) mirrors poisson.
- **Where OE lives:** `open_econs/models/limited/nbreg.py`; `NegBinResult` in
  `open_econs/core/results.py`; fixtures `tests/r/fixtures/expected/nbreg.json`
  + `tests/stata/fixtures/expected/nbreg.dta`; generators
  `tests/{r,stata}/generate-fixtures/nbreg.{R,do}`; tests
  `tests/r/tests/test_r_nbreg.py`, `tests/stata/tests/test_stata_nbreg.py`,
  `tests/non_stata_nor_r/test_nbreg_backend.py`. Root cause in
  `methodology/limited/nbreg.md` §2.
- **Status:** RESOLVED (both). (a) `TestStataNBRegConstantDispersionGap.test_b_x1`
  now calls `dispersion="const_stata"` and passes to 1e-6 (xfail removed). (b)
  `TestStataNBRegStdErrors.test_se_x1/x2` now call `vcov_backend="stata"` and
  pass to 1e-6 (xfail removed). Default `vcov_backend="fixest"` still matches R
  `glm.nb` OIM to 1e-6. Do NOT loosen the validated coef/alpha/LL tolerances
  (rule 2).
- **Next agent:** treat as open; the textbook/R NB2 + Stata `dispersion(mean)`
  parity is the shipped deliverable. A future `dispersion="const_stata"` option
  could cover Stata's constant MLE if a user needs it (rule 15 toggle).


---

## v1.3 Diagnostics — open items

- **DFBETAS vs statsmodels convention divergence (RESOLVED 2026-07-20):**
  The previously reported ~8.6e-4 relative gap is GONE — the LOO-variance
  factor `1/(1-h_i)` was already fixed (see
  `tests/r/tests/test_r_diagnostics.py::test_dfbetas_gap_magnitude`), and
  statsmodels `OLSInfluence.dfbetas` in fact uses the *same* leave-one-out
  variance (`sigma2_not_obsi`), so oe's default `dfbetas()` matches both Stata/R
  and statsmodels to ~9e-14. A `dfbetas(backend="stata_r"|"statsmodels")` toggle
  (default `"stata_r"`, the authoritative Stata/R target) now exposes the
  convention choice per AGENTS.md rule 15. The non-Stata/non-R test
  `tests/non_stata_nor_r/test_diagnostics.py` validates the DEFAULT against the
  authoritative R `stats::dfbetas` fixture (not statsmodels) to ≤1e-6, and a
  separate test pins `backend="statsmodels"` against
  `OLSInfluence.dfbetas` to ≤1e-6. No xfail, no loosened tolerance. See
  methodology/linear/diagnostics.md.
- **diagnostics() vs diagnostics_table() (recommendation, not open):** keep
  both for v1.3.0 (the dict form is asserted by an existing test). In a future
  minor, flip diagnostics() to return the DataFrame and update the one dict
  test.
- **v1.3 diagnostics: all other parity targets met (BG from-scratch n*R2,
  White, Ljung-Box, Cook's D, leverage) — no other open items** beyond the
  DFBETAS/statsmodels gap above.

---

## ROBUST-REG-STATA - Stata rreg parity gap (RESOLVED 2026-07-19)

REWORKED 2026-07-19. oe.robust_reg now targets Stata `rreg` as the PRIMARY
parity default (parity="stata"); R MASS::rlm is a toggle (parity="rlm", exact
1e-6). The prior agent's default (R rlm method="MM") was REJECTED by the PM:
Stata `rreg` is a bisquare M-estimator (NOT MM), so the MM default diverged
from Stata at ~1e-3.

Status of parity="stata" (pure-Python, no R needed) — GAP CLOSED 2026-07-19:
- Coefficients match Stata e(b) to < 3e-10 (was ~1.2e-4).
- SEs (bias-correction OLS VCE V = (rss/(N-k)) (X_in' X_in)^{-1}) match Stata
  e(V) to < 3e-10 (was ~8e-4).
- Algorithm: faithful re-implementation of rreg.ado v3.5.0 (OLS -> Cook's-D drop
  -> Huber init k=1.345 -> bisquare IRLS c=4.685 with top-of-iteration MAD scale
  -> Stata bias-correction regress). The two parity-critical subtleties that
  closed the gap:
    1. The bias-correction step uses the LAST in-loop weight (not a fresh
       re-evaluation), preserving X'(w*resid)=0 so the correction is a no-op.
    2. lambda's N counts only non-zero-weight in-sample obs (Stata regress
       [aw=weight] drops zero-weight outliers -> N_eff=192, not 200).

The strict 1e-6 coef AND se assertions in tests/stata/tests/test_stata_rreg.py
now PASS (xfail removed). Root cause and fix documented in
methodology/linear/robust_reg.md. No further work required.

