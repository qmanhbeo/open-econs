# Future Work — Covariance Estimation Enhancements

Items below are **not blockers** for the v1.1 spike release. They represent
capability gaps identified during the fe/iv/ols pyfixest integration. Each
item names a specific implementation path, not a vague exploration.

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

### D11: did_cs() Aggregation Modes — `.aggte()` Delivered

- **What:** Expose `aggte()`-style dynamic/group/calendar aggregation
  modes in `did_cs()`. R's `did::aggte()` supports
  `type = "dynamic"` (event-time ATTs), `type = "group"` (cohort-specific
  ATTs), and `type = "calendar"` (calendar-time ATTs).
- **Status:** Delivered (2026-07-14). `CsDiDResult.aggte(type=)`
  implements all three types with R parity at `rtol=1e-6`. Four bugs
  found and fixed during validation (wrong divisor, wrong RIF index,
  wrong two-stage aggregation, missing centering in per-level group SE).
  See CHANGELOG for details.
- **Remaining:** `did_sa()` and `did_gardner()` extensions
  still queued (citations verified: Sun & Abraham 2021, Gardner 2022).

### D12: did_gardner() — Gardner (2022) Two-Stage DID — Delivered

- **What:** `did_gardner(data, y, first_stage, second_stage, treatment, cluster=)`
  implements the Gardner (2022) DID2S estimator with cluster-robust SEs via
  two-stage influence functions.  Returns `GardnerResult`.
- **Status:** Delivered (2026-07-14). R parity at `rtol=1e-6` against
  `did2s::did2s()` v1.2.1 (non-bootstrap). 14 tests, 0 regressions.
- **Key finding:** Two-stage IF formula `IF = IF_fs - IF_ss` with
  `gamma = (X10'X10)^{-1} (X1'X2)` (original x1, not zeroed-out x10).
  Naive single-stage VCE underestimates SE by ~17%.
- **Remaining:** `did_sa()` still queued for next session.

### D13: did_sa() — Sun & Abraham (2021) Interaction-Weighted DID — Delivered (R parity)

- **What:** `did_sa(data, y, cohort, period, ref_period, entity, time,
  cluster, covariates)` implements the Sun & Abraham (2021) interaction-weighted
  estimator with cluster-robust SEs. Returns `SaDiDResult` with ATT/SE/t/p,
  period-level and cohort-level aggregated views, full 9×9 VCE.
- **Status:** Delivered (2026-07-14). R parity at `rtol=1e-6` against
  `fixest::sunab()` v0.14.2. 23 tests, 0 regressions.
- **Key findings:**
  - SSC formula confirmed from fixest source (`vcov_cluster_internal`,
    `ssc_compute_K`): `G/(G-1) × (n-1)/(n-K)` where `K = nparams - (G-1)`.
  - Collinearity detection via sequential projection (Gram-Schmidt in original
    column order) matches fixest's Cholesky-based detection.
  - ATT is the time::0 period-level aggregate (cohort-weighted), not the
    mean of all interaction coefficients.
- **Stata-parity gap:** No Stata anchor exists for the Sun-Abraham estimator.
  Stata equivalent would be `csdid` (Callaway & Sant'Anna 2021) with
  `aggte(type="simple")`, or `eventstudyinteract` (Sun & Abraham 2021
  Stata implementation by Sun). These are distinct packages with different
  defaults; parity work is deferred.
  - **Reference:** `csdid` (R & Stata), `eventstudyinteract` (Stata, by Liyang Sun).
  - **Decision required:** Whether to implement Stata parity for D13 using
    `csdid` or `eventstudyinteract` as the anchor, or to treat R fixest as the
    sole parity anchor for this estimator.

---

## Test Layout — Deferred Migrations

### Root-Level Test Files → `non_stata_nor_r/`

- **What:** ~20 unit/cross-check test files at the `tests/` root
  (`test_cem.py`, `test_psm.py`, `test_nls.py`, `test_synth.py`,
  `test_synth_placebo.py`, `test_did.py`, `test_event_study.py`,
  `test_fe.py`, `test_ols.py`, `test_iv.py`, etc.) should migrate to
  `tests/non_stata_nor_r/` per the finalized 4-role taxonomy.
- **Why deferred:** Large scope (~20 files), complex cross-references
  to both Stata and R fixture directories, purely cosmetic (no parity
  impact), real regression risk if paths break. The Stata/R parity
  tests (the core product per standing rule 2) are already correctly
  organized; this is polish, not substance.
- **Status:** Delivered (2026-07-14). 36 files migrated, 4 relative
  imports fixed (`from .r.r_runner` → `from ..r.r_runner`), full suite
  804 passed, 0 regressions.
- **Decision:** 2026-07-14, layout migration session.

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

---

## GMM Convention Differences (Documented, Not Bugs)

### GMM-J: J-Statistic Convention (Closed)

- **What:** OE's one-step J uses model-based S: `J = g'(Z'Z)^{-1}g / sig2`
  (line 158 of `_gmm_core.py`). Stata's `e(J)` uses robust sandwich
  `S_hat = (1/N)Σg_ig_i'`. R's `specTest()` does NOT include `/sig2`.
  Two legitimate, source-confirmed conventions exist.
- **Evidence:** Stata's `gmm.ado` line 1358: `e(J) = Q * N`. With the
  corrected fixture (single-equation + `winitial(unadjusted)`), Stata's
  one-step J=3.7702 (model-based weighting from `winitial(unadjusted)`),
  while OE's one-step J=4.085 (robust S when `cov_type="robust"`).
  Both are valid under their respective conventions.  Two-step J matches
  to machine epsilon (both use efficient S^{-1}).
- **Status:** Resolved 2026-07-17. OE's `/sig2` convention was kept as-is.
  Documented in `_gmm_core.py` module docstring, inline comment at line 158,
  and `gmm()` docstring. See commit `a941114`.
- **No action required.**

### GMM-HAC: HAC Kernel Scope Convention

- **What:** OE applies Bartlett kernel to VCE only (not to weighting matrix).
  R's `gmm(vcov="HAC")` applies kernel to BOTH weighting matrix AND VCE.
  Therefore R's HAC two-step SEs diverge from OE's (R=[0.158, 0.101, 0.900]
  vs OE=[0.145, 0.103, 0.826] on the 300-obs fixture).
- **Status:** Documented in `test_r_gmm.py` and `gmm.R` fixture comments.
  HAC tests check finiteness/J>0 only, not numeric parity.
- **Implementation path:** If parity is desired, add a `hac_weighting=True`
  parameter to `gmm()` that applies the kernel to both W and VCE.
  Reference: R `gmm` package `.myKernHAC` / `.weightFct`.

### GMM-WC: Windmeijer Correction Parity Gap

- **What:** OE's `gmm()` always applies the Windmeijer (2005) finite-sample
  correction to the two-step robust VCE (`_gmm_core.py` lines 224-239).
  Stata's `gmm` command does NOT apply this correction by default — confirmed
  via gmm.ado source (no Windmeijer code, no WC-robust label, no toggle
  option).  Contrast: Stata's `xtabond`/`xtdpd` DO apply it by default.
  Two-step robust SEs therefore differ at ~15% (OE larger, as expected for
  Windmeijer).  On the 300-obs fixture: OE SEs=[0.145, 0.103, 0.826] vs
  Stata SEs=[0.126, 0.099, 0.775].
- **Consequence:** Two-step robust SE assertions against Stata's `gmm` are
  NOT valid parity targets.  Coefficients and J match to machine epsilon.
- **Status:** Documented 2026-07-17 in `test_stata_gmm.py` (SE assertions
  removed for two-step robust) and `FUTURE_WORK.md`.
- **Decision required:** Whether to add a `windmeijer=True` parameter to
  `gmm()` to allow disabling the correction for Stata `gmm` parity, or
  to accept the gap as a deliberate OE improvement (Stata's `xtabond` also
  applies Windmeijer, suggesting Stata's omission in `gmm` is the weaker
  default).

### GMM-GN: Stata Expression-Form Weighting Matrix (Closed)

- **What (corrected 2026-07-17):** The original claim — "Stata's Gauss-Newton
  doesn't converge to 2SLS" — was **wrong**.  For linear models, the GMM
  objective is quadratic and any Newton solver converges in one step (confirmed:
  `e(converged)=1` in all configurations, tightening tolerances has zero effect).
  The 6.8% gap was a **specification difference**: Stata's multi-equation
  expression form with `winitial(identity)` minimizes `(Y-Xb)'ZZ'(Y-Xb)` —
  the ZZ'-weighted objective — NOT the standard 2SLS objective
  `g'(Z'Z)^{-1}g`.  These coincide only when L==p (exactly-identified).
- **Resolution:** Regenerated the Stata fixture (`gmm.do`, `gmm.dta`) using
  single-equation form with `instruments()` + `winitial(unadjusted)`, which
  gives standard 2SLS.  All overidentified coefficients now match OE to ≤1e-7.
  Confirmed: Stata single-equation b2=1.354058 = OE b2=1.354058 =
  Python exact 2SLS b2=1.354058.
- **Status:** Resolved 2026-07-17.  Fixture regenerated, tests rewritten with
  valid parity assertions.  See `tests/stata/generate-fixtures/gmm.do` header
  for the specification rationale.

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

*Last updated: 2026-07-17, GMM parity audit verification (Items 1-2 resolved).*
