# Future Work — Covariance Estimation Enhancements

Items below are **not blockers** for the v1.1 spike release. They represent
capability gaps identified during the fe/iv/ols pyfixest integration. Each
item names a specific implementation path, not a vague exploration.

---

## QUEUED — next-session prioritized performance work (rule 11, bounded prompts)

These are the items from the 2026-07-17 Python-strength inspection plan, queued
for their own **separate, single-concern sessions**. Each must be scoped as its
own bounded supervisor prompt (do not bundle). Each must follow the parity gate
(rule 2: ≤1e-6; rule 5: exclude synth placebo) and reuse the `parallel: bool`
opt-in convention from `placebo.py` / `did_cs.py`. Full plan context: 2026-07-17
handoff (GPU declined; Candidate A synth-analytic-gradient deferred — see bottom
of this file).

### Candidate D — _gmm_core._hac_S vectorize — DONE (commit `897c31a`)
- Implemented 2026-07-17: the inner per-lag / per-t `np.outer` accumulation in
  `_hac_S` (in `open_econs/models/_gmm_core.py`, NOT under `causal/`) replaced
  by a single batched `np.einsum("ti,tj->ij", moments[lag:], moments[:-lag])`
  reduction per entity. The per-entity loop is preserved (ragged time
  dimensions). **Bit-identical** to the scalar loop (atol=0) — verified by new
  `TestHacSVectorization` in `tests/non_stata_nor_r/test_gmm_core.py`
  (parametrized over seeds/lags, adjust flag, full-sample hac_weighting path,
  and no-time pooled path). The reduction order along the time axis is
  unchanged, so `sum(axis=0)` of the (T-lag, L, L) tensor matches the
  sequential `Gamma += np.outer(...)` loop exactly. The contemporaneous term
  `S_ent = moments.T @ moments` (already a single BLAS call in the original)
  is left as-is. Regression anchors `tests/stata/tests/test_stata_gmm.py`
  (HAC two-step, 31 tests) and `test_stata_abond.py` (40 tests) all still pass
  ≤1e-6. No parity drift. Do not re-touch unless a new HAC edge case breaks
  bit-identicality.

### Candidate C — psm.py vectorize — DONE (commit `cdb15be`)
- Implemented 2026-07-17: batched `cKDTree.query` in `_within_treatment_matching`
  / `_opposite_treatment_matching`; fancy-indexed `psi`; padded `(n,h)` /
  `(n,h,p)` tensor reduction for `xi2` and `c_tau` (via `_padded_local_cov`);
  vectorized `matched_arr` mask. **Bit-identical** to the scalar loops
  (atol=0, rtol=0) — verified by new `test_psm_*_bit_identical_to_scalar` and
  `test_psm_c_tau_vectorization_bit_identical`, plus the existing Stata-pinned
  `test_psm_se_nn*` (nn=2/5/10) which still pass ≤1e-6. ~4x faster on the Stata
  fixture (psm nn=10: 0.50s → 0.13s). No parity drift. Do not re-touch unless a
  new neighborhood-size edge case breaks bit-identicality.

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

### iv() FE SE DOF Reconciliation (RESOLVED 2026-07-17)

- **Empirical result (source-verified, N=500, n_g=50, K=3, data
  `df_iv_panel.csv`):** `iv(..., entity="id")` vs Stata `xtivreg y w (x=z1 z2), fe`.

  | cov_type | Stata SE(w) | OE SE(w) (fe_robust="xtivreg") | match? |
  |----------|-------------|--------------------------------|--------|
  | nonrobust | 0.0555038 | 0.0555038 | ✅ exact (≤1e-6) |
  | robust (default `fe_robust="xtivreg"`) | 0.0523534 | 0.0523534 | ✅ exact (≤1e-6) |
  | robust (`fe_robust="hetero"`) | — | 0.056824 | legacy pyfixest HC1, documented non-Stata |

  Coefficients match Stata exactly in all cases (within-transform drops _cons;
  both return only `w`, `x`).

- **Root cause (RESOLVED):** Stata `vce(robust)` on `xtivreg, fe` is **cluster-robust
  by the entity id**, not a heteroskedastic HC estimator.  Verified against
  `xtivreg.ado` `within` program: `vce(robust)` sets the `cluster` local to `2`, which runs
  `_regress ..., cluster(`id')` on the demeaned data.  The line-1816 rescale
  `(e(df_r)/(e(df_r)-n_g+1))` applies ONLY to `cluster==0` (nonrobust `conventional`),
  never to robust — so the prior `fe_dof` df-rescale hypothesis was wrong (a rescale
  made SEs worse: 0.0602 vs Stata 0.0524).  The inner `_regress, cluster(id)` returns
  SE(w)=0.0523534 (df_r=49), identical to the outer `xtivreg, fe vce(robust)` (df_rz=448)
  — confirming the path.  Stata's reported `e(df_rz)` = 448 = `N - n_g + 1 - K`, identical to
  OE's pyfixest `df_resid`, so df was never the issue.

- **Fix (rule 15 toggle):** new `fe_robust: str = "xtivreg"` param on `iv()`.  When FE is
  present and `cov_type in (robust, HC1, heteroskedastic)` with no explicit `cluster`,
  `fe_robust="xtivreg"` → pyfixest `vcov={"CRV1": <entity col>, "debiased": debiased}`
  (matches Stata); `fe_robust="hetero"` → pyfixest `vcov="HC1"` (legacy behavior, does NOT
  match Stata, preserved as an explicit alternative).  An explicit `cluster=` argument
  takes precedence.  `debiased` still applies its `G/(G-1)` SSC; below 1e-6 at G=50.

- **Status:** RESOLVED.  `tests/stata/tests/test_stata_iv.py::TestIVFEStata` asserts both
  `nonrobust` and `robust` coef+SE against `iv_fe.dta` (regenerated 2026-07-17) to ≤1e-6.
  Methodology note updated in `methodology/linear/iv_2sls.md`.  The diagnostic `.do`
  (`iv_fe_diag.do`) + `iv_fe_diag.dta` live in `tests/stata/generate-fixtures/archive/` for
  re-tracing.  Next agent: no further work unless the user wants an R-parity FE-IV test
  (R `AER::ivreg` has no FE; would need `plm`/`lfe`).

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

### GMM-HAC: HAC Kernel Scope Convention (Stata + R parity, PARTIAL 2026-07-17)

- **What (two distinct conventions):**
  1. **Stata / OE-default HAC** (per-entity): the HAC long-run S is Newey-West
     *within each panel entity*, accumulated.  Matches Stata `gmm, wmatrix(hac
     ...) vce(hac ...)`.  Coefficient `[0.892, 2.017, 1.570]`, SE `[0.129,
     0.094, 0.797]` — matched to ≤1e-6 (coef AND SE) under
     `windmeijer=False, robust_meat="two-step"`.
  2. **R-pooled HAC** (`hac_weighting=True`): the HAC S is computed over the
     **full sample as one time series** (each obs its own entity) — R's
     `gmm(vcov="HAC")` applies the Bartlett kernel to BOTH weighting matrix AND
     VCE over the pooled sample.  Coefficient `[0.885, 2.018, 1.534]`.
- **Status:**
  - Stata/per-entity HAC: ACHIEVED (≤1e-6 coef+SE), tested in
    `TestGmmOverIdentifiedTwoStepHAC`.
  - R-pooled HAC: `hac_weighting=True` toggle ADDED.  **Coefficient matches R
    to ≤1e-6** (tested in `TestGmmROverIdentifiedHACWeighting`).  **SE matches
    R to within ~6e-4** (R SE `[0.128, 0.097, 0.802]` vs OE `[0.128, 0.097,
    0.803]`).  The residual ~6e-4 gap is **diagnosed and explained** (NOT a
    bug, NOT a loosened tolerance — see below).  The SE test asserts at
    atol=1e-3 with an explicit documented-divergence comment, NOT at 1e-6.
  - **Root cause of the ~6e-4 SE gap (source-confirmed 2026-07-17):** R's
    `gmm(..., vcov="HAC")` is **internally inconsistent** between the coefficient
    and the reported VCE.  The two-step *coefficient* is optimized with a HAC
    weight `W = S⁻¹` built from the **first-stage 2SLS residuals** (`.weightFct`
    is called with `res1$par` = the 2SLS theta in `momentEstim.baseGmm.
    twoStep.formula`).  But the *reported* `vcov` (`FinRes.baseGmm.res`) builds
    `v = .weightFct(z$coefficient, x, "HAC")` from the **final two-step
    residuals**, and uses it for BOTH `z$w` and `z$vcov`.  Empirically: the
    e1-HAC (2SLS-residual) bread reproduces R's coefficient to 2e-15, while the
    e2-HAC (two-step-residual) bread reproduces R's *SE* to 6 decimals
    (`[0.127674, 0.096713, 0.802281]`) — but gives a DIFFERENT coefficient
    (`[0.888, 2.016, 1.510]`).  So R's own `coef(g)` and `vcov(g)` come from
    two different S matrices.  OE uses ONE consistent S (e1-HAC bread → also
    used in the meat sandwich), so it matches R's coefficient exactly and lands
    within 6e-4 of R's (internally-inconsistent) SE.  **Do NOT "fix" this to
    1e-6** by switching OE's bread to e2-HAC — that would replicate R's
    inconsistency, break OE's own coef↔SE consistency, AND break the exact
    coefficient match.  Keep atol=1e-3 with this note.
  - FIX (2026-07-17): the R fixture `oid_hac_2s` previously stored the PLAIN
    optimal two-step coef `b2=[0.870,...]` as the "R HAC" coef — wrong.  R's
    actual HAC coef is `[0.885,...]`.  `gmm.R` now stores `coef(g_hac_oid)` /
    `vcov(g_hac_oid)`; the old `test_coefficients_match_r` asserting OE==b2 was
    validating against an incorrect R value and has been corrected.
- **Implementation path (residual R HAC SE gap):** NONE remaining — the gap is
  a confirmed R-internals inconsistency (coefficient weight from 2SLS residuals
  vs reported VCE from two-step residuals).  No source-dive needed; no code
  change warranted.  Documented as above; SE test stays at atol=1e-3.

### GMM-RCLUSTER: R "cluster=" is a NO-OP — RESOLVED as R `vcov="iid"` (2026-07-17)

- **Correction (rule 6):** The prior audit treated R's `gmm(..., vcov="iid",
  cluster=df$cluster)` as a distinct "cluster" convention with coef
  `[0.850, 2.012, 1.354]`.  Source-reading `gmm` v1.9-1 (`.weightFct`,
  `FinRes.baseGmm.res`, `gmm()`) proves the `cluster=` argument is **NOT a
  real parameter** — it falls through `...` and is never consumed.  R's `gmm`
  has **NO cluster VCE at all**.  The "R cluster" fixture value is simply R's
  plain `gmm(..., vcov="iid")` two-step GMM.  (The earlier `gc$w` reverse-
  engineering was chasing an ignored argument; the recovered coef was just the
  iid two-step coef.)
- **R `vcov="iid"` recipe (source-confirmed, reproduced to machine precision):**
  homoskedastic efficient weight `S_iid = Z_iid' Z_iid / n` where `Z_iid` =
  [intercept column] + the *explicit* instruments (exogenous regressors
  excluded — they are their own instruments in X).  Meat = `sig2 * S_iid`
  (scaled by two-step residual variance `sig2 = e2'e2/n`).  Coefficient
  `[0.850433, 2.011863, 1.354058]`, SE `[0.131726, 0.101611, 0.804557]`.
  Note R `vcov="MDS"` (EHW robust) instead matches OE's default
  `cov_type="robust"` coef `[0.870, 2.027, 1.464]`; R `vcov="iid"` matches
  OE `weight="iid"` coef `[0.850, 2.012, 1.354]`.
- **OE implementation:** `weight="iid"` toggle now builds this homoskedastic
  S for BOTH bread and meat (with the `n`-scaling required to match R's
  `V = (G' v^-1 G)^-1 / n`).  `TestGmmROverIdentifiedIidTwoStep` asserts
  coef+SE parity to <=1e-6; `TestGmmWeightToggleIidBread` asserts the toggle
  matches R `vcov="iid"` and differs from the Stata-style bread.  DONE.
- **Residual (closed):** none for the coefficient; R's HAC SE residual gap
  (GMM-HAC) is tracked separately.

### GMM-WC: Windmeijer Correction + Robust-Meat Conventions (RESOLVED 2026-07-17)

Two convention differences between OE's `gmm()` and Stata's `gmm` command
were identified, both now exposed as toggles:

**Difference 1 — Windmeijer (2005) correction.** OE applies the Windmeijer
finite-sample correction to the two-step robust VCE by default.  Stata's
`gmm` command does NOT (confirmed in `gmm.ado`: no Windmeijer code, no
WC-robust label, no toggle).  Stata's `xtabond`/`xtdpd` DO apply it.
→ Toggle `windmeijer` added (`gmm()`, `_gmm_core.estimate_gmm`); default True.
- OE default (`windmeijer=True`): matches R `gmm` `vcov="MDS"` to machine
  epsilon.  OE SEs=[0.145, 0.103, 0.826] vs Stata `gmm`=[0.126, 0.099, 0.775].

**Difference 2 — robust MEAT moment-covariance (THE residual 2.7% gap).**
Stata's `gmm` builds the robust VCE as the FULL sandwich
`V = (G' S1^{-1} G)^{-1} (G' S1^{-1} S2 S1^{-1} G) (G' S1^{-1} G)^{-1}`
where **S1** = one-step-residual moment cov (efficient weight / bread) and
**S2** = **two-step-residual** moment cov (robust MEAT).  The econometric
literature and R's `gmm` package (`vcov="MDS"`, `.weightFct` = `crossprod(gt)/n`)
use S1 for BOTH bread and meat, collapsing to `V2 = (G' S1^{-1} G)^{-1}`.
This is the residual ~2.7% gap (max |gap|=0.0208 with `windmeijer=False`
alone).  Stata's two-step S2 lives inside the compiled Mata `_gmm_wrk()`,
so it was confirmed numerically instead: Stata's extracted `e(S)` equals
`(1/N)·Σᵢ(Zᵢ·e2ᵢ)(Zᵢ·e2ᵢ)'` to machine epsilon, and feeding Stata's OWN
extracted `e(S)` into the full-sandwich formula reproduces Stata's `e(V)`
to ~2e-8.
→ Toggle `robust_meat` added; default `"one-step"` (literature/R collapse);
  `"two-step"` builds S2 from e2 and assembles the full sandwich.

**Resolution / parity:** With **`windmeijer=False, robust_meat="two-step"`**
OE reproduces Stata `gmm` two-step robust SEs to **max |gap| = 2.06e-08**
(≤1e-6).  Coefficients and two-step J match to machine epsilon in all cases.
- Stata `gmm` SE: [0.1260902, 0.0986776, 0.7745471]
- OE (`windmeijer=False, robust_meat="two-step"`): [0.1260902, 0.0986776, 0.7745471]
- OE default (`windmeijer=True, robust_meat="one-step"`) = R `gmm`: [0.14527, 0.10322, 0.82625]

**IMPORTANT semantics caution:** `robust_meat="two-step"` does NOT replace
the whole S with e2 — it switches only the robust MEAT S2 to e2 while keeping
the efficient-weight bread S1 at e1 (the full sandwich requires exactly this).
Do not "simplify" it to globally replacing S1 with S2 (that regresses to a
0.00013 gap).

**Tests:** `tests/stata/tests/test_stata_gmm.py`
`TestGmmOverIdentifiedTwoStepRobust` now asserts the ≤1e-6 SE parity with
`windmeijer=False, robust_meat="two-step"`; the default (Windmeijer) path is
documented as matching R rather than Stata `gmm` and is NOT asserted as Stata
parity.  R `gmm` has no `robust_meat="two-step"` equivalent (always e1), so
no R-side assertion is added (rule 3: no clean R anchor — documented, not
forced).  `abond()` is unaffected (inherits `windmeijer=True,
robust_meat="one-step"` defaults, correct for xtabond/xtdpd).

**Status:** RESOLVED — both gaps closed and exposed as toggles; parity tests
added.

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

## ABOND R-Parity (DEFERRED — plm `pgmm` broken; Stata parity complete, accepted as sufficient)

- **Decision (2026-07-17):** R `plm::pgmm` parity for `oe.abond()` is
  **intentionally DEFERRED / accepted-incomplete**.  Stata `xtabond2` parity is
  fully covered (40 tests, green) and is the primary anchor for abond's users.
  R parity adds little given low abond usage and is currently impossible to
  verify (plm `pgmm` is broken upstream).  This is a documented acceptance, NOT
  an open "todo" — do not treat it as pending work unless plm is fixed.

- **What (rule 15 gap):** `oe.abond()` has full Stata `xtabond2` parity
  (`tests/stata/tests/test_stata_abond.py`, 8 flavors × 5 cross-checks = 40
  tests, all green) but **no R parity**.  The canonical R anchor for
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
  authoritative; R parity is intentionally not covered.  The simulation-only
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

*Last updated: 2026-07-17, GMM parity audit verification (Items 1-2 resolved); abond R-parity flagged BLOCKED.*

---

## GPU acceleration — DECLINED project-wide (rule 19 finding)

**Decision (2026-07-17):** GPU (CuPy / numba-cuda) is **not adopted** for any
estimator in OE. Recorded so future sessions do not re-open it.

- **Why:** The genuine hot spots are (1) `scipy.optimize` SLSQP/Nelder-Mead in
  `synth.py` (no GPU backend; reimplementing the solver on GPU is out of scope
  and would break R `Synth` parity), and (2) numpy matmuls that **already run
  multithreaded on CPU** via the bundled BLAS. This environment ships
  **OpenBLAS 0.3.31, DYNAMIC_ARCH, MAX_THREADS=24** — so any vectorized numpy
  loop auto-parallelizes on the CPU without a GPU. GPU transfer overhead
  dominates at the current fixture sizes (tens-to-hundreds of entities /
  periods).
- **Revisit only if:** a user has 100k+ entity panels or a genuinely
  GPU-amenable kernel (e.g. large dense matmul-dominated GMM with no scipy
  optimizer in the hot path). Until then, prefer **vectorize-then-let-BLAS-
  thread** and **ProcessPoolExecutor for Python/GIL-bound loops** (see
  `methodology/performance-conventions.md`).
- **Note:** `ThreadPoolExecutor` is **useless** for numpy/BLAS-bound work
  (GIL held during BLAS calls); only `ProcessPoolExecutor` helps genuinely
  GIL-bound Python loops. Mirrored in `methodology/performance-conventions.md`.

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

## ARDL / UECM + PSS(2001) bounds test (v1.1.2): DONE

- **What:** `ardl_fit()` / `uecm_fit()` + `.bounds_test(case)` (F- and t-bounds,
  all 5 cases, LR multipliers, EC term), wrapping `statsmodels.tsa.ardl`.
- **Status:** COMPLETE. Parity to 1e-6 vs Stata SSC `ardl` (14 tests) and R
  `ARDL` (10 tests) + 27 backend tests. Conventions source-verified against
  `ardl.ado` / `ardlbounds.ado` and R `ARDL` source (rule 1). Root causes and
  the math/command manual are in `methodology/timeseries/ardl.md`.
- **Toggles exposed (rule 15):** `cv_vintage` (`pss2001` default vs
  `statsmodels`), `lr_sign` (`stata` default = −θ/ρ). Both branches tested.
- **Deliberate default (not a bug):** `bounds_test` default `signif=(0.10, 0.05,
  0.01)` — the `"2.5%"` critical-value key exists only when `signif` includes
  `0.025`. The Stata fixture stores 2.5% CVs and `TestStataARDLCritVals25`
  exercises that path. Revisit only if users want 2.5% in the default set.

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
- **Status:** PENDING (out of scope for the v1.1.2 commit set). Recorded per
  rules 12/18 so it is not lost.
- **Context:** root cause fully written up in `methodology/timeseries/ardl.md`
  and the inline comment block of `tests/stata/generate-fixtures/ardl.do`.

---

*Last updated: 2026-07-18, ARDL/UECM v1.1.2 completed to 1e-6 parity (Stata+R); flagged `set type double` audit for other `.do` generators.*

*Last updated: 2026-07-17, GPU declined + Candidate A deferred (Candidate B did_cs bootstrap parallelization implemented and pushed).*
