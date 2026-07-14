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

### D11: staggered_did() Aggregation Modes (Future Enhancement)

- **What:** Expose `aggte()`-style dynamic/group/calendar aggregation
  modes in `staggered_did()`. R's `did::aggte()` supports
  `type = "dynamic"` (event-time ATTs), `type = "group"` (cohort-specific
  ATTs), and `type = "calendar"` (calendar-time ATTs). OE's
  `staggered_did()` currently only returns the simple pooled ATT and
  per-cell ATTs.
- **Status:** Approved as future enhancement, explicitly out of scope
  for the current phase. These are additive scope beyond what's needed
  to close the parity gap (D9).
- **Reference:** R `did::aggte(type = "dynamic"|"group"|"calendar")`.
- **Decision:** D11 (2026-07-14).

---

*Last updated: 2026-07-14, did() R parity anchors complete.*
