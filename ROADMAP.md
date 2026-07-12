## Roadmap

open-econs is built in two horizons. The **committed roadmap** (v0.1 → v1.0) is
what the maintainers are actually building and will be held to. The **North
Star** (v1.1 → v5.0) is the long-run vision — where the project could go if it
earns a community around it. Nothing past v1.0 is a promise; it's a map of the
terrain worth exploring, so contributors and users can see where their work
might fit.

---

### Committed Roadmap

#### v0.1 — Foundation *(shipped)*
- [x] `ols()` / `reg()` — OLS with HC1/robust and clustered standard errors
- [x] `oaxaca()` — two-fold and three-fold Oaxaca-Blinder decomposition
- [x] Bootstrapped standard errors for Oaxaca (seedable, reproducible)
- [x] `predict(newdata)` for regression models
- [x] `Context` object for dataset-scoped workflows
- [x] Immutable results, named `pd.Series`/`pd.DataFrame` outputs everywhere
- [x] `.tidy()`, `.summary()`, `.export()` (JSON)
- [x] Numerical parity verified against raw `statsmodels` output

#### v0.2 — Diagnostics & Quality *(shipped)*
- [x] Default `cov_type` changed to `HC2` (matches modern Stata)
- [x] Per-variable Oaxaca breakdown via `.variable_detail` and `tidy(detail=True)`
- [x] Diagnostic tests: Jarque-Bera, Breusch-Pagan, Durbin-Watson, Ramsey RESET
- [x] Condition number threshold lowered to 30 (Belsley standard); stored on result
- [x] `.export()` now supports CSV output
- [x] Wald / F-test API surface (stored statsmodels result ready for v0.3)
- [x] README example numbers fixed to match real output

#### v0.3 — Real Regression *(current)*
- [x] `summary()` shows diagnostics + condition number
- [x] `wald_test()` / `f_test()` on `OLSResult`
- [x] `ols(weights=...)` — weighted least squares
- [x] `.plot()` implemented (matplotlib as optional extra)
- [x] First PyPI release (`pip install open-econs`)

#### v0.4 — Causal Inference Core *(shipped)*
- [x] `fe()` — one-way and two-way fixed effects (within transformation + absorbed dummies)
- [x] `iv()` — instrumental variables / 2SLS, with first-stage F-stat
- [x] `logit()` / `probit()` — binary choice, with marginal effects (`.margins()`)
- [x] `ctx.vif()` — variance inflation factor / collinearity diagnostics

#### v0.4.1 — Audit Response *(superseded by v0.4.2)*
- [x] `__delattr__` blocks deletion of frozen result attributes
- [x] Oaxaca `swap` parameter documented honestly: only affects sign guarantee, does not reverse decomposition direction
- [x] `.vcov()` on `OLSResult`, `BinaryResult` — returns named `pd.DataFrame` variance-covariance matrix
- [x] `.to_latex()` and `.to_html()` on all result types
- [x] Summary shows `F-statistic ({cov_type}):` so users know which VCV the F-test used

> ⚠️ v0.4.1 incorrectly claimed to block `object.__setattr__` bypass — that is
> impossible in pure Python. v0.4.2 corrects this claim.

#### v0.4.2 — External Audit Fixes *(superseded by v0.4.3)*
- [x] **IV rewritten** — formula syntax ``y ~ exog | endog ~ instruments`` separates
  exogenous controls from endogenous regressors. Honors ``cov_type``.
  Reports Cragg-Donald weak-instrument F-statistic and Hansen J overidentification
  test.
- [x] **FE degrees-of-freedom corrected** — ``df_resid`` now accounts for absorbed
  entity/time dummies; two-way FE uses iterative alternating-projections demeaning
  (Correia 2017) on unbalanced panels.
- [x] **Immutability claim corrected** — ``__setattr__``/``__delattr__`` guard accidental
  mutation; ``object.__setattr__`` bypass is documented as **not prevented**.
- [x] **Context.ols defaults unified** — ``HC2`` to match the top-level ``ols()``.
- [x] **Condition-number warning** — excludes the intercept and column-scales the
  design matrix before computing, reducing false positives.

> ⚠️ v0.4.2 legacy IV syntax omitted exogenous controls from the instrument
> matrix — a classic 2SLS error. v0.4.3 fixes this.

#### v0.4.3 — Instrument-Matrix Correction *(superseded by v0.5)*
- [x] **Legacy IV syntax fixed** — ``y ~ x1 + x2 | z1`` now constructs the
  instrument matrix as ``[x1, x2, z1]`` (not just ``[z1]``).
- [x] **FutureWarning added** — legacy syntax warns users to adopt
  ``y ~ exog | endog ~ instruments``.
- [x] **New syntax unchanged** — ``y ~ w | x ~ z`` correctly passes
  ``w`` as exogenous, ``x`` as endogenous, ``z`` as instrument.

#### v0.5 — Design-Based Causal Inference *(current)*
- [x] `did()` — two-period DiD via ``y ~ treated * post`` with automatic DiD
  coefficient extraction; clustering, covariates, heterogeneous `cov_type`.
- [x] `event_study()` — event-study with `event_time` column; omits reference
  period; `.plot()` for pre-trend visuals.
- [x] `balance()` — covariate balance table with Welch t-tests.
- [x] **All v0.4 features preserved** — no breaking changes.
- [x] **v0.5.1 bug fixes** — FE `df_resid`/`rsd`/`adj_r2` now use the within fit's
  actual df; `event_study()` no longer crashes on interaction-only formulas or
  numeric reference periods.

#### v0.6 — Panel Data Engine *(shipped)*
- [x] First-class `PanelContext(df, entity=, time=)` — panel structure remembered,
  not re-specified per call. Exposes `pooled()`, `fe()`, `re()`, `diff()`,
  `driscoll_kraay()`, `hausman()`, plus cross-sectional delegates.
- [x] `re()` — random-effects (GLS) via `linearmodels.RandomEffects`, with `theta`,
  `rho`, within/between/overall R².
- [x] `hausman()` — Hausman test of FE vs RE (`H = (b_fe − b_re)'(V_fe − V_re)⁺(b_fe − b_re)`)
  with Möbius pseudoinverse; returns `HausmanResult`.
- [x] `driscoll_kraay()` — pooled OLS with DK (spatial/time-series-robust) SEs.
- [x] `diff()` — first-difference estimator (`FirstDifferenceOLS`).
- [x] Legacy `Context` gains panel methods delegating to a transient `PanelContext`.
- [x] **Test coverage** — numerical parity vs `linearmodels`/`statsmodels` on Grunfeld
  and synthetic panels; `hypothesis` property tests; edge-case tests. See `tests/test_panel_*.py`.
- [x] **v0.6.1–v0.6.3** — see below.

#### v0.6.1 — Dynamic panels, staggered DiD, RDD, robust SEs
- [x] `abond()` — Arellano-Bond difference GMM (one/two-step, Windmeijer SEs,
  Hansen J, AR(1)/AR(2)).
- [x] `staggered_did()` — Callaway-Sant'Anna (2021) staggered DiD with group-time
  ATTs, event-study aggregation, entity-clustered SEs.
- [x] `rdd()` — sharp and fuzzy RDD via triangular-kernel local linear regression.
- [x] Multi-way clustering (`cluster=["firm","year"]`, Cameron-Gelbach-Miller) and
  Newey-West HAC (`cov_type="HAC"`).
- [x] RE `llf`/`aic`/`bic` populated; `Context.pooled()` defaults to cluster-by-entity.
- [x] DRY: `_capture_call` centralized; `to_latex()`/`to_html()` no longer require
  `jinja2`; FE demeaning / docstring bug fixes.

#### v0.6.2 — Numerical stability & patch release
- [x] Robust pseudo-inverse for `abond()` weighting matrices (tiny-panel / near-singular stability).
- [x] Edge-case panel test coverage expanded.

#### v0.6.3 — Correctness fixes & applied-econometrics improvements
- [x] **Cragg-Donald Wald F-stat** for multi-endogenous IV (exact `linearmodels` stat).
- [x] **FE intercept stripping** by column name (robust to formulaic reordering).
- [x] **FE `cov_type` default** unified to `"HC2"`.
- [x] **Windmeijer (2005) two-step correction** for AB GMM SEs.
- [x] **Collapsed `abond()` instruments** (Roodman 2009) to curb "too many instruments".
- [x] **IK bandwidth for `rdd()`** (Imbens-Kalyanaraman 2012); Silverman kept as fallback.
- [x] **Honest staggered-DiD labeling** — renamed to "simplified OLS approximation"; ▶ superseded by v0.7.
- [x] Vectorized `_within_transform()`; Oaxaca `get_dummies` fix; FE `vcov()`/`se²` consistency.
- [x] **Hausman test fixed** — one-way FE override, df-corrected `vcov()`, Stata `e(chi2)` ghost-variable handling.

#### v0.6.5 — Arellano-Bond / `xtabond2` numerical parity
- [x] **`abond()` collapsed one-step non-robust matches Stata `xtabond2`** to ~1e-7
  (coefficients *and* VCV), on the 30×5 `df_panel` fixture.
- [x] **GMM instrument lag fixed** — Stata `gmm(L.y, lag(a b))` lags `L.y`, not `y`;
  prior construction silently used the initial observation as an instrument (the
  dominant VCV-gap source).
- [x] **Weighting-matrix `H` corrected** to the true unnormalized `M'M`.
- [x] Ground-truth extraction tooling (`abond_gt_*.csv`) for element-by-element validation; parity test added.

#### v0.6.6 — Non-collapsed / full GMM instruments (Arellano-Bond)
- [x] **Non-collapsed `abond()`** — full GMM expansion matching Stata `_MakeGMMinsts`/`_Explode`;
  block builder isolated, `n_gmm_i` formula corrected for lag offset.
- [x] **All four flavors** (collapsed/non-collapsed × one/two-step) validated vs Stata at machine precision (~1e-8).
- [x] **40 Stata-parity tests**; collapsed path untouched.

#### v0.6.7 — Oaxaca Stata parity & advanced options
- [x] **Stata parity fix** — `.do` files mis-extracted `e(b)` columns; regenerated
  `.dta`. OE matches Stata `pooled`/`threefold` to machine precision.
- [x] **`reference`** (two-fold: pooled/omega/group weights) and **`reverse`** (three-fold) parameters.
- [x] **21 new tests** at 1e-12; 96 Oaxaca tests total.

#### v0.6.8 — RDD with rdrobust backend + event_study fix
- [x] **RDD rdrobust backend** — CCT bandwidth, separate-side LLR, NN cluster-robust
  variance (`pip install open-econs[rd]`, rdrobust >= 2.0).
- [x] **RDD built-in fallback** — IK bandwidth, NN/EHW variance (no rdrobust dependency).
- [x] **event_study() fix** — falls back to first available period when `omitted_period` absent.
- [x] Removed faulty Stata event-study test (DiD vs event-study parameterization mismatch).

#### v0.6.8.1 — Logit/Probit AME correction & Stata parity
- [x] **`logit()/probit().margins()` now AME** (`at="overall"`), matching Stata `margins, dydx(*)`.
- [x] **Fixed logit margins fixture** (`_b[x1]` → `r(b)[1,1]`); tolerances tightened to 1e-6.

#### v0.6.8.2 — CS2021 Doubly-Robust Staggered DiD & cell-by-cell Stata parity
- [x] **`staggered_did()` now the full Callaway & Sant'Anna (2021) DR group-time estimator**
  (was simplified OLS approx): `dripw` (logit PS + OLS outcome reg + IPW RIF) and `reg`.
- [x] **18 cell-by-cell Stata-parity tests**; ATT(g,t) coefficients match at 1e-6.
- [x] **Unbalanced-cohort fixture** added; weight-formula audit (cohort-proportional, not uniform).
- [x] Docstrings updated to state CS2021 DR inference.

#### v0.6.8.3 — Full Oaxaca Stata parity (all decomposition variants)
- [x] **Expanded Oaxaca fixtures** — two-fold (4 reference variants) + three-fold
  (default + reverse) components extracted.
- [x] **20 new parity tests** vs Stata `oaxaca` v4.1.1; Stata↔OE terminology documented.

#### v0.6.8.4 — Newey-West `hac_adjust` + Panel FD fixture fix
- [x] **`ols(hac_adjust=True)` / `newey_west_cov(adjust=True)`** — opt-in Stata-style
  `N/(N−K)` HAC df correction; parity tests at 1e-7.
- [x] **Panel FD fixture corrected** (`regress dy dx dz, noconstant`); tolerances tightened to 1e-6.

#### v0.6.9 — Full Stata-parity coverage & test-suite caching
- [x] **Event-study Stata parity** (synthetic fixture, t-dist inference) at 1e-6.
- [x] **All 8 ABOND flavors live-verified** via `read_stata()` (40 tests at 1e-6);
  non-collapsed ~1e-16, collapsed ~3e-9.
- [x] **Module-level `read_stata()` caching** — 22 Stata calls; suite 235s→94s (2.5×).
- [x] All 149 Stata-parity tests pass.
- [x] **staggered-DiD live `read_stata()` conversion deferred** to the v0.7 DR rewrite.

#### v0.7 — Causal Inference: DR Estimator, RDD Refinements *(current)*
- [x] **v0.7.0 — staggered_did sample-alignment & SE-gap root cause** *(done)*
  - **`.do` sample-alignment** — drops never-treated gvar=5 entities to match Python
    filters; regenerated `.dta` (`e(N)` 150→100 balanced, 150→115 unbalanced).
  - **SE-gap root-caused** to `makerif2` full-sample IF rescaling vs OE per-cell IF
    (sample mismatch ruled out); test constants synced.
- [x] **v0.7 — CS2021 doubly-robust (DR) estimator IF rewrite** *(done)*
  - **`_cell_dripw` reimplemented from R `DRDID` `drdid_panel`** (csdid default
    `dripw`); validated to machine precision vs csdid's saved per-entity RIF
    (per-cell corr 1.0; SE exact: cell (3,3)=0.4652265, (3,4)=0.4941999).
  - **Full-sample weighted-RIF cluster SE** → aggregated **0.41781627** (balanced) /
    **0.62720813** (unbalanced), replacing the wrong `sqrt(mean se²)`.
    - This is validated against **csdid's own influence-function aggregation**, i.e.
      `csdid y x z, saverif(rif)` + `csdid_stats simple`, and the `did` R package's
      `aggte(type="simple")` (`getSE` = `sqrt(mean(if²)/n)` = `sqrt(Σ if_i²/N²)`). All
      three agree to machine precision.
    - **IMPORTANT — `csdid_estat simple` is NOT the reference.** In the installed
      csdid version (v1.6/v1.58) `csdid_estat simple` is buggy: it posts the raw
      per-(g,t) VCoV and prints element [1,1] (the *first, pre-treatment* cell's SE:
      0.7479047 balanced, 0.47824472 unbalanced) as the "simple" ATT SE — which is
      not an aggregation SE at all. Do not compare OE's aggregated SE against
      `csdid_estat simple`; use `csdid_stats` (or the `did` R package) instead.
  - Parity now holds at **rtol=1e-6** (was 0.2/0.6); all 18 staggered-DiD tests pass.
    References: see README.
- [x] `test_stata_staggered_did.py` → live `read_stata()` comparison *(done: fcc8fe8)*
- [x] RDD bandwidth selection (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik) — both `ik` (built-in `_ik_bandwidth`) and `cct` (rdrobust backend) present and tested; `rdd.py:186` / `rdd.py:390`. CCT-without-rdrobust silently falls back to IK, consistent with the established built-in-fallback design.
- [x] McCrary / CJM density test for manipulation at the cutoff — `oe.density_test()` + `RDResult.density_test()`; dual backend (rddensity wrapper + native built-in); matches Stata `rddensity` to machine precision; `rddensity>=2.0`.  See `open_econs/models/causal/rdd.py`.
- ~~Built-in RD plot (binned scatter + fitted lines either side of cutoff)~~ — **descoped.** `rdrobust.rdplot()` already provides this for users with the `[rd]` extra; a wrapper adds no capability, requires re-passing `data` (RDResult does not store the DataFrame), and conflicts with OE's "clean named outputs, not visualization" design scope. Decision recorded 2026-07-11.
- [x] **Plot-audit findings committed (2026-07-11):**
  - `OLSResult.plot()` deprecated in v0.8, removal in v0.9 (panel 4 had been a `"planned for v0.4"` placeholder since v0.3; no OE diagnostics annotated; self-contained but redundant/generic).
  - `EventStudyResult.plot()` kept as-is (not redundant with any dependency; self-contained; minimal but defensible domain convention).
  - Replacement diagnostics-annotated plot logged as non-blocking roadmap item — the audit surfaced that `self.diagnostics()` + `self.condition_number` are already computed but `.plot()` never used them. The analysis/ingredients exist; only the annotation is missing.

> **Non-blocking** — Diagnostics-annotated `OLSResult.plot()` replacement: annotate the existing 4-panel layout with the test statistics/p-values `self.diagnostics()` already computes (JB p-value on the QQ panel, BP p-value on residuals-vs-fitted, DW stat as an annotation, condition number in the margin). Would flip this from "redundant wrapper" to "the one thing the audit found genuinely missing." Not blocking v0.8; can land in the same window or later.

#### v0.8 — Matching & Balance
- [x] `psm()` — 1:1 nearest-neighbor with replacement on logit PS (ATE).
  Validated vs Stata ``teffects psmatch``: ATE exact to 1e-7, SE exact to 1e-6
  across nn=2,5,10 (AI 2012 PS-estimation adjustment implemented).
  **Deviations:** (a) with-replacement only — ``teffects psmatch`` has no
  without-replacement option; without-replacement may be added later if needed.
  (b) Default caliper (1.0) validated; tighter calipers not independently tested.
  See ``tests/test_psm.py``.
- [ ] **Kernel / smooth-weight matching on top of the PS engine — INVESTIGATED 2026-07-11, deliberately NOT built in v0.8 (deferred to v0.9 as its own scoped, parity-validated pass). Classification: (c) investigated, deliberately deferred.**
  - **Reference implementation:** `psmatch2` (Leuven & Sianesi, SSC) is the classic/applied-standard kernel matcher and directly supports `kernel` on the PS (`psmatch2 treat covariates, kernel kerneltype(epan) bw(#) outcome(y) ate` — confirmed live, ATT = −31.07, S.E. = 22.53 on cattaneo2). `kmatch` (Ben Jann, SSC) is the modern alternative (richer options, better-documented variance). **Both are viable parity references for the point estimate**; `psmatch2` is the more "expected" one in applied micro. Confirmed live that `teffects psmatch` does **not** accept `kernel` (r(198)), and neither does `teffects nnmatch` in this Stata 17 — so `teffects` (the command `psm()` is validated against for NN) cannot be the kernel reference. Installed `kmatch` is 1.1.5 (only `kmatch md ..., kernel()`; `psmatch2` runs out of the box).
  - **Variance (read from source) — and why the SE is the hard part:** all three Stata commands compute the *same* kernel-on-PS point estimate but **diverge on the SE**, and none matches OE's `psm()` standard:
    - `psmatch2, kernel` → Abadie–Imbens (2002) influence function, **explicitly without** the PS-estimation correction (its own output note: "S.E. does not take into account that the propensity score is estimated") — known to *understate* SEs.
    - `kmatch` → influence function with weights assumed *fixed* (Jann 2019); `kmatch.sthlp` states analytic SEs are generally *conservative* and recommends `teffects`/bootstrap for consistent SEs.
    - `teffects psmatch` (discrete NN, what `psm()` is validated against) → **full AI-2012**, including the `c'_τ V_γ c_τ` PS-estimation adjustment (`psm.py:382,384,470-473`).
    - **Consequence:** since `teffects` can't do kernel, ANY kernel reference (psmatch2 *or* kmatch) leaves a gap — to stay consistent with OE's teffects-equivalent standard the kernel variance needs *both* the continuous-weight `K/K'` generalization *and* the PS-estimation adjustment, which neither user command provides by default. The point estimate is reusable; the variance is a genuine build.
  - **The continuous-weight trap (the reason it is not a bolt-on):** OE's `psm()` variance (`psm.py`) is built on the discrete NN with-replacement count structure — `K_m(i)` (times matched) and `K'_m(i) = Σ_{j: i∈Ω(j)} 1/|Ω(j)|²` (the `K² + 2K − K'` term, `psm.py:382,405,426`). For kernel weights these counts are replaced by **weight-based aggregates** (total kernel weight supplied/received, and a normalized-squared-weight term). A naive "weighted average" plug-in is wrong — the `K'` normalization is what carries the AI-2012 variance over to continuous weights. This needs its own validated implementation, not an extension of the discrete `K/K'`.
  - **Why deferred (not (a) reuse, not (b) built now):** it is a distinct estimator from the discrete-NN `psm()` variance, not a reuse; and bolting an un-validated continuous-weight variance onto the v0.8 close-out would violate the "built-and-validated or explicitly deferred" + "CI-green" standard. It deserves a dedicated pass with parity fixtures vs `kmatch` (and a decision on analytic-fixed-weights vs bootstrap). Revisit as v0.9 line item.
- [x] Coarsened exact matching

  Core CEM: auto-Sturges coarsening (matching Stata's default), explicit
  cutpoints per variable, exact matching for categorical variables, ATT
  weights matching Stata's ``cem-mata.do`` formula.  Strata, weights, and
  matched flags validated per-observation against Stata's ``cem`` SSC
  package on a 1000-obs fixture at the exact-match level (no tolerance).
  See ``tests/test_cem.py``.

  **Pass 1 only** — SE (Pass 2) assessed and deliberately skipped: no
  CEM-specific SE formula exists (Stata's cem.ado, the Stata Journal paper,
  and Iacus/King/Porro 2019 inference theory all confirm CEM is preprocessing
  only — standard weighted-regression SEs are valid under stratified sampling).
  OE's existing ``ols(weights=..., cov_type="HC3")`` already covers the use
  case; a convenience ``estimate()`` wrapper would be scope-for-its-own-sake.
  (Revisit if users request it post-release.)
- [x] **Pass 3a — Alternate auto-coarsening**: added ``autocuts`` parameter
  (``"sturges"``, ``"fd"``, ``"scott"``, ``"ss"``) to ``cem()``.  Formulas
  verified against ``cem-mata.do`` (IQSS/cem-stata) and per-observation
  strata/weights/matched validated against Stata 17 for all four methods on a
  500-obs synthetic fixture.  FD includes Stata's MAD fallback when IQR=0.
  See ``tests/stata/test_stata_cem_autocuts.py``.
  k2k matching and L1 balance diagnostics remain as Pass 3 (investigated,
  deliberately not built — see prior investigation report).
- [x] Pass 1 — weighted balance diagnostics in ``ctx.balance()``:
      ``weights=`` parameter with SMD (unweighted pooled-SD denominator,
      Rosenbaum–Rubin convention), variance ratio (weighted group variances),
      and WLS t-test (Stata ``[iw]`` convention).  Uniform-weight semantics
      (no treated-weight override) — structurally identical to pstest for
      OE's matching estimators where treated weight = 1 by construction.
      Stata parity via ``pstest`` formula confirmed against ``balance_weighted.dta``.
- [x] Pass 2 — wire ``PSMResult.balance()`` and ``CEMResult.balance()`` to
      ``ctx.balance(weights=)`` using each estimator's internal weight vector.
      (Also exposed ``PSMResult.weights`` and ``PSMResult.matched`` as public
      ``pd.Series`` attributes; ``CEMResult`` already had those.)
- [x] Sensitivity analysis (Rosenbaum bounds) — ``rosenbaum_bounds()`` standalone function +
      ``PSMResult.sensitivity()`` wired to stored ``_pairs`` dict.  Validated against
      Stata ``rbounds`` (Gangl v1.1.6) at Γ = 1, 2, 3.  Zero-difference pairs follow
      Stata convention (included in rank computation, psp=psm=0) — confirmed from
      ``rbounds.ado`` ``rbrksm`` lines 131–132.  Pass 1: p-value bounds only (no
      Hodges-Lehmann, no CIs, no CEM).

#### v0.9 — Structural Foundations & Release Candidate
- [x] `gmm()` — general GMM estimation framework other estimators can build on
  - Shipped as a generic linear GMM framework, extracted from and validated against `abond()`'s solver (byte-identical pre/post extraction), with AB-specific conventions (`sig2_scale`, `small_sample_correction`) decoupled into explicit parameters; public API reuses `iv()`'s formula grammar, validated against `iv()`/2SLS on exactly-identified systems and Hansen J size/power via Monte Carlo; wired into `PanelContext` as a thin cluster-defaulting delegate.
- [x] `nls()` — nonlinear least squares
  - Shipped as a nonlinear least squares estimator using `scipy.optimize.least_squares` with sympy-based analytic Jacobian (numerical fallback when analytic differentiation fails, flagged not silent); new `white_cov()` added to `core/cov.py` for HC0-HC3; validated against `curve_fit`, R's `nls()`, and Stata's `nl` (all three converging to the same SE); `sympy` shipped as an optional `[nls]` extra, not a hard dependency.
- [x] `mlogit()` — multinomial logit (shipped, see v0.8)
- [ ] `nlogit()` — nested logit *(deferred: recon complete, documented in `docs/nlogit-recon.md`. Blockers: R `mlogit` can't run full Stata-equivalent spec (nest-level covariates cause singularity on `webuse restaurant`); no validated fixture with τ∈(0,1); analytic gradient ~200 lines of recursive tree traversal needs domain-expert implementation.)*
- [x] `synth()` — synthetic control (Abadie-Diamond-Hainmueller) — **core point estimator shipped** (v0.9)
  - Shipped: nested V+W optimization (outer predictor weights via R `Synth`'s two-start equal/regression procedure; inner donor-weight QP, `W>=0` & `sum W=1`, via SLSQP), `SynthResult` exposing `weights` / `predictor_weights` / `pre_mspe` / `post_mspe` / `gap_path`, `custom.v` fixed-`V` support, and gated R `Synth` (primary) + Stata `synth` (secondary) parity tests that skip cleanly where R/Stata are absent (e.g. CI). Default predictors = outcome's own pre-treatment path (one per period); explicit predictors = user-supplied covariates aggregated by pre-window mean.
  - Shipped (v0.9, this pass): `placebo_space()` / `placebo_time()` ADH permutation inference on top of the validated `synth()` solver — no estimator logic duplicated. `SynthResult` now also stores the original `predictors` argument (the one fit-config field it previously lacked) so the delegate can reconstruct each placebo call. `exclude_pre_mspe_multiple` is an **opt-in, space-only** pre-fit exclusion parameter (default `None`: never applied silently; ADH's chosen multipliers vary by application, so no single value is hard-coded). `placebo_time()` deliberately does **not** accept `exclude_pre_mspe_multiple` (in the in-time loop "pre-MSPE" is the treated unit's own fit against itself, so the space-style exclusion concept does not carry over; passing it raises `TypeError`). Verified against R `Synth` via a gated parity test (p-value matches exactly; per-donor ratio agreement `< 0.1` for the well-determined majority; residual up-to-`~3` divergence on a handful of rank-deficient placebo donors is the same documented nonconvex-V solver divergence as the core `synth()` parity test — reported, not forced to match).
   - Deferred (own future items, currently `NotImplementedError` by design): `plot()`; `predict()` out-of-sample counterfactual.
- [ ] `synth()` solver cross-OS nondeterminism *(deferred: root-caused & documented in `docs/synth-cross-os-solver-recon.md`, NOT fixed)*. `synth()`'s inner QP solver (SLSQP-based `_optimize_v` / `_solve_w`) is **cross-OS nondeterministic for rank-deficient donor pools** (donor count ≤ predictor count): the inner weight problem is rank-deficient, so `scipy` SLSQP follows a different trajectory on Linux vs Windows and lands on a different local optimum. Evidence (measured, do not re-derive): synthetic-control covariate mismatch `cov_mm_py ≈ 2.67e-07` on Windows vs `≈ 3.27e-01` on Linux for identical inputs; R's reference `Synth` solver is OS-stable at `~3.8e-07` on both. Consequence (honest, deliberate, documented interim — not a silently-reduced test count): two parity tests, `test_synth_rank_deficient_qp_same_objective_different_w` and `test_placebo_space_parity_r`, are gated behind R availability (`skipif(not R_AVAILABLE)`, i.e. skip on CI) rather than running as full fixture-based CI tests. Fix scope is a real tracked correctness gap (affects rank-deficient donor pools under nonconvex-V optimization, not the well-determined case), not cosmetic; candidate fixes (pin a deterministic inner-QP solver, or detect rank-deficiency and switch to a minimum-norm `W` formulation) are noted in the recon doc — which approach to take is a future scoping decision, not committed here.
- [x] Placebo-in-space and placebo-in-time inference (ADH permutation p-value; shipped as `placebo_space()` / `placebo_time()`)
- [x] Newey-West HAC standard errors as a `cov_type` option — shipped for `ols()`/`reg()` (v0.6.8.4) and `fe()` (period-aggregation / Driscoll-Kraay convention, validated against statsmodels `cov_nw_groupsum` + R `sandwich`); remaining: `iv()`/other estimators *(tracked separately from the open `PanelContext` `"kernel"`/`"HAC"` naming-inconsistency item)*
- [ ] API freeze candidate — no more breaking signature changes without a deprecation cycle
- [ ] Full docstring coverage + type-checked public API
- [ ] **Population-averaged GEE** (`ctx.pooled(..., method="gee")`) — equivalent to
  Stata's `xtreg, pa`. Candidate for v1.0+ if demand materialises.

#### v1.0 — Stable Release
- [ ] Semver-committed public API — breaking changes require a major version bump
- [ ] Tutorial documentation: OLS, FE, IV, DiD, RDD, PSM, synthetic control walkthroughs
- [ ] "Migrating from Stata" and "Migrating from R" guides
- [ ] Numerical parity test suite against Stata/R reference output, **published and re-run in CI on every release**
- [ ] Benchmark suite (speed vs. statsmodels/linearmodels on large panels)
- [ ] First tagged PyPI release announced beyond the initial contributor circle

---

### North Star *(vision — not a commitment)*

This is the "imagine it's five years from now" section. It's here so the
long-run shape of the project is visible, not so any individual line is a
promise with a date on it.

#### v1.x — Method Breadth
The estimator library grows outward from the causal-inference core into the
adjacent methods empirical economists actually reach for:
- **v1.1** — Quantile regression; heteroskedasticity- and outlier-robust regression (MM-estimators)
- **v1.2** — Dynamic panel breadth: Blundell-Bond system GMM, extending the existing `abond()`/GMM-core foundation *(new — this is a genuine addition, not just a reorder: the project has Arellano-Bond difference GMM but not system GMM, which is the natural next ask for anyone doing dynamic panel work, so it fills a real gap)*
- **v1.3** — Complex survey design: weighting, stratification, replicate-weight variance estimation *(up from v1.8: enormous practical pull from anyone working with household survey data; currently a large unaddressed gap)*
- **v1.4** — High-dimensional methods: LASSO/post-double-selection for inference with many controls *(up from v1.7: increasingly a default expectation in applied inference, not a specialist tool)*
- **v1.5** — Spatial econometrics: spatial lag/error models, Moran's I diagnostics *(down one from v1.2)*
- **v1.6** — ML-assisted causal inference: double/debiased ML (Chernozhukov et al.), causal forests, targeted maximum likelihood *(down from v1.3)*
- **v1.7** — Network econometrics: peer effects, network formation models *(down from v1.4)*
- **v1.8** — Structural discrete choice: BLP demand estimation, dynamic discrete choice (Rust-style) *(down from v1.5: narrower IO/labor-structural audience, high effort per estimator)*
- **v1.9** — Bayesian econometrics: Bayesian VAR, hierarchical models, MCMC-backed inference as an `inference="bayesian"` path on existing estimators *(down from v1.6: architecturally closer to a v2.0-style infrastructure question than a standalone estimator ship; keep in v1.x for now but may migrate toward the v2.0 plugin-architecture discussion later)*
- **v1.10** — Text-as-data: dictionary methods, embeddings-based regressors, econometrically-valid LLM-derived features (with measurement-error caveats explicit) *(last: lowest migration-pull, separate dependency universe)*

Design constraint carried through all of v1.x: **every new estimator ships
with a parity test against an existing reference implementation before merge.**

#### v2.0 — The Plugin Architecture: "Papers as Packages"
- [ ] `open_econs.register_estimator()` — stable plugin API so a method can use the
  `BaseModel` contract without living in core.
- [ ] **Methodology registry** — each estimator's docstring links its source paper(s)
  via a machine-readable citation block (`estimator.citation` → BibTeX).
- [ ] `open-econs-contrib` PyPI namespace (mirroring scikit-learn-contrib).
- [ ] **New-method intake template** — parity test, edge-case tests, docstring example.
- [ ] Versioned methodology: track which package version implements which version of a method.

#### v3.0 — Reproducibility Infrastructure
- [ ] `ctx.export_replication()` — data snapshot hash, exact versions, formula strings, regenerable tables.
- [ ] Journal replication-archive integration (AEA, REStat-style).
- [ ] Open **replication registry** with pass/fail badges, versioned against the paper's release.
- [ ] Automated "does this still reproduce" CI for registered replications.

#### v4.0 — Education Layer
- [ ] Notebook companion following Angrist & Pischke (*Mostly Harmless Econometrics*, *Mastering 'Metrics*).
- [ ] "Show the math" mode: `result.derivation()` prints classical closed-form numpy alongside the robust result.
- [ ] Course adoption kit + known-answer autograder built on the parity-test infrastructure.
- [ ] Localized documentation for major non-English user communities.

#### v5.0 — Community Foundation
- [ ] Formal governance: maintainer council, RFC process (modeled on NumFOCUS).
- [ ] Fiscal sponsorship / foundation status for CI + docs infrastructure.
- [ ] Annual/biennial contributor sprint.
- [ ] open-econs becomes a citable methodological standard ("estimated using open-econs v5.x, method registry entry X").

---

*The committed roadmap above is what's being built. The North Star is
deliberately larger than any one contributor can promise — that's the point.
If a piece of it excites you, [open an issue](https://github.com/qmanhbeo/open-econs/issues)
and help decide when it moves from "vision" to "roadmap."*

---

## Known Issues

- **CEM autocuts parity test operator-precedence bug** (`tests/stata/test_stata_cem_autocuts.py`, `test_summary_counts`): the expression `t == 1 & m` parses as `t == (1 & m)`, not `(t == 1) & m`, so the intended "treated AND matched" subset is not what is actually computed. **Non-blocking** — the identical buggy expression is applied symmetrically on the Stata side, so the test never turns red. The row subset actually validated may not match the author's intent. Tracked inline with `# KNOWN ISSUE:` comments at the two affected lines. Do **not** fix silently; re-confirm intent before changing.

- **Incidental fix, tracked 2026-07-11:** The `iv()` formula-parser extraction (f61cc07, done to let `gmm()` reuse `iv()`'s formula grammar) added an index intersection (`x_index.intersection(Z_instr.index)`) when building Y/X/Z that the pre-refactor code lacked. Pre-refactor, Z was built from the full dataset with no alignment check against X/y's index — meaning rows with missing instrument values but complete regressor data could have silently misaligned Z against y/X. The refactor incidentally hardens this. No parity impact on the existing Stata fixture (no missing values present), so this is unverified in either direction on a real missing-data case. Not urgent — no known user has hit this — but worth a dedicated unit test with actual missing instrument rows at some point to confirm the new behavior is correct (not just different).


