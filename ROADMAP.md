## Roadmap

open-econs exists to provide a rigorous, source-verified econometric
foundation for causal inference in policy and resource-allocation analysis.
It targets Stata/R migrators and is built in two horizons: the **committed
roadmap** (what's actually being built) and the **North Star** (long-run
vision if the project earns a community).

---

### North Star

The longer-term motivation is an RL-based resource allocation system for
small, self-sustaining communities (50–150 people, farming + solar).
`open-econs` provides the econometric toolkit that work and adjacent applied
research — urban energy burden analysis, agent-based simulation — need. The
library is also useful on its own as a Stata/R migration target, but that's
secondary to the causal-inference-for-allocation mission.

---

### Committed Roadmap (v0.1 → v1.0) — Shipped

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
- [x] Default `cov_type` changed to `HC2` (OE defaults to HC2 rather than Stata's bare `nonrobust` default because defaulting to non-robust SEs is widely considered poor applied practice; users who want exact Stata-default parity should pass `cov_type='nonrobust'` explicitly; note: Stata's `regress, robust` uses HC1, not HC2)
    - [x] Per-variable Oaxaca breakdown via `.variable_detail` and `tidy(detail=True)`
- [x] Diagnostic tests: Jarque-Bera, Breusch-Pagan, Durbin-Watson, Ramsey RESET
- [x] Condition number threshold lowered to 30 (Belsley standard); stored on result
- [x] `.export()` now supports CSV output
- [x] Wald / F-test API surface (stored statsmodels result ready for v0.3)
- [x] README example numbers fixed to match real output

#### v0.3 — Real Regression *(shipped)*
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

#### v0.4.x — Audit & IV/diagnostic hardening *(shipped)*
Collapsed from v0.4.1–v0.4.3 (each superseded by the next):
- `__delattr__`/`__setattr__` immutability guards; `object.__setattr__` bypass documented as **not prevented**.
- `iv()` rewritten with `y ~ exog | endog ~ instruments` grammar; legacy syntax fixed + `FutureWarning`; reports Cragg-Donald weak-instrument F-stat and Hansen J.
- `fe()` degrees-of-freedom corrected (iterative alternating-projections demeaning, Correia 2017) on unbalanced panels.
- Oaxaca `swap` documented honestly; `.vcov()` / `.to_latex()` / `.to_html()` added; condition-number warning excludes the intercept and column-scales the design matrix.

#### v0.5 — Design-Based Causal Inference *(shipped)*
- [x] `did()` — two-period DiD via `y ~ treated * post` with automatic DiD coefficient extraction; clustering, covariates, heterogeneous `cov_type`.
- [x] `event_study()` — event-study with `event_time` column; omits reference period; `.plot()` for pre-trend visuals.
- [x] `balance()` — covariate balance table with Welch t-tests.
- [x] **All v0.4 features preserved** — no breaking changes.
- [x] **v0.5.1 bug fixes** — FE `df_resid`/`rsd`/`adj_r2` now use the within fit's actual df; `event_study()` no longer crashes on interaction-only formulas or numeric reference periods.

#### v0.6 — Panel Data Engine *(shipped)*
- [x] First-class `PanelContext(df, entity=, time=)` — panel structure remembered, not re-specified per call. Exposes `pooled()`, `fe()`, `re()`, `diff()`, `driscoll_kraay()`, `hausman()`, plus cross-sectional delegates.
- [x] `re()` — random-effects (GLS) via `linearmodels.RandomEffects`, with `theta`, `rho`, within/between/overall R².
- [x] `hausman()` — Hausman test of FE vs RE with Möbius pseudoinverse; returns `HausmanResult`.
- [x] `driscoll_kraay()` — pooled OLS with DK (spatial/time-series-robust) SEs.
- [x] `diff()` — first-difference estimator (`FirstDifferenceOLS`).
- [x] Legacy `Context` gains panel methods delegating to a transient `PanelContext`.
- [x] **Test coverage** — numerical parity vs `linearmodels`/`statsmodels` on Grunfeld and synthetic panels; `hypothesis` property tests; edge-case tests.

#### v0.6.x — Dynamic panels, RDD, Oaxaca & Stata-parity hardening *(shipped)*
Landed across v0.6.1–v0.6.9 (patch-level numeric detail → `docs/v06x-parity-recon.md`):
- `abond()` — Arellano-Bond difference GMM (one/two-step, Windmeijer SEs, Hansen J, AR(1)/AR(2)); matches Stata `xtabond2` at machine precision (collapsed + non-collapsed, all four flavors).
- `did_cs()` — upgraded from a simplified OLS approx to the full Callaway-Sant'Anna (2021) **doubly-robust** group-time estimator (`dripw` + `reg`); 18 cell-by-cell Stata-parity tests.
- `rdd()` — sharp and fuzzy RDD via triangular-kernel LLR; `rdrobust` backend (CCT) + built-in IK fallback; McCrary/CJM `density_test()`.
- `oaxaca()` — Stata parity across all decomposition variants at machine precision.
- Newey-West `hac_adjust` (Stata-style `N/(N−K)` df correction); panel FD fixture fix.
- Full Stata-parity suite (149 tests) + `read_stata()` module-level caching (suite 235s→94s).
- Multi-way clustering and Newey-West HAC (`cov_type="HAC"`) support added.

#### v0.7 — DR Estimator, RDD Refinements *(shipped)*
- [x] `did_cs` sample-alignment & SE-gap root cause → `docs/did-cs-recon.md`.
- [x] **CS2021 doubly-robust (DR) estimator IF rewrite** (validated vs `csdid`/`did` R at machine precision; includes the `csdid_estat simple` bug warning) → `docs/cs2021-dr-recon.md`.
- [x] RDD bandwidth selection (IK built-in + CCT `rdrobust` backend, silent IK fallback); McCrary/CJM density test.
- ~~Built-in RD plot~~ — **descoped:** `rdrobust.rdplot()` already covers it; a wrapper adds no capability and conflicts with OE's "clean named outputs" scope (decision recorded 2026-07-11).
- [x] **Plot-audit findings** (OLS/EventStudy `.plot()` fate; non-blocking annotated-plot item) → `docs/plot-audit-recon.md`.

#### v0.8 — Matching & Balance *(shipped)*
- [x] `psm()` — 1:1 nearest-neighbor with replacement on logit PS (ATE). Validated vs Stata `teffects psmatch`: ATE exact to 1e-7, SE exact to 1e-6 across nn=2,5,10 (AI 2012 PS-estimation adjustment). Deviations: with-replacement only; default caliper (1.0) validated. See `tests/test_psm.py`.
- [x] Coarsened exact matching — auto-Sturges coarsening, explicit cutpoints, exact matching for categoricals, ATT weights matching Stata `cem-mata.do`; validated per-observation vs Stata `cem` on a 1000-obs fixture. **Pass 1 only** — CEM-specific SE deliberately skipped (CEM is preprocessing; `ols(weights=, cov_type="HC3")` covers inference).
- [x] **Pass 3a — Alternate auto-coarsening** (`autocuts`: `"sturges"`/`"fd"`/`"scott"`/`"ss"`) validated vs `cem-mata.do` on a 500-obs fixture.
- [x] Weighted balance diagnostics in `ctx.balance()` (SMD, variance ratio, WLS t-test); `PSMResult.balance()` / `CEMResult.balance()` wired to stored weights.
- [x] Sensitivity analysis — `rosenbaum_bounds()` + `PSMResult.sensitivity()`, validated vs Stata `rbounds` at Γ = 1, 2, 3 (p-value bounds only).

#### v0.9 — Structural Foundations & Release Candidate *(shipped)*
- [x] `gmm()` — generic linear GMM framework extracted from and validated against `abond()`'s solver (byte-identical); reuses `iv()` grammar; wired into `PanelContext`.
- [x] `nls()` — nonlinear least squares via `scipy.optimize.least_squares` with sympy analytic Jacobian (numerical fallback flagged, not silent); `white_cov()` for HC0-HC3; validated vs `curve_fit`/R `nls()`/Stata `nl`; `sympy` shipped as optional `[nls]` extra.
- [x] `mlogit()` — multinomial logit (shipped, see v0.8).
- [x] `synth()` — synthetic control (Abadie-Diamond-Hainmueller) core point estimator + `placebo_space()` / `placebo_time()` ADH permutation inference. Cross-os nondeterminism fixed (tiny L2 ridge on the inner QP) → `docs/synth-cross-os-solver-recon-update.md`.
- [x] Newey-West HAC as a `cov_type` option (`ols()`/`fe()`/`nls()`/`PanelContext.driscoll_kraay()`); HAC rolled out to all five estimators (canonical for `iv()`/`gmm()`/`did()`/`event_study()`; **project convention, not externally validated** for `did_cs()`) → `docs/did-cs-recon.md` + `docs/cov-type-recon.md`.
- [x] **Cross-estimator `cov_type` validation unified** (single `validate_cov_type` helper; `"hac"` alias only where `"HAC"` already valid) → `docs/cov-type-recon.md`.
- [x] `PanelContext` intercept-detection bug fixed (tokenizes RHS, matches `1`/`0` as standalone terms).
- [x] API freeze candidate — no breaking signature changes without a deprecation cycle (see `docs/api_stability.md`; enforced from v1.0.0).
- [x] Full docstring coverage + type-checked public API — mypy clean on all 40 source files; every public `__all__` method carries a docstring.

#### v1.0 — Stable Release *(shipped)*
- [x] Semver-committed public API — breaking changes require a major version bump (see `docs/api_stability.md`; enforced from v1.0.0).
- [x] Tutorials (full set): OLS, FE, IV, DiD, **plus RDD, PSM, synthetic control** walkthroughs in `docs/tutorials/`.
- [x] "Migrating from Stata" (updated) and "Migrating from R" (new) guides.
- [x] Numerical parity test suite against Stata/R — **re-run in CI on every release** (`ci-parity.yml`). Runs against committed fixtures with **zero skips** on free runners; live fixture *regeneration* needs self-hosted Stata/R (documented gap, not "done").
- [x] Benchmark suite (`benchmarks/ols_fe.py`).
- [x] First tagged PyPI release — published at **v1.0.0** (tag + GitHub Release + PyPI); v1.0.1 is a documentation-correction patch.

---

### Committed Roadmap (post-v1.0)

Post-v1.0 work splits into **core path** (causal-inference-critical methods
serving the allocation/policy analysis mission) and **secondary breadth**
(time-series variants and Stata-parity coverage that's useful but not
mission-critical). Both tiers ship with the same parity discipline.

#### Core Path — Causal/Allocation-Relevant Methods

These are the methods that directly serve the project's primary purpose:
causal inference for policy analysis and resource allocation. Items are
ordered by dependency (later items build on earlier ones).

##### v1.1 — Time-series econometrics *(in progress)*

**Backend strategy: wrap-and-verify over `statsmodels.tsa` + `arch` (hard
dependencies), not a from-scratch numerical reimplementation.**

> **Wrapping is an implementation strategy, NOT a parity exemption.** Every
> wrapped function still requires the same source-verified parity discipline as
> a from-scratch build.

- **New `open_econs/models/timeseries/` module** wrapping:
  - `statsmodels.tsa` — `VAR` (`.irf()`, `.fevd()`, `.test_causality()`), `VECM` + `coint_johansen` (Johansen trace/max-eigenvalue), `ARDL`/`UECM` (Pesaran-Shin-Smith 2001 `.bounds_test()`, all 5 deterministic-term cases).
  - `arch` — `unitroot` (`ADF`, `DFGLS`, `PhillipsPerron`, `KPSS`, `ZivotAndrews`); `unitroot.cointegration` (`engle_granger`, `phillips_ouliaris`); `arch_model` (full GARCH family).
- **`TimeSeriesContext`** — the `tsset` equivalent: remembers time ordering, frequency, and lag-operator conventions.
- Parity vs Stata `dfuller` / `pperron` / `kpss` / `vecrank` / `var` / `vargranger` / `fcast` / `arima` / `arch` and R `urca` / `vars` / `forecast` / `rugarch`.
- **Sub-milestones (all under committed v1.1):**
  - **1.1.0** — unit-root tests (`adf()`/`pp()`/`kpss()`/`dfgls()`/`zivot_andrews()`) + `arima()`/`arma()` + `garch()`. *(shipped)*
  - **1.1.1** — VAR/VECM + IRF/FEVD/Granger + Johansen cointegration + dual IC (Stata + Lütkepohl). *(shipped)*
  - **1.1.2** — ARDL/ECM via `statsmodels.tsa.ardl` (`ARDL`, `UECM`, `.bounds_test()`). *(shipped)*

##### v1.2 — Count & limited dependent variable models *(committed)*
New `open_econs/models/limited/` module:
- `poisson()`, `nbreg()` (NB1/NB2), `ologit()` / `oprobit()` (ordered)
- **`poisson()` / `nbreg()` MUST be FE-backed via the existing HDFE demeaning core** (Correia 2016 / Guimarães & Portugal, the `fixest::fepois` convention). Validation target is Stata `ppmlhdfe` and R `fixest::fepois` at the option level, not just coefficient equality.
- `tobit()` — MLE (statsmodels has no Tobit); validate vs R `censReg` / `AER::tobit`
- `heckman()` — selection model (two-step + MLE); validate vs R `sampleSelection`
- `feglm` binomial FE absorption — **decision required, NOT in v1.2 as scoped**; flag in `FUTURE_WORK.md`.
- First-class `.margins()` / `.predict()`; parity vs Stata `poisson` / `nbreg` / `tobit` / `heckman` / `ologit` / `oprobit`
- Ships at **Beta** in the maturity table; `tobit` / `heckman` backend recon recorded under "Explicitly Deferred Estimators" if it blocks the release

##### v1.3 — Diagnostics: build missing + test *(committed)*
Promote the existing JB / Breusch-Pagan / Durbin-Watson / Ramsey RESET / VIF into
a consistent first-class result API, and implement the missing `estat` battery:
- `bg_test()` (Breusch-Godfrey), `white_test()`, `ljung_box()`
- `cooks_distance()`, `leverage()`, `dfbetas()`, `influence()`
- `result.diagnostics()` summary `DataFrame`
- Parity vs Stata `estat hettest` (BP/White), `estat bgodfrey`, `estat ovtest` (RESET), `estat vif`, `predict, cooksd` / `dfbeta`, `sktest` / `swilk`, `estat archlm`

#### Secondary — Stata-Parity Breadth

These items broaden the library's Stata/R coverage but are not
causal-inference-critical. They ship when a specific downstream need pulls
them forward, or when capacity allows after core-path items are complete.

##### v1.x — Time-series breadth (secondary)
- ARDL/UECM bounds test (v1.1.2, shipped — technically core-path but
  time-series-specific, so listed here for visibility)
- Additional VAR/VECM variants (structural VAR, Bayesian VAR)
- Additional IC conventions beyond the two already implemented
- Ng-Perron sequential lag selection for DFGLS (TS-2 gap, tracked in `FUTURE_WORK.md`)
- Legacy CV-table ports (Fuller/ERS/ZA small-N tables, TS-1 gap)

##### v1.x — Method breadth (aspirational; no committed version)
- **v1.4** — Quantile regression; heteroskedasticity- and outlier-robust regression (MM-estimators)
- **v1.5** — Dynamic panel breadth: Blundell-Bond system GMM, extending the existing `abond()`/GMM-core foundation
- **v1.6** — Complex survey design (`.svy`): weighting, stratification, replicate-weight variance estimation
- **v1.7** — High-dimensional methods: LASSO/post-double-selection for inference with many controls
- **v1.8** — Spatial econometrics: spatial lag/error models, Moran's I diagnostics
- **v1.9** — ML-assisted causal inference: double/debiased ML (Chernozhukov et al.), causal forests, targeted maximum likelihood
- **v1.10** — Network econometrics: peer effects, network formation models
- **v1.11** — Structural discrete choice: BLP demand estimation, dynamic discrete choice (Rust-style)
- **v1.12** — Bayesian econometrics: Bayesian VAR, hierarchical models, MCMC-backed inference
- **v1.13** — Text-as-data: dictionary methods, embeddings-based regressors, econometrically-valid LLM-derived features

Design constraint carried through all of v1.x: **every new estimator ships
with a parity test against an existing reference implementation before merge.**

#### Done post-1.0 (deferred v1.0 stubs, now completed)

- **RDD / PSM / synthetic-control TUTORIALS** — the estimators (`rdd()`,
  `psm()`/`cem()`, `synth()`) shipped in v1.0; the `docs/tutorials/` walkthroughs
  (`rdd.md`, `psm.md`, `synth_control.md`) are now written and locally
  smoke-tested. Pure documentation, no estimator code changed, no new parity
  work — local runnability (not Stata/R numerical parity) is the validation bar,
  and each tutorial states its known limitations honestly. A tutorials index was
  added at `docs/tutorials/README.md`.

- **`oe.placebo_space` / `oe.placebo_time` top-level exports** — the two ADH
  permutation-inference helpers were previously only reachable via the submodule
  path `from open_econs.models.causal.placebo import ...`. They are now added to
  the top-level `open_econs/__init__.py` exports (and `__all__`), so
  `oe.placebo_space` / `oe.placebo_time` work directly. The `synth_control.md`
  tutorial was updated to use the top-level path and drop the submodule-import
  note. No estimator logic changed. (Source-only + doc; no release.)

- **v1.1.2 — ARDL / UECM + PSS(2001) bounds test (implemented; release
  pending version bump)** — `ardl_fit()` / `uecm_fit()` wrapping
  `statsmodels.tsa.ardl` (`ARDL` / `UECM`), plus a `.bounds_test(case)`
  computing the Pesaran-Shin-Smith (2001) **F**- and **t**-bounds tests over
  all 5 deterministic cases, long-run multipliers, and the error-correction
  term. Parity anchored to the published PSS(2001) tables (`cv_vintage=
  "pss2001"`, default) with `"statsmodels"` (simulated) as a documented toggle.
  - Cross-tool parity to **1e-6** against Stata SSC `ardl` (14 tests) and R
    `ARDL` (10 tests) on the canonical Pesaran denmark example; F-stat / t-stat
    / EC term / LR multipliers / all critical values (incl. 2.5%) match.
    Conventions source-verified against `ardl.ado` / `ardlbounds.ado` and the
    R `ARDL` package source (rule 1).
  - **Footgun fixed & recorded (rules 16, 18):** an apparent ~1e-5 Stata-only
    gap was root-caused to `import delimited` reading numeric columns as
    single-precision `float`; fixed with `set type double` in the generator.
    OE was never wrong. See `methodology/timeseries/ardl.md`.
  - Full suite green (1099 passed, excluding `synth_placebo`).

- **v1.0.3 — Performance hardening (shipped, 2026-07-17)** — Python-strength
  audit of the hot loops, all **bit-identical** to the prior scalar code (no
  parity tolerance loosened; new determinism tests guard each one):
  - `did_cs` bootstrap / permutation reps parallelized via an opt-in
    `parallel=` `ProcessPoolExecutor` (bit-identical). Commits `3f56aea`,
    `3eb0e92`.
  - `psm` fully vectorized: batched `cKDTree` k-NN, padded `(n,h)` / `(n,h,p)`
    tensor reductions for `xi2` and `c_tau`, vectorized `matched_arr`. **~4×
    faster** on the Stata `teffects psmatch` fixture (nn=10: 0.50s → 0.13s).
    Commit `cdb15be`.
  - `_gmm_core._hac_S` Newey-West lag accumulation vectorized into a single
    batched `einsum` (feeds `abond` / `gmm` VCE + Hansen J). Commit `897c31a`.
  - **GPU offload (CuPy / CUDA) deliberately declined**: SciPy optimizers have
    no GPU backend and BLAS matmuls are already CPU-multithreaded; transfer
    overhead dominates at current fixture sizes. Rationale in
    `methodology/performance-conventions.md`. Revisit only at 100k+ entities.
  - Full suite green (1048 passed, excluding `synth_placebo`) before release.

---

### North Star (vision — not a commitment)

This is the "imagine it's five years from now" section. It's here so the
long-run shape of the project is visible, not so any individual line is a
promise with a date on it.

#### v2.0 — The Plugin Architecture: "Papers as Packages"
- [ ] `open_econs.register_estimator()` — stable plugin API so a method can use the `BaseModel` contract without living in core.
- [ ] **Methodology registry** — each estimator's docstring links its source paper(s) via a machine-readable citation block (`estimator.citation` → BibTeX).
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

---

## Explicitly Deferred Estimators

- `nlogit()` — nested logit. **Recon complete** (`docs/nlogit-recon.md`, branch
  `feature/nlogit` off `db2dfe5`); **implementation NOT built**. Blockers: (a) R
  `mlogit` cannot run a full Stata-equivalent spec — nest-level covariates cause
  singularity on `webuse restaurant`, so it is only a simple-spec reference, not
  the full one; Stata alone is the primary reference. (b) No validated fixture
  with τ∈(0,1) (needs a genuinely nested, non-degenerate case to lock the
  estimator). (c) The analytic gradient (~200 lines of recursive tree traversal,
  Stata's `_dldtau`/`_dldx` Mata) needs a domain-expert implementation; a wrong
  gradient silently yields wrong SEs. Also: no library backend exists
  (statsmodels has only `MNLogit`), so it is a from-scratch build of ~500–800
  lines + ~300 lines parity tests. Do NOT start until these are resolved.

- `psm()` kernel / smooth-weight matching — INVESTIGATED 2026-07-11, deliberately
  NOT built in v0.8 (deferred to its own scoped, parity-validated pass). Full
  investigation → `docs/psm-kernel-recon.md`.

- `cem()` k2k matching + L1 balance diagnostics — Pass 3, investigated,
  deliberately not built (see prior investigation report).

## Known Limitations (not implemented)

- `synth()` — `plot()` and `predict()` (out-of-sample counterfactual) are **not
  implemented**: `SynthResult` defines no such methods. The `synth` module
  docstring notes they are intentionally out of scope for the core pass and are
  a separate, later-scoped task.
