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
  entity/time dummies. Two-way FE uses iterative alternating-projections demeaning
  (Correia 2017) for correct estimates on unbalanced panels. ``rsd``, R², adjusted
  R² all use the corrected df.
- [x] **Immutability claim corrected** — ``__setattr__`` / ``__delattr__`` guard normal
  accidental mutation; ``object.__setattr__`` bypass is documented as
  **not prevented** (impossible in pure Python without a C extension).
- [x] **Context.ols defaults unified** — changed from ``HC1`` to ``HC2`` to match
  the top-level ``ols()`` function.
- [x] **Condition-number warning** — now excludes the intercept column and
  column-scales the design matrix before computing the condition number,
  reducing false positives for routine regressions with a constant.

> ⚠️ v0.4.2 legacy IV syntax omitted exogenous controls from the instrument
> matrix — a classic 2SLS error producing wrong coefficients. v0.4.3 fixes this.

#### v0.4.3 — Instrument-Matrix Correction *(superseded by v0.5)*
- [x] **Legacy IV syntax fixed** — ``y ~ x1 + x2 | z1`` now constructs the
  instrument matrix as ``[x1, x2, z1]`` (not just ``[z1]``), matching the
  textbook 2SLS requirement that included exogenous regressors must also
  appear as instruments.
- [x] **FutureWarning added** — legacy syntax now warns users to adopt
  ``y ~ exog | endog ~ instruments`` for clarity.
- [x] **New syntax unchanged** — ``y ~ w | x ~ z`` correctly passes
  ``w`` as exogenous, ``x`` as endogenous, ``z`` as instrument.

#### v0.5 — Design-Based Causal Inference *(current)*
- [x] `did()` — two-period difference-in-differences via ``y ~ treated * post``
  with automatic DiD coefficient extraction. Supports clustering, covariates,
  and heterogeneous cov_type (HC0-HC3, nonrobust). Returns ``DiDResult``
  with ``.did_coefficient``, ``.did_std_error``, ``.did_t_stat``, ``.did_p_value``.
- [x] `event_study()` — event-study specification with ``event_time`` column.
  Omits user-specified reference period (default -1). Returns ``EventStudyResult``
  with ``.event_coefficients`` DataFrame and ``.plot()`` for pre-trend visuals.
- [x] `balance()` — covariate balance table comparing treatment/control means
  with Welch t-tests. Accessible as ``ctx.balance(treatment="treated")``
  or standalone ``oe.balance(df, treatment="treated")``.
- [x] **All v0.4 features preserved** — no breaking changes to OLS, FE, IV, logit, probit.
- [x] **v0.5.1 bug fixes** — fixed FE `df_resid`/`rsd`/`adj_r2` being inconsistent with
  the reported standard errors (they now use the within fit's actual degrees of
  freedom, matching a manual group-demeaned OLS reference); fixed `event_study()`
  crashing on interaction-only formulas like ``y ~ treated * post`` and on the
  numeric `Treatment` reference period (the covariance RHS is now parsed by
  splitting on top-level `+` terms rather than fragile string surgery).

#### v0.6 — Panel Data Engine *(shipped)*
- [x] First-class `PanelContext(df, entity=, time=)` — panel structure remembered,
  not re-specified per call. Exposes `pooled()`, `fe()`, `re()`, `diff()`,
  `driscoll_kraay()`, `hausman()`, plus cross-sectional delegates (`ols`, `logit`,
  `probit`, `did`, `event_study`, `balance`).
- [x] `re()` — random-effects (GLS) estimator via `linearmodels.RandomEffects`,
  returning a `RandomEffectsResult` with `theta`, `sigma2_u`/`sigma2_e`, `rho`, and
  within/between/overall R². An explicit intercept is always included.
- [x] `hausman()` — Hausman test of FE vs RE consistency:
  `H = (b_fe − b_re)'(V_fe − V_re)⁺(b_fe − b_re) ~ χ²(k)`, with Möbius
  pseudoinverse for non-positive-definite differences. Returns a `HausmanResult`
  with `statistic`, `p_value`, `df`, and `rejected_at(alpha)`.
- [x] `driscoll_kraay()` — pooled OLS with Driscoll-Kraay (spatial/time-series-robust)
  standard errors via `linearmodels.PooledOLS(cov_type="kernel")`.
- [x] `diff()` — first-difference estimator (`FirstDifferenceOLS`), returned as a
  `FirstDifferenceResult` (an `OLSResult` tagged `method="first-difference"`).
- [x] Legacy `Context` gains panel methods (`fe`, `re`, `pooled`, `diff`,
  `driscoll_kraay`, `hausman`) that delegate to a transient `PanelContext`.
- [x] **Excessive test coverage** — numerical parity vs `linearmodels`/`statsmodels`
  on the Grunfeld dataset and deterministic synthetic panels; `hypothesis`
  property tests (FE slopes match within-OLS reference, RE `theta ∈ [0,1]`,
  Hausman statistic ≥ 0, etc.); edge-case tests (single entity/time, duplicate
  (entity,time) index, constant outcome, singular Hausman difference); context
  delegation and result-API tests. See `tests/test_panel_*.py`.
- [x] **v0.6.1–v0.6.3** — see changelog below.

#### v0.6.1 — Dynamic panels, staggered DiD, RDD, robust SEs
- [x] `abond()` — Arellano-Bond difference GMM (one/two-step, Windmeijer SEs,
  Hansen J overidentification test, AR(1)/AR(2) serial-correlation tests).
- [x] `staggered_did()` — Callaway-Sant'Anna (2021) staggered / heterogeneous-timing
  DiD with group-time ATTs, event-study aggregation, and entity-clustered SEs.
- [x] `rdd()` — sharp and fuzzy regression discontinuity via triangular-kernel
  local linear regression.
- [x] Multi-way clustering (`cluster=["firm","year"]`, Cameron-Gelbach-Miller
  minik estimator) and Newey-West HAC (`cov_type="HAC"`).
- [x] RE `llf`/`aic`/`bic` now populated; `RandomEffectsResult.theta` is a scalar.
- [x] `Context.pooled()` defaults to panel-robust (cluster-by-entity) inference.
- [x] DRY: `_capture_call` centralized; `to_latex()`/`to_html()` no longer require
  `jinja2`; bug fixes from the v0.6.0 review (FE demeaning, docstrings).

#### v0.6.2 — Numerical stability & patch release
- [x] Robust pseudo-inverse for `abond()` weighting matrices — stabilizes
  tiny-panel and near-singular cases.
- [x] Test coverage expanded for edge-case panels.

#### v0.6.3 — Correctness fixes & applied-econometrics improvements
- [x] **Cragg-Donald Wald F-stat** for multi-endogenous IV — now uses
  `linearmodels`' exact `cragg_donald_stat` instead of `min(first_stage_F)`.
- [x] **FE intercept stripping** — boolean indexing by column name, not
  position, so formulaic column ordering changes cannot silently break output.
- [x] **FE `cov_type` default** unified to `"HC2"` (matching `ols()` and
  modern Stata).
- [x] **Windmeijer (2005) two-step correction** — replaced centered-moment
  sandwich with the proper Windmeijer additive correction for AB GMM.
- [x] **Collapsed instruments** for `abond()` (`collapse=True` default) —
  one instrument per lag depth (Roodman 2009), mitigating the "too many
  instruments" problem that biases two-step SEs downward.
- [x] **IK bandwidth** for `rdd()` — Imbens-Kalyanaraman (2012) MSE-optimal
  bandwidth replaces Silverman's density rule; Silverman kept as fallback.
- [x] **Honest staggered-DiD labeling** — renamed to "simplified OLS
  approximation", added `bootstrap` parameter, docstring caveats.
  ▶ *Superseded by v0.6.8.2 — the estimator now implements the full
  CS2021 doubly‑robust estimator (not a simplified approximation).*
- [x] **Vectorized `_within_transform()`** — pandas groupby replaces Python
  loop for speed on large panels.
- [x] **Oaxaca `get_dummies` fix** — uses `formulaic` `ensure_full_rank`
  instead of fragile column-name pattern matching.
- [x] **FE `vcov()` consistency** — `fe()` now stores a df-scaled covariance
  matrix (`_cov`) so `vcov()` returns values consistent with the panel-adjusted
  standard errors. Previously `se² ≠ diag(vcov())` by a factor of
  `(N−k)/(N−g−k)`.
- [x] **Hausman test fixed** — three bugs resolved:
  (1) `PanelContext.fe()` now supports `time=None` override via sentinel
  pattern to force one-way FE (matching Stata's `xtreg, fe`);
  (2) `vcov()` returns the df-corrected covariance so `V_fe − V_re` is
  positive definite (was negative definite, causing `H = 0.0` clamping);
  (3) Stata's `e(chi2)` is a ghost variable (stores 667.36 while the
  displayed statistic is 0.29) — the `.do` file now computes the
  displayed quadratic form directly.

#### v0.6.5 — Arellano-Bond / `xtabond2` numerical parity
- [x] **`abond()` collapsed one-step non-robust now matches Stata `xtabond2`**
  to ~1e-7 on coefficients *and* the full variance-covariance matrix. Verified
  on the 30×5 `df_panel` fixture against `xtabond2 y L.y x z, gmm(L.y, lag(2 4)
  collapse) iv(x z) nolevel small`.
- [x] **GMM instrument lag fixed** — Stata's `gmm(L.y, lag(a b))` instruments
  are lags *of the variable L.y* (`y_{t-lags-lag}`), not `y_{t-lag}`. The
  previous `y[j-lag]` construction was off by one lag and silently used the
  initial observation `y₀` as an instrument (Stata never does). This was the
  dominant source of the VCV gap.
- [x] **Weighting-matrix `H` corrected** — `H = M'M` (the first-difference
  operator) has diagonal **2** (off-diagonal −1) on usable equations, not the
  previously-hardcoded 3. The `svmat`-saved `e(H)` is a normalized view; the
  true unnormalized `M'M` is what the one-step weighting `W = (Z'HZ)⁻¹` uses.
- [x] Ground-truth extraction tooling added (`tests/stata/do/abond_gt_matrices.do`
  + `abond_gt_*.csv`) that pulls Stata's exact per-observation `X`, `Y`, `Z`,
  `H` matrices via `svmat` for element-by-element validation.
- [x] Parity test `tests/stata/test_stata_abond.py::TestAbondCollapsedOneStep`
  (coefficients, standard errors, dimensions) added.

#### v0.6.6 — Non-collapsed / full GMM instruments (Arellano-Bond)
- [x] **Non-collapsed (uncollapsed) `abond()`** — full GMM-style instrument
  expansion (block-diagonal staircase, one column per (depth × usable period)),
  matching Stata's `_MakeGMMinsts` / `_Explode` from `xtabond2` 3.7.2.
- [x] **Block builder isolated** as `_build_noncollapsed_gmm_block(var, depth, T,
  lag_offset)` — handles both L.y (`lag_offset=lags`) and predetermined regressor
  (`lag_offset=0`) conventions, structurally zero columns omitted.
- [x] **`n_gmm_i` formula corrected** — accounts for `lag_offset` per variable type:
  L.y columns = `max(0, T-d-lags)`, gmm_c columns = `max(0, T-d)`.
- [x] **AR-test Z dispatch** — non-collapsed branch reuses the same block-builder
  construction as the estimation path (eliminating the stale collapsed-formula
  path that produced wrong Z matrices for AR tests).
- [x] **All four flavors validated against Stata `xtabond2` at machine precision**
  (~1e-8):
  - One-step non-robust: `b = [-0.08671378, 1.14723439, -0.30353782]`
  - Two-step non-robust: `b = [-0.09296598, 1.12778676, -0.29591272]`
  - One-step robust: coefficients identical, `se = [0.21161410, 0.15665591, 0.09549873]`
  - Two-step robust: `se = [0.23358608, 0.17061935, 0.10576369]`
- [x] **40 Stata-parity tests** — 20 collapsed (no regression) + 20 new non-collapsed.
- [x] Ground-truth `.do`/`.dta` fixture (`abond_noncollapsed.do`) captures all four
  flavors at full double precision.
- [x] **Collapsed path untouched** — no regressions.

#### v0.6.7 — Oaxaca Stata parity & advanced options
- [x] **Stata parity fix** — `.do` files mis-extracted the `e(b)` matrix (columns 1/2/3
  instead of 3/4/5), causing all 3 Stata-parity tests to fail against group means instead
  of decomposition components. Fixed both `oaxaca_two_fold.do` and `oaxaca_three_fold.do`;
  regenerated `.dta` fixtures. OE now matches Stata's actual `pooled` and `threefold`
  output to machine precision.
- [x] **`reference` parameter** (two-fold) — `"pooled"` (default), `"omega"` (pooled without
  group dummy), `"group1"`/`"group2"` (single-group coefficients), or a float (0–1) custom
  weight. Maps to Stata's `pooled`, `omega`, and `weight()` options.
- [x] **`reverse` parameter** (three-fold) — `reverse=True` uses Group 1 coefficients as
  the reference (Stata's `threefold(reverse)`). Components sum to gap; variable-level
  detail matches.
- [x] **21 new tests** at `1e-12` tolerance (component consistency, var-detail sums, string
  ↔ float alias equivalence, invalid-reference error, reverse three-fold cross-checks)
  — 96 Oaxaca tests total, all passing.

#### v0.6.8 — RDD with rdrobust backend + event_study fix
- [x] **RDD rdrobust backend** — delegates to `rdrobust` Python package for CCT
  bandwidth selection, separate-side local linear estimation, and NN cluster-robust
  variance. Requires `pip install open-econs[rd]` (rdrobust >= 2.0).
- [x] **RDD built-in fallback** — IK bandwidth (Imbens-Kalyanaraman 2012), 
  nearest-neighbour or EHW variance. Works without rdrobust dependency.
- [x] **event_study() fix** — gracefully falls back to the first available period
  when the specified `omitted_period` is not in the data, instead of crashing.
- [x] **Removed faulty Stata event study test** — the test was incorrectly comparing
  DiD intercept to event study intercept (different parameterizations). Rely on
  comprehensive unit tests in `test_event_study.py`.

#### v0.6.8.1 — Logit/Probit AME correction & Stata parity
- [x] **`logit().margins()` / `probit().margins()` now compute AME** — changed
  `get_margeff(at="mean")` to `get_margeff(at="overall")`, matching Stata's
  default `margins, dydx(*)` (average marginal effects instead of marginal
  effects at the mean).
- [x] **Fixed logit margins Stata fixture** — `logit_margins.do` was using
  `_b[x1]` which retrieves the raw logit coefficient, not the marginal effect.
  Changed to `r(b)[1,1]` (same pattern as `probit_margins.do`). Regenerated
  `.dta` via StataMP.
- [x] **Tightened test tolerance** — `rtol=0.8` → `rtol=1e-6` for both logit
  and probit margins tests. Both now match Stata at machine precision.
- [x] **Removed faulty Stata event study test** — was comparing DiD intercept
  to event study intercept (different parameterizations by construction).
  Rely on unit tests in `test_event_study.py`.

#### v0.6.8.2 — CS2021 Doubly‑Robust Staggered DiD & cell‑by‑cell Stata parity
- [x] **`staggered_did()` now implements the full Callaway & Sant'Anna (2021)
  doubly‑robust group‑time estimator.** Previously a simplified OLS
  approximation; now uses the canonical `dripw` formulation: per‑cell logit
  propensity score + OLS outcome regression (controls at base period) +
  inverse‑probability‑weighted RIF, with centered‑IF cluster‑robust SEs.
- [x] **`covariates` and `method` parameters** — `method="dripw"` (DR, default
  when covariates given) or `method="reg"` (outcome regression / OLS within
  each cell). Without covariates, DR collapses to the simple mean‑difference
  (backward compatible; default `method="reg"`).
- [x] **Cell‑by‑cell Stata parity** — 18 Stata‑parity tests (13 cell‑by‑cell
  with covariates + 5 no‑covariates sanity checks). Individual ATT(g,t)
  coefficients match at rtol=1e‑6; SEs compared at rtol=0.2 (balanced) /
  rtol=0.6 (unbalanced), acknowledging csdid's full‑sample IF rescaling.
  Balanced and unbalanced‑cohort fixtures both verified. All 21 staggered‑DID
  tests pass.
- [x] **Unbalanced‑cohort fixture added** — `df_panel_unbalanced.csv` with
  three cohort groups (never‑treated, g=3, g=5) and corresponding
  `staggered_did_unbalanced.do` for reference extraction.
- [x] **Weight‑formula audit** — confirmed csdid uses cohort‑proportional
  weights (not uniform): `weight(g,t) = P(G=g) / Σ_i P(G=i) × n_times(i)`.
  Verified against Stata's `makerif2` source.
- [x] Removed "simplified OLS approximation" framing from all docstrings.
  The `staggered_did()` docstring now states "Implements the Callaway &
  Sant'Anna (2021) group‑time estimator with doubly‑robust inference."

#### v0.6.8.3 — Full Oaxaca Stata parity (all decomposition variants)
- [x] **Expanded `oaxaca_two_fold.do`** — extracts gap, explained, and
  unexplained for all four two‑fold reference variants: ``pooled``,
  ``omega``, ``weight(1)`` (group‑1 coefficients), ``weight(0)`` (group‑2
  coefficients). Prior fixture only stored ``pooled`` gap + explained.
- [x] **Expanded `oaxaca_three_fold.do`** — extracts gap, endowment,
  coefficients, and interaction for both default and ``threefold(reverse)``.
  Prior fixture only stored the gap.
- [x] **20 new Stata‑parity tests** (6 test classes) — every decomposition
  component vs Stata ``oaxaca`` v4.1.1 (``e(b)[1,3–6]``, verified against
  source).  Tolerances: gap/explained/unexplained rtol=1e‑6; interaction
  atol=1e‑6 (near‑zero floor).
- [x] **Stata↔OE terminology documented** — both ``oaxaca()`` and
  ``OaxacaResult`` docstrings now carry a cross‑reference table mapping
  Stata's ``e(b)`` names (``gap``, ``explained``, ``unexplained``,
  ``endowment``, ``coefficients``, ``interaction``) to OE attributes.
  Includes explicit note that three‑fold ``.explained``/``.unexplained``
  *are* endowment/coefficients, not the two‑fold meaning of those terms.
- [x] All Oaxaca unit tests continue to pass unchanged.

#### v0.6.8.4 — Newey‑West `hac_adjust` parameter + Panel FD fixture fix
- [x] **`ols(hac_adjust=True)` / `newey_west_cov(adjust=True)`** — opt‑in Stata‑style
  `N/(N−K)` degrees‑of‑freedom correction for HAC standard errors. Default remains
  `False` (original Newey‑West 1987 formula, matching statsmodels and R `sandwich`).
  Full comparison table (NW1987, OE, statsmodels, R sandwich, Stata, MATLAB) in
  `ols()` docstring.
- [x] **Stata parity for `hac_adjust=True`** — three new tests
  (`TestOLSHACAdjust`, `test_hac_adjust_lags0_matches_hc1`,
  `test_hac_adjust_panel_cluster`) verify the correction at rtol=1e‑7.
- [x] **Panel FD fixture corrected** — Stata's `regress dy dx dz` implicitly includes
  an intercept (Stata default). The correct first‑difference model has no intercept
  (the fixed effects cancel out). Changed to `regress dy dx dz, noconstant`;
  regenerated `panel_fd.dta` via StataMP. Test tolerances tightened from
  `rtol=1e‑2` to `rtol=1e‑6`, matching FE/Pooled/RE parity.

#### v0.6.9 — Full Stata‑parity coverage & test‑suite caching
- [x] **Event‑study Stata parity** — new synthetic fixture (seed 11, 600 obs),
  rewritten `.do` file matching `reg y post if treated==1, vce(hc2)`. Fixed
  t‑distribution inference (`cov_kwds={"use_t": True}`) to match Stata's
  default. 4 tests at rtol=1e‑6 (p‑values at rtol=1e‑4).
- [x] **All 8 ABOND flavors live‑verified** — replaced hardcoded constants with
  `read_stata("abond")` lookup. 40 tests (8 flavors × coefficients/SEs/AR1/AR2)
  all at rtol=1e‑6. Non‑collapsed matched at ~1e‑16; collapsed at ~3e‑9
  (truncation only, no transcription errors). Unifies and supersedes
  v0.6.5–v0.6.6 manual‑extraction regime.
- [x] **Module‑level `read_stata()` caching** — every test file now loads Stata
  results once at module level rather than per‑test‑method. 22 unique Stata
  invocations instead of ~50+. Suite runtime: 235s → 94s (2.5× speedup).
- [x] **Logit/probit cached** — final test file converted (4 module‑level labels,
  7 → 4 Stata calls).
- [x] **All 149 Stata‑parity tests pass.**
- [x] **Deferred: staggered‑DID live‑`read_stata()` conversion** — held back
  until the CS2021 doubly‑robust rewrite (v0.7) lands, because:
  - The DR rewrite changes every coefficient/SE under test; wiring up
    `read_stata()` now produces ground truth that goes stale within the
    same sprint.
  - SE tolerances (rtol=0.2 / 0.6) mask a known SE‑methodology gap (OLS
    without influence‑function correction) that the DR fix is designed to
    close. Wrapping the gap in `read_stata()` would relabel an open bug
    as infrastructure maturity.
  - `.do`‑file sample‑alignment fix — **done as a separate isolated pass
    (2026‑07‑10)**, ahead of the DR build, because it's a clean negative
    result that root‑causes half the SE‑gap ambiguity (see v0.7.0). The DR
    rewrite still lands separately; the alignment did not touch estimator
    code or tolerances.
  - *See session handoff for action‑item details.*

#### v0.7 — Causal Inference: DR Estimator, RDD Refinements *(current)*
- [x] **v0.7.0 — staggered_did sample‑alignment & SE‑gap root cause** *(done)*
  - **`.do` sample‑alignment** — `staggered_did.do` and `staggered_did_unbalanced.do`
    now drop the gvar=5 entities (which never turn on: max time=4 < treatment
    time=5) to match the Python‑side `entity < 20` / `entity < 23` filters exactly.
    Regenerated both `.dta` fixtures via StataMP (csdid, dripw). `e(N)` drops
    150→100 (balanced) and 150→115 (unbalanced), confirming the entities were
    actually excluded.
  - **SE‑gap root cause resolved (half of the v0.6.9 ambiguity)** — after
    alignment, the g=3 ATT(g,t) and SEs are unchanged beyond floating‑point
    noise; only the csdid weights move (0.125→0.25 balanced, 0.133→0.25
    unbalanced) because fewer cells share the mass. This **positively attributes
    the rtol=0.2/0.6 SE gap to `makerif2` full‑sample IF rescaling vs OE's
    per‑cell IF — NOT to sample mismatch.** The DR‑adjacent SE‑methodology work
    remains to close the gap; the sample‑alignment half is closed.
  - **Test constants synced** — `_CELL_B`/`_CELL_SE`/`_CELL_W` and the
    unbalanced `_B`/`_SE`/`_W` updated to the new sample‑aligned Stata values.
    All 18 staggered‑DiD Stata‑parity tests pass at the **unchanged** tolerances
    (no loosening). `TestStaggeredDiDNoCovariates` untouched; the
    `read_stata()` live‑conversion remains deferred to the DR rewrite.
  - [x] **v0.7 — CS2021 doubly‑robust (DR) estimator IF rewrite** *(done)*
   - **Formula sourced & validated from R `DRDID`** — `drdid_panel` (the `trad`
     method, which is `csdid`'s default `dripw`; confirmed via `csdid.ado`
     "change default to dripw from drimp").  The exact influence function was
     transcribed and validated to machine precision against `csdid`'s own saved
     per‑entity RIF (`staggered_did_rif_save.dta`): per‑cell IF corr 1.00000
     (max centered diff ~1e-6) for all 4 cells; per‑cell SE matches exactly
     (cell (3,3) → 0.4652265, cell (3,4) → 0.4941999); aggregated SE matches
     exactly (0.41781627).
   - **Influence function** (per entity, per cell):
     `att_inf_func = inf_treat - inf_control`, with
     `inf_treat = (dr_att_treat - w_treat·eta_treat - asy_lin_rep_wols·M1)/mw_treat`
     and `inf_control = (dr_att_cont - w_cont·eta_cont + asy_lin_rep_ps·M2
     - asy_lin_rep_wols·M3)/mw_cont`.  The `asy_lin_rep_ps·M2` (PS‑estimation)
     and `asy_lin_rep_wols·M3` (OLS‑estimation) corrections were the missing
     terms in the old plug‑in IF — `inf_cont_2` exists only on the control side,
     which is why treated units previously matched (corr 0.9985) but control
     units did not (corr 0.0019).
   - **Per‑cell SE**: `V = Σ_i (RIF_i - mean(RIF))² / N²`, `N` = full‑sample
     entity count, no `n/(n-1)` correction.  Stored `RIF` is shifted so
     `mean(RIF) = ATT(g,t)`.
   - **Aggregated SE** (the other half of the old gap): full‑sample weighted
     average of per‑entity RIFs across post‑treatment cells with equal weight
     `1/K`, then `V = Σ_i (agg - mean)² / N²` → 0.41781627 (was `sqrt(mean se²)`
     = 0.4799, which was wrong).
   - `method="reg"` keeps the prior 2×2 OLS cluster‑SE path (no per‑entity RIF).
   - Parity now holds at **rtol=1e-6** (was 0.2 balanced / 0.6 unbalanced):
     balanced per‑cell SEs and aggregated SE exact; unbalanced per‑cell SEs also
     exact (~1e-8).  All 18 staggered‑DiD Stata‑parity tests pass.
    - References: see README "References" (Callaway & Sant'Anna 2021;
      Sant'Anna & Zhao 2020; R `DRDID` package).
  - [ ] `test_stata_staggered_did.py` converted to live `read_stata()` comparison
      (still deferred — separate task, not bundled with the IF rewrite)
  - [ ] Bandwidth selection (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik)
  - [ ] McCrary density test for manipulation at the cutoff
  - [ ] Built-in RD plot (binned scatter + fitted lines either side of cutoff)

#### v0.8 — Matching & Balance
- [ ] `psm()` — propensity score matching (nearest-neighbor, caliper, kernel)
- [ ] Coarsened exact matching
- [ ] Post-matching balance diagnostics reusing `ctx.balance()` from v0.5
- [ ] Sensitivity analysis (Rosenbaum bounds)

#### v0.9 — Structural Foundations & Release Candidate
- [ ] `gmm()` — general GMM estimation framework other estimators can build on
- [ ] Nonlinear least squares
- [ ] Discrete choice groundwork (multinomial logit, nested logit)
- [ ] `synth()` — synthetic control (Abadie-Diamond-Hainmueller)
- [ ] Placebo-in-space and placebo-in-time inference
- [ ] Newey-West HAC standard errors as a `cov_type` option across estimators
- [ ] API freeze candidate — no more breaking signature changes without a deprecation cycle
- [ ] Full docstring coverage + type-checked public API
- [ ] **Population-averaged GEE** (`ctx.pooled(..., method="gee")`) — equivalent to
  Stata's `xtreg, pa`. Requires its own working-correlation-structure and sandwich-SE
  implementation; distinct from pooled OLS (`ctx.pooled()` default). Open question
  whether this is needed for current user base — candidate for v1.0+ if demand materialises.

#### v1.0 — Stable Release
- [ ] Semver-committed public API — breaking changes require a major version bump
- [ ] Tutorial documentation: OLS, FE, IV, DiD, RDD, PSM, synthetic control walkthroughs
- [ ] "Migrating from Stata" and "Migrating from R" guides
- [ ] Numerical parity test suite against Stata/R reference output, **published and re-run in CI on every release**
- [ ] Benchmark suite (speed vs. statsmodels/linearmodels on large panels)
- [ ] First tagged PyPI release announced beyond the initial contributor circle

---

---

### North Star *(vision — not a commitment)*

This is the "imagine it's five years from now" section. It's here so the
long-run shape of the project is visible, not so any individual line is a
promise with a date on it.

#### v1.x — Method Breadth
The estimator library grows outward from the causal-inference core into the
adjacent methods empirical economists actually reach for:
- **v1.1** — Quantile regression; heteroskedasticity- and outlier-robust regression (MM-estimators)
- **v1.2** — Spatial econometrics: spatial lag/error models, Moran's I diagnostics
- **v1.3** — ML-assisted causal inference: double/debiased ML (Chernozhukov et al.), causal forests, targeted maximum likelihood
- **v1.4** — Network econometrics: peer effects, network formation models
- **v1.5** — Structural discrete choice: BLP demand estimation, dynamic discrete choice (Rust-style)
- **v1.6** — Bayesian econometrics: Bayesian VAR, hierarchical models, MCMC-backed inference as an alternative `inference="bayesian"` path on existing estimators rather than a parallel API
- **v1.7** — High-dimensional methods: LASSO/post-double-selection for inference with many controls
- **v1.8** — Complex survey design: weighting, stratification, replicate-weight variance estimation
- **v1.9** — Text-as-data: dictionary methods, embeddings-based regressors, econometrically-valid ways to use LLM-derived features as regressors (with the measurement-error caveats made explicit, not hidden)

Design constraint carried through all of v1.x: **every new estimator ships
with a parity test against an existing reference implementation before
merge.** Breadth is only allowed to grow as fast as verification can keep up
— this is the lesson from how v0.1 shipped fast but still needed independent
correctness checks before anyone should have trusted it.

#### v2.0 — The Plugin Architecture: "Papers as Packages"
This is the structural turn from *a library of estimators the core team
wrote* to *a platform other researchers can publish onto*.
- [ ] `open_econs.register_estimator()` — a stable plugin API so a method
      doesn't need to live in the core repo to use the `BaseModel` contract
      (`.tidy()`, `.summary()`, `.export()`, frozen results, named pandas
      objects)
- [ ] A **methodology registry**: each estimator's docstring links to its
      source paper(s), with a machine-readable citation block
      (`estimator.citation` → BibTeX), so `result.summary()` can print
      "Method: Callaway & Sant'Anna (2021)" automatically
- [ ] `open-econs-contrib` namespace on PyPI, mirroring scikit-learn-contrib —
      new econometric methods from working papers can ship as installable
      packages that speak the same `BaseModel` interface, without waiting on
      core maintainer bandwidth
- [ ] A **new-method intake template**: a paper's authors (or anyone
      implementing their method) fill in a checklist — parity test against
      the paper's published replication numbers, edge-case tests, a
      docstring example — before an estimator is eligible for the registry
- [ ] Versioned methodology: when a paper is revised (e.g., a corrigendum
      changes a finite-sample correction), the registry tracks which package
      version implements which version of the method

#### v3.0 — Reproducibility Infrastructure
The project stops being just a package and starts being connective tissue
for how empirical results get checked.
- [ ] `ctx.export_replication()` — one call producing a full replication
      package: data snapshot hash, exact `open-econs` + dependency versions,
      formula strings, and regenerable output tables
- [ ] Integration hooks for journal data-and-code archives (AEA, REStat-style
      replication requirements) — export in the format editors actually ask for
- [ ] An open **replication registry**: published papers that used
      open-econs can register their replication package; anyone can re-run
      it and get a pass/fail badge, versioned against the paper's original
      open-econs release
- [ ] Automated "does this still reproduce" CI — re-run registered
      replications against new open-econs releases and flag silent breakage
      before it reaches users, not after

#### v4.0 — Education Layer
The API was always meant to be legible to someone learning econometrics
alongside Python, not just a tool for people who already know both.
- [ ] Interactive notebook companion following Angrist & Pischke's *Mostly
      Harmless Econometrics* and *Mastering 'Metrics*, reproducing textbook
      examples estimator-by-estimator
- [ ] A "show the math" mode: `result.derivation()` prints the classical
      closed-form computation in plain numpy alongside the production
      robust/clustered result, for students who want to see what's under
      the hood without reading the statsmodels source themselves
- [ ] Course adoption kit: problem sets, syllabus-ready modules, a
      known-answer autograder built on the parity-test infrastructure from
      v0.1–v1.0
- [ ] Localized documentation, starting with the languages of the largest
      non-English-speaking user communities that adopt the project

#### v5.0 — Community Foundation
If the project is still growing at this point, it has outgrown being one
person's repository.
- [ ] Formal governance: a maintainer council, RFC process for estimator API
      changes, and a documented decision-making process (openly modeled on
      how NumFOCUS projects run, not invented from scratch)
- [ ] Fiscal sponsorship or foundation status, so infrastructure (CI compute
      for the parity-test suite, docs hosting) isn't dependent on one
      person's goodwill
- [ ] A named annual or biennial contributor gathering / sprint, the way
      scikit-learn and pandas run sprints — where new estimator proposals
      get implemented and reviewed in person
- [ ] open-econs becomes a citable methodological standard: papers write
      "estimated using open-econs v5.x, method registry entry X" the way
      papers today cite "Stata 18" or "statsmodels" — because the parity
      and replication infrastructure built in v3.0 made that citation
      actually mean something verifiable

---

*The committed roadmap above is what's being built. The North Star is
deliberately larger than any one contributor can promise — that's the point.
If a piece of it excites you, [open an issue](https://github.com/qmanhbeo/open-econs/issues)
and help decide when it moves from "vision" to "roadmap."*
