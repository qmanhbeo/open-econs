# open-econs

[![PyPI version](https://img.shields.io/pypi/v/open-econs?color=blue)](https://pypi.org/project/open-econs/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-econs)](https://pypi.org/project/open-econs/)

**The scikit-learn of open economics.**

A Python library that bridges the gap between traditional Stata/R econometrics
workflows and modern, production-grade Python systems.  Every estimator follows
the same interface — `summary`, `tidy`, `export` — so researchers and
AI agents never have to learn a new API.

> **Current version (v0.6.4):** Stata-parity test suite — 27 hand-written
> `.do` files with committed `.dta` fixtures, dual-mode execution (live Stata
> or CI fallback), FE df correction, within R² fix, plus all v0.6.3 features.

## Quick Start

```python
import open_econs as oe
import pandas as pd

df = pd.DataFrame({
    "income":    [30, 45, 55, 70, 85, 40, 60, 95],
    "education": [10, 12, 14, 16, 18, 11, 15, 20],
    "age":       [25, 30, 35, 40, 45, 28, 38, 50],
    "female":    [0,  0,  0,  0,  1,  1,  1,  1],
    "province":  ["A","A","B","B","C","C","A","B"],
})

# --- OLS with named coefficients and cluster-robust SEs ---
r = oe.ols("income ~ education + age", data=df, cluster="province")
r.coefficients          # pd.Series with name index
# Intercept   -32.82
# education     8.97
# age          -1.03

r.tidy()              # coefficient table as DataFrame
r.predict(df.head(2))# out-of-sample predictions
# 0    31.28
# 1    44.10

print(r.summary())    # printable summary (also __repr__)

# --- Logit / Probit ---
r_logit = oe.logit("female ~ education + age", data=df)
r_logit.tidy()        # coef, z, P>|z| table
r_logit.margins()     # average marginal effects
r_logit.predict(proba=False)  # binary class prediction

# --- Fixed effects ---
r_fe = oe.fe("income ~ education + age", data=df, entity="province")
r_fe.tidy()           # within-transformed coefficients

# --- IV / 2SLS ---
r_iv = oe.iv("income ~ education | age", data=df)
r_iv.tidy()           # 2SLS coefficients
r_iv.first_stage()    # first-stage F-stat

# --- VIF diagnostics ---
ctx = oe.Context(df)
ctx.vif("income ~ education + age")  # VIF per variable

# --- Oaxaca-Blinder decomposition ---
# NOTE: the 'by' column must also appear on the RHS of the formula
d = oe.oaxaca("income ~ education + age + female", data=df, by="female")
d.explained          # 16.00  (covariate-driven gap)
d.unexplained        #  4.00  (coefficient-driven gap)
d.total_gap          # 20.00  (female mean - male mean)

# --- Context remembers the dataset ---
ctx.ols("income ~ education + age")            # same as oe.ols(..., data=df)
ctx.logit("female ~ education + age")
ctx.probit("female ~ education + age")
ctx.vif("education + age")

# --- Immutability ---
r.f_statistic = 0.0  # AttributeError: OLSResult is immutable
```

## Installation

```bash
pip install open-econs                           # core: OLS, Oaxaca, FE, IV, Logit, Probit
pip install open-econs[plot]                      # + matplotlib for .plot()
pip install open-econs[dev,lint]                  # + development & linting tools
pip install git+https://github.com/qmanhbeo/open-econs.git    # latest dev
```

Requires Python ≥ 3.10.

## Design Principles

- **Every result is immutable** once `fit()` completes.
- **All numeric artifacts are named** (`pd.Series`/`pd.DataFrame` with
  variable-name indices).  No raw `numpy.ndarray` crosses the public API.
- **Every error tells you what to fix.** Missing column → names the column,
  lists what's available. Non-binary `by` → shows the values found.
- **Consistent interface across estimators**: `summary()`, `tidy()`,
  `export()`, `predict()` (where applicable).

## Estimators

| Function | Description |
|---|---|
| `ols()` / `reg()` | OLS with HC1/robust/clustered SEs, **multi-way clustering** (`cluster=["a","b"]`), **Newey-West HAC** (`cov_type="HAC"`), WLS |
| `fe()` | Fixed effects (one-way entity, two-way entity + time) |
| `iv()` | Instrumental variables / 2SLS with first-stage F-stat |
| `logit()` | Binary logit with `.margins()`, `.predict()` |
| `probit()` | Binary probit (same API as logit) |
| `oaxaca()` | Oaxaca-Blinder decomposition (two-fold, three-fold) |
| `ctx.vif()` | Variance inflation factor / collinearity diagnostics |
| `abond()` | Arellano-Bond dynamic panel (difference GMM), two-step Windmeijer SEs, Hansen J + AR(1)/AR(2) |
| `staggered_did()` | Callaway-Sant'Anna (2021) staggered / heterogeneous-timing DiD |
| `rdd()` | Sharp / fuzzy regression discontinuity (local linear, triangular kernel) |
| `did()` / `event_study()` / `balance()` | Two-period DiD, event-study, balance tables |
| `oe.PanelContext(...)` | `pooled/fitted/fe/re/diff/driscoll_kraay/hausman/abond` with remembered entity/time |

## Result API

Every estimator returns an object with:

| Method | Returns |
|---|---|---|
| `.summary()` | Printable string (also `__repr__`) |
| `.tidy()` | `pd.DataFrame` — coefficient or effect table |
| `.vcov()` | `pd.DataFrame` — variance-covariance matrix |
| `.predict(newdata)` | `pd.Series` — only on regression models |
| `.export(path)` | JSON / CSV serialization |
| `.plot()` | Residual diagnostics plot (requires `pip install open-econs[plot]`) |
| `.to_dict()` | `dict` — full result metadata |
| `.to_latex()` | LaTeX table string |
| `.to_html()` | HTML table string |

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

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

#### v0.6 — Panel Data Engine *(current)*
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
- [x] **Vectorized `_within_transform()`** — pandas groupby replaces Python
  loop for speed on large panels.
- [x] **Oaxaca `get_dummies` fix** — uses `formulaic` `ensure_full_rank`
  instead of fragile column-name pattern matching.

#### v0.7 — Regression Discontinuity refinements
- [x] Bandwidth selection (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik)
- [ ] McCrary density test for manipulation at the cutoff
- [ ] Built-in RD plot (binned scatter + fitted lines either side of cutoff)

#### v0.8 — Matching & Balance
- [ ] `psm()` — propensity score matching (nearest-neighbor, caliper, kernel)
- [ ] Coarsened exact matching
- [ ] Post-matching balance diagnostics reusing `ctx.balance()` from v0.5
- [ ] Sensitivity analysis (Rosenbaum bounds)

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
