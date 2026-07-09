# open-econs

**The scikit-learn of empirical economics.**

A Python library that bridges the gap between traditional Stata/R econometrics
workflows and modern, production-grade Python systems.  Every estimator follows
the same interface — `fit`, `summary`, `tidy`, `export` — so researchers and
AI agents never have to learn a new API.

## Quick Start

```python
import open_econs as oe

# Ordinary least squares – coefficients are named (pd.Series)
result = oe.ols("income ~ education + age", data=df, cluster="province")
print(result)
print(result.coefficients["education"])   # named access, not positions
result.tidy()                              # coefficient table (DataFrame)
result.predict(newdata=df_test)            # out-of-sample predictions

# reg() is an alias for ols()
result2 = oe.reg("income ~ education + age", data=df)

# Oaxaca-Blinder decomposition
decomp = oe.oaxaca(
    "income ~ education + age + female",
    data=df, by="female",
)
decomp.tidy()

# Context-based (dataset remembered)
ctx = oe.Context(df)
r1 = ctx.ols("income ~ education + age")
r2 = ctx.oaxaca("income ~ education + age + female", by="female")
```

## Installation

```bash
pip install git+https://github.com/qmanhbeo/open-econs.git
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

## v0.1 Estimators

| Function | Description |
|---|---|
| `ols()` / `reg()` | Ordinary least squares with HC1/robust/clustered SEs |
| `oaxaca()` | Oaxaca-Blinder decomposition (two-fold, three-fold) |

### Planned (future releases)

- `fe()` — fixed effects
- `iv()` — instrumental variables
- `logit()` / `probit()` — binary choice
- `did()` — difference-in-differences
- `psm()` — propensity score matching

## Result API

Every estimator returns an object with:

| Method | Returns |
|---|---|
| `.summary()` | Printable string (also `__repr__`) |
| `.tidy()` | `pd.DataFrame` — coefficient or effect table |
| `.predict(newdata)` | `pd.Series` — only on regression models |
| `.export(path)` | JSON serialization (`.json` only in v0.1) |
| `.plot()` | *Not yet implemented — raises with clear message* |
| `.to_dict()` | `dict` — full result metadata |

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

#### v0.2 — Causal Inference Core
- [ ] `fe()` — one-way and two-way fixed effects (within transformation + absorbed dummies)
- [ ] `iv()` — instrumental variables / 2SLS, with first-stage F-stat and weak-instrument warnings
- [ ] `logit()` / `probit()` — binary choice, with marginal effects (`.margins()`) not just raw coefficients
- [ ] `.plot()` implemented (matplotlib as an optional extra — `pip install open-econs[plot]`)
- [ ] Variable-level contribution breakdown for Oaxaca (currently aggregate-only)

#### v0.3 — Design-Based Causal Inference
- [ ] `did()` — two-period and staggered difference-in-differences
- [ ] Callaway–Sant'Anna and Sun–Abraham estimators for staggered treatment timing (the "bad comparisons" problem in TWFE)
- [ ] Event-study specification with pre-trend coefficient plots
- [ ] `ctx.vif()` — variance inflation factor / collinearity diagnostics
- [ ] `ctx.balance()` — covariate balance tables for treatment/control splits

#### v0.4 — Panel Data Engine
- [ ] First-class `PanelContext(df, entity=, time=)` — panel structure remembered, not re-specified per call
- [ ] Random effects, Hausman test (`fe()` vs `re()` comparison helper)
- [ ] Dynamic panel: Arellano-Bond / Blundell-Bond GMM estimator
- [ ] Driscoll-Kraay standard errors for cross-sectional dependence

#### v0.5 — Regression Discontinuity
- [ ] `rdd()` — sharp and fuzzy RDD, local linear/polynomial estimation
- [ ] Bandwidth selection (Imbens-Kalyanaraman, Calonico-Cattaneo-Titiunik)
- [ ] McCrary density test for manipulation at the cutoff
- [ ] Built-in RD plot (binned scatter + fitted lines either side of cutoff)

#### v0.6 — Matching & Balance
- [ ] `psm()` — propensity score matching (nearest-neighbor, caliper, kernel)
- [ ] Coarsened exact matching
- [ ] Post-matching balance diagnostics reusing `ctx.balance()` from v0.3
- [ ] Sensitivity analysis (Rosenbaum bounds)

#### v0.7 — Comparative Case Studies
- [ ] `synth()` — synthetic control (Abadie-Diamond-Hainmueller)
- [ ] Placebo-in-space and placebo-in-time inference
- [ ] Generalized synthetic control / interactive fixed effects (Xu 2017)

#### v0.8 — Structural Foundations
- [ ] `gmm()` — general GMM estimation framework other estimators can build on
- [ ] Nonlinear least squares
- [ ] Discrete choice groundwork (multinomial logit, nested logit)

#### v0.9 — Time Series & Release Candidate
- [ ] `ar()` / `var()` — autoregressive and vector autoregression
- [ ] Cointegration tests (Engle-Granger, Johansen)
- [ ] Newey-West HAC standard errors as a `cov_type` option across estimators
- [ ] API freeze candidate — no more breaking signature changes without a deprecation cycle
- [ ] Full docstring coverage + type-checked public API

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
