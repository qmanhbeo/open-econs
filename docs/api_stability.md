# API Stability & Versioning Policy

**Status:** in force as of v1.0.0 (updated for v1.0.2). This document is the authoritative statement of
what "stable" means for open-econs and supersedes looser wording in earlier
roadmap entries.

## What is the public API

The public API is **exactly** the names exported from `open_econs` via
`__all__` in `open_econs/__init__.py`. As of v1.0.0 that set is:

```
ols, reg, logit, probit, mlogit, fe, iv, oaxaca,
did, event_study, balance, abond, did_cs, density_test, cem,
psm, rdd, rosenbaum_bounds, gmm, GMMResult,
nls, NLSResult, synth, SynthResult,
Context, PanelContext
```

Plus the package version string `open_econs.__version__`.

Everything else is **not** public and may change without notice:

- Anything prefixed with a leading underscore (`_internal`, `_gmm_core`,
  `_capture_call`, `_version`, private helpers, module-internal classes).
- Positional / keyword argument *ordering* of functions not in `__all__`.
- The exact text of `summary()` / error messages (treat them as human-readable,
  not machine-parseable).
- `tests/` and `docs/` (except this policy and the tutorials, which are
  documentation, not API).

## Semantic versioning commitment

open-econs follows [SemVer](https://semver.org):

- **MAJOR** (`x.0.0`): a breaking change to the public API.
- **MINOR** (`1.x.0`): new functionality in a backwards-compatible way.
- **PATCH** (`1.0.x`): backwards-compatible bug fixes.

A change is **breaking** if it alters the behavior or signature of anything in
`__all__` in a way that could break correct calling code, including:

- removing a public function/class or renaming it;
- removing or renaming a public keyword argument;
- changing the *meaning* of an existing argument (e.g. a different default that
  changes results);
- changing the type/structure of a returned result object's documented public
  attributes.

## Deprecation policy

No breaking change to the public API may be made **without a deprecation
cycle**:

1. The old behavior is kept and emits a `DeprecationWarning` (or, for removal of
   an alias, a `FutureWarning`) describing the replacement and the version in
   which it will be removed.
2. The deprecation is documented in the release notes and in this file's
   "Deprecations" table below.
3. The removal happens only in the next **MAJOR** version (or later).
4. Emergency security fixes are the only exception and are called out
   explicitly.

Stable result objects are frozen (`_freeze()`) and expose immutable, named
`pd.Series` / `pd.DataFrame` outputs. Adding *new* public attributes to a
result object is non-breaking; removing or renaming one is breaking.

## Optional dependencies

Several estimators depend on optional extras (`[nls]`, `[rd]`, `[plot]`).
Importing / calling such an estimator without the extra installed raises a clear
`ImportError` naming the extra to `pip install` — this is expected behavior, not
a breaking change. `sympy` (NLS) and `rdrobust`/`rddensity` (RDD) are optional;
the relevant estimators degrade to documented fallbacks where one exists.

## Deprecations

| Version | Item | Replacement | Removal planned |
|---------|------|-------------|-----------------|
| —       | —    | —           | —               |

*(No deprecations are active as of v1.0.0.)*

## Known limitations carried into v1.0.0

- `did_cs(..., cov_type="HAC")` is a **project convention**, not an
  externally validated variance estimator. Use `cov_type="cluster"` (default)
  for publication. See `open_econs.models.causal.did_cs`.
- `nlogit()` remains unimplemented (deferred; see roadmap North Star / recon
  doc).
- Tutorial coverage at v1.0.0 covers OLS, FE, IV, and DiD; RDD, PSM, and
  synthetic-control walkthroughs are planned post-1.0.
