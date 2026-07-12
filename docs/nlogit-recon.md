# `nlogit()` — Nested Logit Recon Report

**Status:** Investigated, deliberately deferred (not built)  
**Branch:** `feature/nlogit` (based on `db2dfe5` off `main`)  
**Date:** 2026-07-12  
**Linked roadmap entry:** `ROADMAP.md` line 289 (nlogit, deferred)

---

## Why This Was Investigated

Nested logit is listed in the v0.9 roadmap under "Discrete choice groundwork." The three-question test (§F) passes: it is self-contained, no existing Python implementation covers it, and it requires genuine domain expertise. However, the risk profile warrants deferral.

## What Was Confirmed

### A. No Existing Python Implementation

- **statsmodels:** Only `Logit`, `LogitResults`, `MNLogit`. No nested logit anywhere.
- **pip ecosystem:** `pip list` and PyPI search both empty for nested/nlogit.
- **Dependency tree:** No nested logit in any transitive dependency.
- **Conclusion:** Implementation would be from-scratch MLE. There is no wrapper path like `mlogit()` uses with `sm.MNLogit`.

### B. Stata `nlogit` — Verified Reference

**Syntax** (from `nlogit.sthlp`, 583 lines):
```
nlogit depvar [indepvars] ifin [weight] || lev1_equation || lev2_equation ... || altvar:, case(varname) [options]
```

Tree declaration is a separate step:
```
nlogitgen type = restaurant(fast: Freebirds | MamasPizza, family: CafeEccell | ...)
nlogittree restaurant type, choice(chosen) case(family_id)
```

**Live output** (batch `/b do` on `webuse restaurant`):
- Log likelihood = −485.47331, 10 params, 300 cases
- τ values: fast=1.712878, family=2.505113, fancy=4.099844 (all >1)
- IIA LR test: chi2(3)=6.87, p=0.0762

**`e()` returns:** `e(b)` is 1×10, `e(V)` is 10×10, `e(ll)`, `e(chi2_c)`, `e(p_c)`, `e(rum)`.

**Source architecture:**
- `nlogit.ado` (761 lines): Orchestrates estimation using `ml model d1` (user-supplied analytic gradient, BFGS optimizer).
- `_nlogitmodel.class` (Mata, ~800 lines readable): Contains `.evaluate` (likelihood) and `.scores` (gradient). Likelihood computed recursively bottom-up through the tree.
- `nlogit_lf.ado` (18 lines): Trivial wrapper delegating to Mata class.
- `nlogitgen.ado` (110 lines): Creates nesting variable from `label: alt1 | alt2` syntax.

### C. Likelihood Formula (McFadden 1978)

The nested logit model decomposes choice probability as:

```
P(j | m) = P(j | m) × P(m)
```

Bottom level: `P(j | m) = exp(X_j'β / τ_m) / Σ_{k∈m} exp(X_k'β / τ_m)`  
Upper level: `P(m) = exp(η_m + τ_m × I_m) / Σ_n exp(η_n + τ_n × I_n)`  
Inclusive value: `I_m = ln Σ_{k∈m} exp(X_k'β / τ_m)`

**RUM vs nonnormalized:**
- RUM-consistent (Stata default): τ_m is the dissimilarity parameter. τ ∈ (0,1) required for utility maximization.
- Nonnormalized (`nonnormalized` option): uses 1/τ_m scaling.
- R `mlogit`: uses inclusive-value parameter directly. `unscaled=TRUE` gives nonnormalized.

### D. SE Convention

- **Default:** OIM (Observed Information Matrix). Available: `robust`, `cluster`, `bootstrap`, `jackknife`.
- **Forbidden:** `vce(opg)`, `technique(bhhh)`.
- **IIA test:** LR test, only under OIM VCE. Suppressed with robust/cluster.
- **Implementation path:** `scipy.optimize.minimize` (L-BFGS-B or BFGS) + analytic gradient (recursive tree traversal) + numerical Hessian (OIM) or sandwich VCE.

### E. Nesting-Tree API Design

Three candidate shapes evaluated:

| Option | Form | Pros | Cons |
|--------|------|------|------|
| 1. Dict-based | `nesting={"type": {"fast": [...], ...}}` | Self-documenting, inline | First dict arg in API; multi-level unwieldy |
| 2. Column-based | `nesting="type"` | Scales to any depth; Stata-like | Requires pre-created column; less self-documenting |
| 3. Hybrid | Both supported | Best of both worlds | Two code paths; type ambiguity |

**Recommended:** Option 1 (dict-based). Cleanest for first implementation. Column-based can be added later as alias.

**Convention deviation:** First estimator to take a structured dict arg (existing pattern is formula + scalar kwargs). Justified by inherent structure of nesting trees.

### F. Three-Question Test

| Question | Answer |
|----------|--------|
| Self-contained? | Yes — no existing dependency provides nested logit |
| Something better already exists? | No — genuine gap in Python ecosystem |
| Requires real domain expertise? | Yes — identification, τ boundary conditions, degenerate nests, gradient correctness |

---

## Why Deferred

### 1. R `mlogit` Cannot Run Full Stata-Equivalent Specification

**Re-verified on R 4.6.1 + mlogit 2.0.0 (binary, matched versions).** The full specification (with nest-level covariates `income + kids`) fails with:
```
system is computationally singular: reciprocal condition number = 9.73028e-19
```

**Root cause identified (not a guess):** Income and kids are case-level variables with **identical means across all three nests** (income: 40.45 in all; kids: 2.69 in all). Since each case belongs to exactly one nest, these variables are perfectly collinear across nests. Stata handles this by pinning `base(family)` to zero and estimating relative effects; R's mlogit parameterization causes singularity.

The simple specification (3 params: cost, distance, rating) works fine with R mlogit.

**Implication:** R `mlogit` is a parity reference for simple nested logit only, not for the full Stata `nlogit` feature set. Stata alone serves as the primary parity reference for nest-level covariate specifications.

### 2. τ > 1 Is a Known Property of This Dataset

The `webuse restaurant` dataset produces τ values > 1 (1.71, 2.51, 4.10), violating RUM consistency. **This is documented and acknowledged by Stata itself.** The manual states:

> "Dissimilarity parameters greater than one imply that the model is inconsistent with RUM; Hensher, Rose, and Greene (2005, sec. 13.6) discuss this in detail. We will ignore the fact that all our dissimilarity parameters exceed one."

This means the canonical teaching dataset is unsuitable as a τ∈(0,1) parity fixture. A validated fixture dataset needs to be sourced or constructed.

### 3. Analytic Gradient Is the Critical Path

Stata's gradient code is ~200 lines of recursive Mata (in `._dldtau` and `._dldx` programs within `_nlogitmodel.class`). This computes scores for both the β parameters and the τ parameters through the tree hierarchy. An incorrect gradient silently produces wrong SEs. Implementation requires either:
- Line-by-line Mata→Python translation (fragile, hard to test)
- Independent derivation from the likelihood formula (requires deep domain expertise)

### Effort Estimate

| Scope | Estimate |
|-------|----------|
| Conservative (full-featured) | 4–6 weeks |
| Minimum viable (two-level, basic) | 2–3 weeks |
| Comparison | `mlogit()`: ~230 lines (wrapper); `gmm()`: ~260 lines (custom MLE); `nlogit()`: ~500–800 lines + ~300 lines parity tests |

### Prerequisites for Un-Deferral

1. Upgrade R to match mlogit build version, or rebuild mlogit from source (done — R 4.6.1 + mlogit 2.0.0 confirmed working for simple spec)
2. Source/construct a validated fixture dataset with τ ∈ (0, 1) and known ground truth
3. Implement likelihood + gradient on paper/notebook before production code
4. Scope first pass to two-level nesting only (one nest variable, no deeper hierarchy)
