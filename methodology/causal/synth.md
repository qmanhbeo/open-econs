# Synthetic Control (`synth` / `placebo_space` / `placebo_time`)

## Usage

`oe.synth()` fits the Abadie-Diamond-Hainmueller synthetic control point
estimator: it returns the donor weights `W`, predictor weights `V`, the
pre-/post-treatment MSPE, and the treated/synthetic/gap path. The core estimator
is checked against R `Synth` (primary reference) and Stata `synth` (secondary).

```python
import open_econs as oe

# `data` must be a balanced long-format panel with an outcome, an entity id,
# and a time id column. `pre_period` is the last pre-treatment period
# (inclusive); `post_period` is the first post-treatment period (inclusive).
result = oe.synth(
    data=df,
    outcome="gdp",
    treated_unit="A",
    donor_pool=["B", "C", "D", "E"],
    pre_period=2000,
    post_period=2001,
)

# Donor weights W (sum to 1) and predictor weights V:
print(result.weights)
print(result.predictor_weights)

# Treated / synthetic / gap path over the analysis window:
print(result.gap_path)

# Fit quality diagnostics:
print(result.pre_mspe, result.post_mspe)
```

Optional arguments (see `open_econs/models/causal/synth.py`):
- `entity` / `time` (default `"entity"` / `"time"`): column names for the unit
  and time identifiers.
- `predictors`: explicit covariate columns aggregated by their pre-treatment mean
  (when omitted, the outcome's own pre-treatment path is used as predictors).
- `predictor_weights`: fixed `V` (mirrors R `Synth`'s `custom.v`), which skips the
  outer `V` optimization.

## Performance / parallelism (2026-07-17)

### Where the time goes
`oe.abond`? No — this is about `synth` permutation inference. `placebo_space` and
`placebo_time` each call `synth()` **once per donor unit** (space) or **once per
candidate pre-treatment date** (time). A single `synth()` fit is ~1.4 s on this
machine (N=12, 2 predictors) because the nested V-optimization (`_optimize_v` →
`_fn_v` → `_solve_w`) runs SLSQP with **numeric gradients** (`scipy`
`approx_derivative`), ~1000+ objective evals per fit. The permutation loop is
therefore O(#donors) or O(#candidate-dates) × ~1.4 s — that is the entire cost.

### Parallelization decision
- **Embarrassingly parallel**: every `synth()` fit is a *pure function* of its
  args — no shared mutable state. The only RNG use is a fixed-seed
  (`default_rng(17)`) Dirichlet start inside `_solve_w`, so each run is
  deterministic and reproducible regardless of process.
- **Process pool, NOT thread pool**: a `ThreadPoolExecutor` gives ~0.98x
  (slower) because the SLSQP/QM path holds the GIL (BLAS here is single-threaded,
  no GIL release). A `ProcessPoolExecutor` gives a real speedup.
- **Measured speedups (this machine, Windows, conda env):** N=6 → 0.90x
  (slower — spawn/pickle overhead dominates); N=12 → 1.61–1.68x. So parallelize
  **only when `#items >= _MIN_PARALLEL_ITEMS (=8)`**; below that the loop is
  sequential by default.
- **Bit-identical**: sequential vs parallel results match to `atol=0, rtol=0`
  (ratios, p-value, gap paths). Verified by `test_placebo_space_parallel_matches_sequential`
  and `test_placebo_time_parallel_matches_sequential`.

### What was NOT done (flagged footguns, rule 18)
- **GPU / numpy vectorization**: NOT applicable. Each fit is a sequential nested
  SLSQP + QP; the per-donor fits cannot be batched into one tensor op. The inner
  `_solve_w` QP could be batched across donors in theory, but that is a solver
  rewrite with parity risk (see below) — out of scope.
- **Analytic gradient for `_fn_v`**: the dominant cost is `scipy`'s numeric
  gradient (`approx_derivative`) inside SLSQP. Supplying an analytic gradient of
  the pre-treatment MSPE w.r.t. `V` would cut evals dramatically. **Not done**:
  it changes the SLSQP convergence path and risks diverging from R `Synth`'s
  `optimx` local optimum for `V` (the nonconvex-V multi-modality is already
  documented in `test_synth_placebo.py`). Would need a full R re-parity pass
  before enabling. Flagged, not silently shipped.
- **`parallel` is opt-in** (`parallel=False` default) to preserve current
  behaviour exactly and avoid spawn overhead on the small panels used by the
  fast-gate tests.

## Parity status
- Primary reference: R `Synth` (v1.1-10). Secondary: Stata `synth`.
- `W`, pre-MSPE, and gap path agree closely with R; `V` may differ slightly due
  to nonconvex V-landscape (accepted, reported honestly — see `synth.py` docstring
  and the multi-modal-donor exclusion in `test_synth_placebo.py`).
- `placebo_space`/`placebo_time` exercise the full validated `synth` solver per
  permutation unit; no estimator logic duplicated.
