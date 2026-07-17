# open-econs v1.0.3 — Release Notes

**Status: PREPARED.** Version bumped to `1.0.3` in `pyproject.toml`; README,
ROADMAP, and this document updated. Tagging `v1.0.3`, the GitHub Release, and
the PyPI upload are downstream of maintainer sign-off.

## Highlights

- **Version `1.0.3`**, Development Status **5 — Production/Stable**.
- **Performance hardening, bit-identical to prior releases.** Three hot loops
  were vectorized / parallelized with **zero parity tolerance loosened** — each
  change is guarded by a new determinism test that asserts exact (`atol=0`)
  reproduction of the previous scalar implementation.
- **PyPI package description rewritten** for clarity and discoverability — it now
  reads *"Empirical economics and causal inference in Python — a scikit-learn-
  style unified API with Stata/R-grade numerical parity."* (the old
  "scikit-learn of empirical economics" line understated the project's scope).
- **README refreshed**: corrected parity-test count, expanded feature coverage
  (DID family, GMM / Arellano-Bond, panels, time-series), added a Performance
  section, removed two stale duplicate sections.

## Performance (honest, sourced)

All numbers below come from the committed work and its determinism tests — no
new benchmark scripts were invented for this release.

- **`psm()` — fully vectorized (~4× faster, bit-identical).** k-NN matching,
  1:1 nearest-neighbor with replacement, the `xi2` neighborhood-variance loop,
  and the `c_tau` influence-function term were rewritten as batched
  `scipy`/`numpy` reductions (padded `(n, h)` / `(n, h, p)` tensors, fancy-
  indexed `psi`). On the Stata `teffects psmatch` fixture, `psm` at nn=10 went
  from **0.50s → 0.13s (~4×)**. New bit-identical determinism tests
  (`test_psm_*_bit_identical_to_scalar`, `test_psm_c_tau_vectorization_bit_identical`)
  and the existing Stata-pinned `test_psm_se_nn*` (nn=2/5/10) all still pass.
  Commit `cdb15be`.

- **`_gmm_core._hac_S` — Newey-West lag accumulation vectorized (bit-identical).**
  The per-lag / per-t `np.outer` accumulation is now a single batched `einsum`
  reduction per entity. The reduction order along the time axis is unchanged, so
  the result is **bit-identical** to the scalar loop (guarded by
  `TestHacSVectorization` in `tests/non_stata_nor_r/test_gmm_core.py`). This
  matrix feeds the `abond` / `gmm` variance and Hansen J-statistic; the Stata
  abond / gmm HAC parity tests are unchanged at ≤1e-6. Commit `897c31a`.

- **`did_cs()` — opt-in parallel bootstrap (bit-identical).** Permutation /
  bootstrap repetitions run through an opt-in `parallel=` `ProcessPoolExecutor`.
  The parallel path is **bit-identical** to the serial path. Commit `3f56aea`
  (code) + `3eb0e92` (docs).

- **GPU offload (CuPy / CUDA) deliberately declined.** The hot spots are SciPy
  optimizers (no GPU backend) and BLAS matmuls (already CPU-multithreaded);
  transfer overhead dominates at current fixture sizes. Revisit only at 100k+
  entities. Rationale in `methodology/performance-conventions.md` and
  `FUTURE_WORK.md`.

## Parity discipline

- **550+ Stata- and R-parity tests** (330+ vs Stata, 220+ vs R) run in CI on
  every release; the full suite is **1000+ tests**. A numerical-equivalence
  regression fails the build before it ships.
- No tolerance was loosened for v1.0.3. Refactors are held to **bit-identical**
  reproduction; the public API is unchanged (see `docs/api_stability.md`).

## Scope

v1.0.3 is a **parity-hardening + performance + documentation** release. No new
estimators ship in it — the next method additions follow the roadmap
(`v1.1` time-series already in progress; `v1.2` limited-dependent-variable
models). The deferred `synth` analytic-gradient item (Candidate A) remains
deferred behind a full R `Synth` re-parity budget.

## Upgrade

```bash
pip install -U open-econs
```

No breaking changes. Deprecation shims from v1.0.2 (`staggered_did` → `did_cs`,
`did_sun_abraham` → `did_sa`) remain in place and will be removed in v2.0.0.
