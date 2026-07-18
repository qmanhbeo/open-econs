# Performance Conventions (Python-strength utilization)

Recorded 2026-07-17 during the Python-strength inspection (rule 16 / rule 19).
These facts are environment-verified and should stop future sessions from
re-deriving them or re-opening GPU.

## BLAS backend (numpy / scipy)

This environment ships **OpenBLAS 0.3.31, `DYNAMIC_ARCH`,
`MAX_THREADS=24`** (confirmed via `numpy.show_config()`). Consequences:

- **Large numpy matmuls already run multithreaded on CPU.** Any loop you
  *vectorize* into bigger array ops (batched `einsum`, `matmul`, `cKDTree.query`
  of many points at once) gets thread-level parallelism for free — no process
  pool required.
- **`ThreadPoolExecutor` is useless for numpy/BLAS-bound work.** BLAS calls hold
  the GIL, so a thread pool does not overlap with BLAS compute. Use it only for
  genuinely I/O-bound work.
- **`ProcessPoolExecutor` helps only genuinely Python/GIL-bound loops** — e.g.
  repeated calls into `scipy.optimize` (SLSQP/Nelder-Mead, which release the GIL
  but are pure-Python-heavy), or per-rep resampling that rebuilds DataFrames.

## Opt-in parallelism convention (reuse, do not reinvent)

`placebo.py` established the pattern; `did_cs.py` mirrors it exactly:

- kwarg `parallel: bool = False` (opt-in; default never regresses).
- gate threshold `_MIN_PARALLEL_ITEMS = 8` below which the loop runs
  sequentially (avoids process-spawn/pickling overhead on small loops).
- workers are **module-level pure functions** (picklable under Windows spawn).
- **bit-identical** to the sequential path: pre-generate any RNG sequence in
  the parent from the seeded `RandomState` and pass each draw to its worker, so
  the deterministic draw order is preserved (rule 2: atol=0, rtol=0).

## GPU — declined project-wide

See `FUTURE_WORK.md` (bottom, "GPU acceleration — DECLINED"). Hot spots are
scipy optimizers (no GPU backend) and BLAS matmuls (already CPU-multithreaded);
GPU transfer overhead dominates at current fixture sizes. Revisit only at
100k+ entity panels.

## Implemented so far

- `placebo.py` (`placebo_space` / `placebo_time`): ProcessPool over per-donor /
  per-date `synth()` fits (commit `51ac7eb`).
- `did_cs.py` (`did_cs` bootstrap): ProcessPool over pre-generated entity
  resamples (commit `3f56aea`). ~2x speedup at 400 entities / 200 reps,
  bit-identical for both `reg` and `dripw` methods.

## Completed vectorizations (bit-identical — do NOT re-touch)

- **Candidate C — `psm.py` vectorize (commit `cdb15be`).** Batched
  `cKDTree.query` in `_within_treatment_matching` / `_opposite_treatment_matching`;
  fancy-indexed `psi`; padded `(n,h)` / `(n,h,p)` tensor reduction for `xi2` and
  `c_tau` (via `_padded_local_cov`); vectorized `matched_arr` mask.
  **Bit-identical** (atol=0, rtol=0) — verified by `test_psm_*_bit_identical_to_scalar`
  and `test_psm_c_tau_vectorization_bit_identical`; the Stata-pinned
  `test_psm_se_nn*` (nn=2/5/10) still pass ≤1e-6. **~4× faster** on the Stata
  `teffects psmatch` fixture (nn=10: 0.50s → 0.13s). Do NOT re-touch unless a new
  neighborhood-size edge case breaks bit-identicality.
- **Candidate D — `_gmm_core._hac_S` vectorize (commit `897c31a`).** Per-entity
  batched `np.einsum("ti,tj->ij", moments[lag:], moments[:-lag])` replacing the
  inner per-lag/per-t `np.outer` accumulation (ragged per-entity loop preserved).
  **Bit-identical** (atol=0) — `TestHacSVectorization` in
  `tests/non_stata_nor_r/test_gmm_core.py`; Stata anchors `test_stata_gmm.py`
  (HAC two-step) + `test_stata_abond.py` still ≤1e-6. Full write-up in
  `methodology/linear/gmm.md` (Root-Cause Knowledge). Do NOT re-touch unless a
  new HAC edge case breaks bit-identicality.
