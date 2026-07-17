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

## Queued (see FUTURE_WORK.md top)

- Candidate C — `psm.py` vectorize (batched `cKDTree.query` + fancy-indexed
  variance loops). Safe, no parity risk.
- Candidate D — `_gmm_core._hac_S` vectorize (per-entity batched `einsum`).
  Parity-sensitive (feeds abond/gmm VCE/J-stat to ≤1e-6); verify vs Stata
  fixtures.
