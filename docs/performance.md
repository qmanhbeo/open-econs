# open-econs Performance — Vectorization & Parallelization

**open-econs uses Python's strengths (vectorization, parallelization) without
ever loosening numerical parity.** Refactors of existing float math are held to
**bit-identical** reproduction of the reference implementation; new methods are
validated to the standard ≤1e-6 tolerance. This page is the long-form companion
to the README's performance summary.

## Discipline

- Pure refactors of existing float math must be **bit-identical** (atol=0,
  rtol=0) to the prior implementation. Each is guarded by a determinism test
  that asserts exact equality against an independent scalar re-implementation.
- New methods are validated to ≤1e-6 against Stata / R.
- GPU offload (CuPy / CUDA) was evaluated and **deliberately declined**: the hot
  spots are SciPy optimizers (no GPU backend) and BLAS matmuls (already
  CPU-multithreaded), and transfer overhead dominates at current fixture sizes.
  Revisit only at 100k+ entities. Rationale:
  [methodology/performance-conventions.md](../methodology/performance-conventions.md).

## v1.0.3 hardening (bit-identical)

- **Propensity-score matching (`psm`)** — k-NN matching, the variance
  accumulation loops, and the `c_tau` influence-function term were fully
  vectorized into batched `scipy` / `numpy` reductions. **Bit-identical** to the
  prior scalar code and **~4× faster** on the Stata `teffects psmatch` fixture
  (nn=10: 0.50s → 0.13s).
- **GMM HAC weighting (`_hac_S`)** — the Newey-West lag accumulation was
  vectorized into a single batched `einsum` reduction. **Bit-identical** to the
  scalar loop; it feeds the `abond` / `gmm` variance and Hansen J-statistic.
- **Callaway-Sant'Anna DiD bootstrap (`did_cs`)** — permutation / bootstrap
  repetitions run through an opt-in `parallel=` process pool. Bit-identical to
  the serial path; no parity risk.

## See also

- [README](../README.md)
- [Release notes v1.0.3](release_notes_v1.0.3.md)
- [FUTURE_WORK.md](../FUTURE_WORK.md) — queued / deferred performance work.
