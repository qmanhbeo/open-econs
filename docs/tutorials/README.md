# Tutorials

Worked, runnable walkthroughs for open-econs estimators. Each mirrors the same
structure: generate a small dataset → estimator call → `tidy()` / `summary()` /
key attributes → output interpretation → honest parity note.

| Tutorial | Estimator(s) | Maps to |
|----------|--------------|---------|
| [OLS](ols.md) | `oe.ols` | Stata `reg` / R `lm` |
| [Fixed Effects](fe.md) | `oe.fe` | Stata `xtreg, fe` / R `fixest::feols` |
| [IV](iv.md) | `oe.iv` | Stata `ivregress` / R `AER::ivreg` |
| [DiD](did.md) | `oe.did`, `oe.event_study`, `oe.staggered_did` | Stata `did` / R `fixest` / `csdid` |
| [RDD](rdd.md) | `oe.rdd`, `oe.density_test` | R `rdrobust` / Stata `rddensity` |
| [PSM](psm.md) | `oe.psm`, `oe.cem`, `oe.balance`, `oe.rosenbaum_bounds` | R `MatchIt` / `Matching` / Stata `teffects psmatch` / `rbounds` |
| [Synthetic Control](synth_control.md) | `oe.synth`, `placebo_space`, `placebo_time` | R `Synth` / Stata `synth` |

**Validation bar.** These tutorials are smoke-tested locally for runnability
and sensible numbers. They are *not* a claim of full Stata/R numerical parity;
parity is established by the project's gated fixture tests (see each tutorial's
parity note for the relevant test file). Each tutorial states its known
limitations plainly.
