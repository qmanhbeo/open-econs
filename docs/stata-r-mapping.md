# open-econs ↔ Stata / R Command Mapping

**If you know the Stata or R command, you already know the open-econs
equivalent.** This is the full mapping companion to the README's short list.
Every open-econs estimator is validated against the corresponding Stata / R
implementation to ≤1e-6 (machine precision for IV / Arellano-Bond / synthetic
control).

## Stata → open-econs

| Stata                | open-econs                  |
| -------------------- | --------------------------- |
| `regress`            | `oe.ols()`                  |
| `xtreg, fe`          | `oe.fe()`                   |
| `ivregress 2sls`     | `oe.iv()`                   |
| `logit` / `probit`   | `oe.logit()` / `oe.probit()`|
| `mlogit`             | `oe.mlogit()` (multinomial logit) |
| `oaxaca`             | `oe.oaxaca()`               |
| `xtabond2`           | `oe.abond()`                |
| `gmm`                | `oe.gmm()`                  |
| `csdid`              | `oe.did_cs()`               |
| `eventstudyinteract`| `oe.did_sa()` / `oe.event_study()` |
| `did2s`              | `oe.did_gardner()`          |
| `rdrobust`           | `oe.rdd()`                  |
| `rddensity`          | `oe.density_test()`         |
| `teffects psmatch`   | `oe.psm()`                  |
| `teffects psmatch` (cem) | `oe.cem()`               |
| `synth_runner`       | `oe.synth()` / `oe.placebo_space()` / `oe.placebo_time()` |
| `dfuller` / `pperron` / `vars` / `var` | `oe.adf()` / `oe.pp()` / `oe.var()` / `oe.arima()` |
| `arch`               | `oe.garch()`                |
| `ardl`               | `oe.ardl_fit()` / `oe.uecm_fit()` |
| `ardl` (bounds) / `ardlbounds` | `oe.uecm_fit(...).bounds_test()` |

## R → open-econs

| R                    | open-econs                  |
| -------------------- | --------------------------- |
| `fixest` / `plm`     | `oe.fe()` / `oe.PanelContext()` |
| `AER::ivreg`         | `oe.iv()`                   |
| `did`                | `oe.did_cs()`               |
| `fixest::sunab`      | `oe.did_sa()`               |
| `fixest::did2s`      | `oe.did_gardner()`          |
| `synth`              | `oe.synth()`                |
| `MatchIt`            | `oe.psm()`                  |
| `cobalt`             | `oe.balance()`             |
| `rbounds`            | `oe.rosenbaum_bounds()`    |
| `urca` / `vars`      | `oe.adf()` / `oe.var()`    |
| `ARDL` / `dynamac`   | `oe.ardl_fit()` / `oe.uecm_fit()` / `.bounds_test()` |

## Full method catalog

See [README](../README.md#supported-methods) for the complete
`Supported Methods` table (every estimator, with status). Step-by-step porting
walkthroughs live in [Tutorials](tutorials/README.md).

## See also

- [Migrating from Stata](migrating_from_stata.md) / [Migrating from R](migrating_from_r.md)
- [Methodology](../methodology/) — per-estimator specification and Stata/R equivalents.
