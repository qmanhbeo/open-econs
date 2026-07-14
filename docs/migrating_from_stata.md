# Migrating from Stata to open-econs

`open-econs` aims to be the "scikit-learn of empirical economics": one uniform,
immutable result object (`tidy()` / `summary()` / `export()` / `vcov()`) for
every estimator, so the mental model you build once transfers across methods.

## Side-by-side cheat sheet

| Stata | open-econs |
| --- | --- |
| `reg y x1 x2` | `oe.ols("y ~ x1 + x2", data=df)` |
| `reg y x, robust` | `oe.ols("y ~ x", data=df, cov_type="HC2")` |
| `reg y x, cluster(firm)` | `oe.ols("y ~ x", data=df, cluster="firm")` |
| `reg y x, cluster(firm year)` | `oe.ols("y ~ x", data=df, cluster=["firm", "year"])` |
| `newey y x, lag(2)` | `oe.ols("y ~ x", data=df, cov_type="HAC", lags=2, time="year")` |
| `xtset firm year` + `xtreg y x, fe` | `oe.PanelContext(df, "firm", "year").fe("y ~ x")` |
| `xtreg y x, re` | `.re("y ~ x")` |
| `xtreg y x, be` | `.fe("y ~ x")` after mean-centering, or pooled OLS |
| `ivregress 2sls y (x = z)` | `oe.iv("y ~ x | z", data=df)` |
| `xtabond / xtdpdsys` (dynamic panel) | `oe.abond("y ~ x", data=df, entity="firm", time="year")` |
| `logit y x` / `probit y x` | `oe.logit("y ~ x", data=df)` / `oe.probit(...)` |
| `did` (two-period) | `oe.did("y ~ treat*post", data=df, treatment="treat", post="post")` |
| `did_multiplegt` / `csdid` (staggered) | `oe.did_cs(df, y="y", entity="firm", time="year", treatment="treat")` |
| `rdrobust y x, c(0)` | `oe.rdd(df, y="y", running="x", cutoff=0.0)` |
| `nl (y = {a}*exp(-{b}*x)+{c}), initial(a 1 b 1 c 0)` | `oe.nls("y ~ a*exp(-b*x)+c", data=df, start_values={"a": 1.0, "b": 1.0, "c": 0.0})` |
| `esttab` (LaTeX/HTML) | `result.to_latex()` / `result.to_html()` |

## Differences to know

- **Formula syntax** uses Python `formulaic` (R-style): `y ~ x1 + x2`.
- **Inference is entity-clustered by default for panel-pooled OLS** (Stata's
  `xtreg, vce(cluster)` behaviour), matching modern applied practice.
- **Dynamic panels** use the Arellano-Bond difference GMM estimator with
  Windmeijer (2005) two-step standard errors, the Hansen J overidentification
  test, and Arellano-Bond AR(1)/AR(2) serial-correlation tests — the same
  diagnostics `xtabond2` reports.
- **Results are immutable** and return named `pandas` Series/DataFrames, so
  they are easy to consume programmatically and from AI agents.
- **Export** any result with `result.export("out.json")` or `result.export("out.csv")`.

## v1.0 status notes

- **Synthetic control (`synth`) shipped in v0.9** (Abadie-Diamond-Hainmueller
  core point estimator + ADH placebo-in-space / placebo-in-time inference).
  It is no longer "planned"; see `docs/api_stability.md` for limitations
  (`synth.predict()` / `synth.plot()` remain `NotImplementedError` by design).
- **`did_cs` HAC is experimental.** `did_cs(..., cov_type="HAC")`
  is a *project convention* (Newey-West temporal correction on the aggregated
  influence function), **not** externally validated. For publication prefer the
  default `cov_type="cluster"`. A `UserWarning` is raised on HAC use.
- **`nls` requires the optional `[nls]` extra** (`pip install "open-econs[nls]"`
  → pulls `sympy`). It is not a hard dependency.
- **`nlogit` (nested logit) is deferred** — documented in `docs/nlogit-recon.md`.
- **`reghdfe`-style multi-way *absorbed* fixed effects** are not yet implemented;
  `fe()` absorbs entity and time via the within transform.
- **Bootstrap inference for staggered DiD** is not implemented; use
  cluster-robust SEs (or the experimental HAC convention).
