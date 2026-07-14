# Migrating from R to open-econs

`open-econs` gives you one uniform, immutable result object
(`tidy()` / `summary()` / `export()` / `vcov()`) across every estimator, so the
mental model you build once transfers across methods — the same goal as the
`fixest` / `estimatr` / `plm` ecosystem, but with a single consistent API.

## Side-by-side cheat sheet

| R | open-econs |
| --- | --- |
| `stats::lm(y ~ x1 + x2)` | `oe.ols("y ~ x1 + x2", data=df)` |
| `estimatr::lm_robust(y ~ x, se_type="HC2")` | `oe.ols("y ~ x", data=df, cov_type="HC2")` |
| `estimatr::lm_robust(y ~ x, clusters = firm)` | `oe.ols("y ~ x", data=df, cluster="firm")` |
| `sandwich::vcovHAC(lm, lag = 2)` / `NeweyWest` | `oe.ols("y ~ x", data=df, cov_type="HAC", lags=2, time="year")` |
| `plm(y ~ x, model="within", index=c("firm","year"))` | `oe.PanelContext(df, "firm", "year").fe("y ~ x")` |
| `plm(y ~ x, model="random")` | `.re("y ~ x")` |
| `plm(y ~ x, model="pooling")` | `.pooled("y ~ x")` |
| `fixest::feols(y ~ x \| firm + year)` | `.fe("y ~ x")` after encoding `firm`/`year` in the formula, or `.fe("y ~ x + C(firm) + C(year)")` |
| `AER::ivreg(y ~ x \| z)` | `oe.iv("y ~ x \| z", data=df)` |
| `fixest::did(y ~ treat * post)` | `oe.did("y ~ treat*post", data=df, treatment="treat", post="post")` |
| `fixest::eventstudydid(...)` | `oe.event_study("y ~ treat*period", data=df, treatment="treat", post="post")` |
| `did::att_gt` / `csdid` (staggered) | `oe.did_cs(df, y="y", entity="firm", time="year", treatment="treat")` |
| `rdrobust::rdrobust(y, x, c=0)` | `oe.rdd(df, y="y", running="x", cutoff=0.0)` |
| `rdl::density_test` (McCrary) | `oe.density_test(df, running="x", cutoff=0.0)` |
| `MatchIt::matchit(treat ~ x1 + x2)` | `oe.psm(df, treatment="treat", covariates=["x1","x2"])` |
| `Synth::synth(...)` | `oe.synth(df, outcome="y", treated_unit="A", entity="unit", time="year", pre_period=.., post_period=..)` |

## Differences to know

- **Formula syntax** uses Python `formulaic` (R-style): `y ~ x1 + x2`. Factor
  terms use `C(name)` (e.g. `C(firm)`); `fixest`-style `| firm + year` absorbs
  are expressed by adding the factor terms to the formula (open-econs `fe()`
  applies the within transform for the entity/time you declare on the
  `PanelContext`).
- **Results are immutable** and return named `pandas` Series/DataFrames, so they
  are easy to consume programmatically and from AI agents.
- **Inference is entity-clustered by default for panel-pooled OLS**, matching
  modern applied practice.
- **Export** any result with `result.export("out.json")` or `result.export("out.csv")`.

## v1.0 status notes

- **`did_cs` HAC is experimental.** `cov_type="HAC"` is a *project
  convention* (Newey-West on the aggregated influence function), **not**
  externally validated — there is no `fixest`/`Synth` reference for it. For
  publication prefer the default `cov_type="cluster"`; a `UserWarning` is raised
  on HAC use.
- **`nls` requires the optional `[nls]` extra** (`pip install "open-econs[nls]"`
  → pulls `sympy`).
- **`nlogit` (nested logit) is deferred** — see `docs/nlogit-recon.md`.
- **Synthetic control (`synth`) shipped in v0.9** (Abadie-Diamond-Hainmueller
  point estimator + ADH placebo inference). `synth.predict()` / `synth.plot()`
  remain `NotImplementedError` by design.
- **Parity CI** runs `stata`/`r`-marked tests only where the optional
  `[stata]`/`[r]` fixtures are installed; on free GitHub runners those tests
  skip. Full parity needs self-hosted Stata/R runners — see `docs/api_stability.md`.
