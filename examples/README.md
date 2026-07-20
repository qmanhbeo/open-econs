# open-econs examples

Runnable, self-contained quickstart scripts demonstrating the
[open-econs](https://github.com/anomalyco/opencode) applied-micro
econometrics API. Every script generates its own small synthetic dataset
(numpy / pandas) — **no Stata or R subprocess is invoked** (per project
Standard of Practice rule 24).

The project's thesis is machine-precision parity with Stata and R
(numeric tolerance ≤ 1e-6 on point estimates, standard errors, and
associated quantities). These examples focus on *usage*; see the
methodology notes for parity proofs. They do not independently verify
parity — run the test suite for that.

## Run

From the repo root:

```bash
python examples/ols_fe.py
python examples/poisson.py
python examples/tobit.py
python examples/ordered.py
python examples/nbreg.py
python examples/diagnostics.py
```

Each script fits a model, prints a `summary()`-style table, and shows
the relevant robust / cluster standard errors.

## What each script shows

| Script | Model | Key API | SE demonstrations |
| --- | --- | --- | --- |
| `ols_fe.py` | OLS + two-way fixed effects | `oe.ols`, `oe.fe` | `cov_type="HC1"` (robust), `cluster="firm"`, `fixed_effects=["firm","year"]` (multi-way) |
| `poisson.py` | Poisson / PPML | `oe.poisson` | `cov_type="HC1"`, `cluster="firm"`, `vcov_backend="stata"` |
| `tobit.py` | Tobit (censored normal) | `oe.tobit` | `cov_type="nonrobust"`/`"HC1"`, `cluster="group"`; censoring via `left=` |
| `ordered.py` | Ordered logit | `oe.ologit` | `cov_type="nonrobust"`/`"HC1"`; 3+ level ordered `y` |
| `nbreg.py` | Negative binomial | `oe.nbreg` | `dispersion="const_stata"`, `cov_type="HC1"`, `cluster="group"` |
| `diagnostics.py` | Influence | `result.dfbetas(backend="stata_r")` | standardized DFBETAS on a small OLS fit |

## Notes on the API used

- **Fixed effects require fixed effects.** `oe.poisson` requires
  `fixed_effects=` (or `entity=`/`time=`); `oe.fe` requires
  `entity=`/`time=` or `fixed_effects=`.
- **DFBETAS** is exposed as a result method `result.dfbetas(backend=...)`,
  not as a top-level `oe.dfbetas` import.
- **`vcov_backend`** on `oe.poisson` / `oe.nbreg` selects the
  small-sample convention: `"fixest"` (default, R fixest) or `"stata"`
  (Stata `ppmlhdfe` / `nbreg`). Point estimates are identical either way.
- Overdispersion in `oe.nbreg`: `dispersion="const"` (NB2), `"mean"`
  (NB1), or `"const_stata"` (Stata `nbreg, dispersion(constant)`, pooled
  only).

See `open_econs/__init__.py` for the full export list.
