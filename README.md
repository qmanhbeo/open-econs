# open-econs

**The scikit-learn of empirical economics.**

A Python library that bridges the gap between traditional Stata/R econometrics
workflows and modern, production-grade Python systems.  Every estimator follows
the same interface — `fit`, `summary`, `tidy`, `export` — so researchers and
AI agents never have to learn a new API.

## Quick Start

```python
import open_econs as oe

# Ordinary least squares – coefficients are named (pd.Series)
result = oe.ols("income ~ education + age", data=df, cluster="province")
print(result)
print(result.coefficients["education"])   # named access, not positions
result.tidy()                              # coefficient table (DataFrame)
result.predict(newdata=df_test)            # out-of-sample predictions

# reg() is an alias for ols()
result2 = oe.reg("income ~ education + age", data=df)

# Oaxaca-Blinder decomposition
decomp = oe.oaxaca(
    "income ~ education + age + female",
    data=df, by="female",
)
decomp.tidy()

# Context-based (dataset remembered)
ctx = oe.Context(df)
r1 = ctx.ols("income ~ education + age")
r2 = ctx.oaxaca("income ~ education + age + female", by="female")
```

## Installation

```bash
pip install open-econs
```

Requires Python ≥ 3.12.

## Design Principles

- **Every result is immutable** once `fit()` completes.
- **All numeric artifacts are named** (`pd.Series`/`pd.DataFrame` with
  variable-name indices).  No raw `numpy.ndarray` crosses the public API.
- **Every error tells you what to fix.** Missing column → names the column,
  lists what's available. Non-binary `by` → shows the values found.
- **Consistent interface across estimators**: `summary()`, `tidy()`,
  `export()`, `predict()` (where applicable).

## v0.1 Estimators

| Function | Description |
|---|---|
| `ols()` / `reg()` | Ordinary least squares with HC1/robust/clustered SEs |
| `oaxaca()` | Oaxaca-Blinder decomposition (two-fold, three-fold) |

### Planned (future releases)

- `fe()` — fixed effects
- `iv()` — instrumental variables
- `logit()` / `probit()` — binary choice
- `did()` — difference-in-differences
- `psm()` — propensity score matching

## Result API

Every estimator returns an object with:

| Method | Returns |
|---|---|
| `.summary()` | Printable string (also `__repr__`) |
| `.tidy()` | `pd.DataFrame` — coefficient or effect table |
| `.predict(newdata)` | `pd.Series` — only on regression models |
| `.export(path)` | JSON serialization (`.json` only in v0.1) |
| `.plot()` | *Not yet implemented — raises with clear message* |
| `.to_dict()` | `dict` — full result metadata |

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```