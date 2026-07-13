# Tutorial: PSM (Propensity Score Matching) — *planned post-1.0*

> **Status: deferred from the v1.0 tutorial set.** The estimator is implemented
> (`oe.psm`, `oe.cem`, `oe.balance`, `oe.rosenbaum_bounds` in
> `open_econs.models.causal`), but a full worked walkthrough is scheduled after
> the v1.0 release.

Quick pointer until then:

```python
import open_econs as oe

psm_res = oe.psm(df, treatment="treat", covariates=["x1", "x2"])   # 1:1 nearest-neighbour
psm_res.tidy(); psm_res.summary()

cem_res = oe.cem(df, treatment="treat", covariates=["x1", "x2"])   # coarsened exact matching
bal = oe.balance(df, treatment="treat", covariates=["x1", "x2"])   # covariate balance table
```

Matching + balance + Rosenbaum sensitivity (`oe.rosenbaum_bounds`) map to R's
`MatchIt` / `Matching` and Stata's `teffects psmatch`. A complete tutorial is
planned post-1.0.
