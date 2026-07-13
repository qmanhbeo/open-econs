# Tutorial: RDD (Regression Discontinuity) — *planned post-1.0*

> **Status: deferred from the v1.0 tutorial set.** The estimator is implemented
> (`oe.rdd` and `oe.density_test` in `open_econs.models.causal.rdd`), but a full
> worked walkthrough is scheduled after the v1.0 release.

Quick pointer until then:

```python
import open_econs as oe

res = oe.rdd(df, y="y", running="x", cutoff=0.0)   # sharp RDD
res.tidy(); res.summary()
oe.density_test(df, running="x", cutoff=0.0)       # McCrary density test
```

`oe.rdd` mirrors R's `rdrobust` / `rddensity` (optional `[rd]` extras). A
complete tutorial with bandwidth selection, fuzzy RDD, and plotting will land
post-1.0.
