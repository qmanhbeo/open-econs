# Tutorial: Synthetic Control — *planned post-1.0*

> **Status: deferred from the v1.0 tutorial set.** The estimator is implemented
> and shipped in v0.9 (`oe.synth`, `oe.SynthResult`, plus ADH permutation
> inference `placebo_space()` / `placebo_time()` in
> `open_econs.models.causal.synth`). A full worked walkthrough is scheduled after
> the v1.0 release.

Quick pointer until then:

```python
import open_econs as oe

res = oe.synth(
    df,
    outcome="y", treated_unit="A", entity="unit", time="year",
    pre_period=2000, post_period=2010,
    predictors=["x1", "x2"],        # explicit controls (optional)
)
res.tidy()                          # donor weights
res.summary()
res.placebo_space(df)               # ADH permutation-in-space p-value
res.placebo_time(df)                # ADH permutation-in-time p-value
```

`oe.synth` mirrors R's `Synth` and Stata's `synth` (gated parity tests exist and
skip cleanly where R/Stata are absent). Known limitations: `SynthResult.predict()`
and `SynthResult.plot()` are `NotImplementedError` by design in this pass. A
complete tutorial is planned post-1.0.
