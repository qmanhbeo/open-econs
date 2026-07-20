"""Influence diagnostics: DFBETAS on a small OLS model.

DFBETAS measure how much each coefficient changes when an observation is
dropped (standardized by the leave-one-out SE). oe exposes it as a result
method: result.dfbetas(backend='stata_r'), which matches R stats::dfbetas
and Stata predict, dfbeta. Python API only.
"""

import numpy as np
import pandas as pd

import open_econs as oe

rng = np.random.default_rng(1)
n = 30

x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
y = 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5, n)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

fit = oe.ols("y ~ x1 + x2", data=df, cov_type="HC1")
print("=" * 70)
print("OLS fit")
print("=" * 70)
print(fit.summary())

print("=" * 70)
print("DFBETAS (standardized, Stata/R backend)")
print("=" * 70)
d = fit.dfbetas(backend="stata_r")
print(d.to_string())

print("\nObservations with |DFBETAS| > 1 (influential by the common rule):")
mask = (d.abs() > 1).any(axis=1)
print(d.index[mask].tolist())
