"""Ordered logit with robust standard errors.

oe.ologit fits a proportional-odds ordered logit. The dependent variable
must be an integer-coded ordered variable with at least 3 levels. cov_type
selects nonrobust / HC0 / HC1 / HC2 / HC3. Python API only.
"""

import numpy as np
import pandas as pd

import open_econs as oe

rng = np.random.default_rng(11)
n = 45

x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
# Latent score mapped to 3 ordered categories.
score = 1.0 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
y = np.digitize(score, bins=[-0.8, 0.8]).astype(int)  # levels 0,1,2

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

print("=" * 70)
print("Ordered logit, nonrobust (OIM) SEs")
print("=" * 70)
ologit_oim = oe.ologit("y ~ x1 + x2", data=df, cov_type="nonrobust")
print(ologit_oim.summary())
print("cutpoints:", dict(ologit_oim.cutpoints))

print("=" * 70)
print("Ordered logit with HC1 (robust) SEs")
print("=" * 70)
ologit_hc1 = oe.ologit("y ~ x1 + x2", data=df, cov_type="HC1")
print(ologit_hc1.summary())
