"""Tobit (censored normal) regression with robust and cluster SEs.

oe.tobit estimates a censored-normal MLE. Censoring limits are set with
left= / right=. cov_type selects the robust estimator (HC0/HC1/HC2/HC3);
cluster= requests cluster-robust CRV1 SEs. Python API only.
"""

import numpy as np
import pandas as pd

import open_econs as oe

rng = np.random.default_rng(3)
n = 40

x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
group = rng.integers(0, 8, n)
latent = 1.2 * x1 - 0.7 * x2 + rng.normal(0, 1, n)
# Left-censor at 0 (Stata: tobit y x, ll(0)).
y = np.maximum(0.0, latent)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})

print("=" * 70)
print("Tobit with left censoring at 0, nonrobust (OIM) SEs")
print("=" * 70)
tobit_oim = oe.tobit("y ~ x1 + x2", data=df, left=0, cov_type="nonrobust")
print(tobit_oim.summary())
print("sigma (scale):", tobit_oim.sigma)

print("=" * 70)
print("Tobit with HC1 (robust) SEs")
print("=" * 70)
tobit_hc1 = oe.tobit("y ~ x1 + x2", data=df, left=0, cov_type="HC1")
print(tobit_hc1.summary())

print("=" * 70)
print("Tobit with cluster-robust (by group) SEs")
print("=" * 70)
tobit_clust = oe.tobit("y ~ x1 + x2", data=df, left=0, cluster="group")
print(tobit_clust.summary())
