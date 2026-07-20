"""Poisson / PPML with robust and cluster-robust standard errors.

oe.poisson wraps pyfixest's fepois and requires fixed effects
(fixed_effects= or entity=/time=). vcov_backend selects the small-sample
convention: 'fixest' (default, matches R fixest) or 'stata' (matches
Stata ppmlhdfe). All computation uses the open-econs Python API only.
"""

import numpy as np
import pandas as pd

import open_econs as oe

rng = np.random.default_rng(7)
n_firm = 12
n_year = 6
n = n_firm * n_year

firm = np.repeat(np.arange(n_firm), n_year)
year = np.tile(np.arange(n_year), n_firm)
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
fe_firm = np.repeat(rng.normal(0, 0.5, n_firm), n_year)
fe_year = np.tile(rng.normal(0, 0.5, n_year), n_firm)
mu = np.exp(0.6 * x1 - 0.3 * x2 + fe_firm + fe_year)
y = rng.poisson(mu)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})

print("=" * 70)
print("Poisson PML (PPML) with fixed effects, robust (HC1) SEs")
print("=" * 70)
pois_hc1 = oe.poisson("y ~ x1 + x2", data=df,
                     fixed_effects=["firm", "year"], cov_type="HC1")
print(pois_hc1.summary())

print("=" * 70)
print("PPML with cluster-robust SEs (by firm), Stata vcov convention")
print("=" * 70)
pois_clust = oe.poisson("y ~ x1 + x2", data=df,
                        fixed_effects=["firm", "year"],
                        cluster="firm", vcov_backend="stata")
print(pois_clust.summary())

print("\nIncidence-rate ratios (irr):")
print(pois_clust.irr())
