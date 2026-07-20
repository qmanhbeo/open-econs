"""OLS and two-way fixed effects with robust + cluster standard errors.

Demonstrates oe.ols (robust via cov_type) and oe.fe (two-way FE via
entity=/time= or fixed_effects=) on a small synthetic panel. All computation
is done with the open-econs Python API; no Stata/R subprocess is used.
"""

import numpy as np
import pandas as pd

import open_econs as oe

rng = np.random.default_rng(42)
n_firm = 10
n_year = 5
n = n_firm * n_year

firm = np.repeat(np.arange(n_firm), n_year)
year = np.tile(np.arange(n_year), n_firm)
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
# firm- and year-specific effects
fe_firm = np.repeat(rng.normal(0, 1, n_firm), n_year)
fe_year = np.tile(rng.normal(0, 1, n_year), n_firm)
y = 1.5 * x1 - 0.8 * x2 + fe_firm + fe_year + rng.normal(0, 0.5, n)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})

print("=" * 70)
print("OLS with HC1 (robust) standard errors")
print("=" * 70)
ols_robust = oe.ols("y ~ x1 + x2", data=df, cov_type="HC1")
print(ols_robust.summary())

print("=" * 70)
print("OLS with cluster-robust (by firm) standard errors")
print("=" * 70)
ols_cluster = oe.ols("y ~ x1 + x2", data=df, cluster="firm")
print(ols_cluster.summary())

print("=" * 70)
print("Two-way fixed effects (firm + year) with HC1 SEs")
print("=" * 70)
fe_model = oe.fe("y ~ x1 + x2", data=df, entity="firm", time="year", cov_type="HC1")
print(fe_model.summary())

print("=" * 70)
print("Two-way FE with multi-way cluster-robust SEs (firm + year)")
print("=" * 70)
fe_cluster = oe.fe("y ~ x1 + x2", data=df, fixed_effects=["firm", "year"])
print(fe_cluster.summary())

print("\nPoint estimates (FE slopes):")
print(fe_model.coefficients)
