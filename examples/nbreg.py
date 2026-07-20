"""Negative binomial regression with Stata constant dispersion.

oe.nbreg fits NB1 (dispersion='mean') or NB2. dispersion='const_stata'
reproduces Stata's nbreg, dispersion(constant) MLE (pooled only). Robust
and cluster SEs are available. Python API only.
"""

import numpy as np
import pandas as pd

import open_econs as oe

rng = np.random.default_rng(5)
n = 50

x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
mu = np.exp(0.5 * x1 - 0.3 * x2)
# Overdispersed counts (negative binomial with alpha = 1.0).
alpha = 1.0
p = alpha / (alpha + mu)
y = rng.negative_binomial(n=1.0 / alpha, p=p / (p + (1 - p)))

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": rng.integers(0, 10, n)})

print("=" * 70)
print("Negative binomial (Stata constant dispersion) with HC1 SEs")
print("=" * 70)
nb_stata = oe.nbreg("y ~ x1 + x2", data=df,
                    dispersion="const_stata", cov_type="HC1")
print(nb_stata.summary())
print("alpha:", nb_stata.alpha(), " theta (1/alpha):", nb_stata.theta())

print("=" * 70)
print("Negative binomial (NB2, default) with cluster-robust SEs (by group)")
print("=" * 70)
nb_clust = oe.nbreg("y ~ x1 + x2", data=df,
                    dispersion="const", cluster="group")
print(nb_clust.summary())
