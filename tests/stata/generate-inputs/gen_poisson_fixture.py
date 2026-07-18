"""Generate the canonical Poisson-FE parity input dataset (committed artifact).

Run once to (re)create ``poisson_input.csv``. The CSV is the committed input read
by BOTH the R (`fixest::fepois`) and Stata (`ppmlhdfe`) generators and the Python
parity tests, so all three sides see identical bytes.

Design: a two-way FE count panel (25 firms x 10 years, N=500) with a genuine
Poisson DGP, plus an ``exposure`` column (log-offset) and a positive analytic
``wt`` column so offset/weights options are exercised. Seeded for reproducibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "r" / "fixtures" / "inputs" / "poisson_input.csv"

rng = np.random.default_rng(12345)
n = 500
firm = rng.integers(0, 25, n)
year = rng.integers(2000, 2010, n)
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
fe_firm = rng.normal(0, 0.5, 25)[firm]
fe_year = rng.normal(0, 0.3, 10)[year - 2000]
eta = 0.5 * x1 - 0.3 * x2 + fe_firm + fe_year
mu = np.exp(eta)
y = rng.poisson(mu)
exposure = rng.uniform(0.5, 2.0, n)
wt = rng.uniform(0.5, 2.0, n)

df = pd.DataFrame({
    "y": y,
    "x1": x1,
    "x2": x2,
    "firm": firm,
    "year": year,
    "exposure": exposure,
    "wt": wt,
})
df.to_csv(OUT, index=False)
print("wrote", OUT, df.shape, "ysum", int(y.sum()))
