"""Generate CEM fixture data with explicit cutpoints.

DGP:
  x1 ~ N(0, 1)
  x2 ~ Bernoulli(0.5) -- binary
  x3 ~ uniform integer {0, 1, 2} -- categorical
  T|X ~ Bernoulli(logit(-0.2 + 0.5*X1 + 0.5*X2 + 0.3*X3))
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"

rng = np.random.default_rng(12345)
n = 1000

x1 = rng.normal(0, 1, n)
x2 = rng.binomial(1, 0.5, n)
x3 = rng.integers(0, 3, n)

lp = -0.2 + 0.5 * x1 + 0.5 * x2 + 0.3 * x3
p = 1 / (1 + np.exp(-lp))
t = (rng.uniform(0, 1, n) < p).astype(float)

df = pd.DataFrame({"t": t, "x1": x1, "x2": x2, "x3": x3})
df.to_csv(FIXTURES_DIR / "df_cem.csv", index=False)

n_t = int(t.sum())
n_c = n - n_t
print(f"Wrote {FIXTURES_DIR / 'df_cem.csv'} ({n} rows)")
print(f"Treated: {n_t}, Control: {n_c}")
