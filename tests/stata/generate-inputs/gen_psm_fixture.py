"""Generate PSM fixture data with real logit-based treatment assignment.

DGP:
  X1, X2 ~ N(0, 1)
  T|X ~ Bernoulli(logit(-0.2 + 0.5*X1 + 0.5*X2))
  Y = 2.0 + 1.5*T + 1.0*X1 + 0.8*X2 + eps,  eps ~ N(0, 1)

True ATE = 1.5.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "inputs"

rng = np.random.default_rng(12345)
n = 1000
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
lp = -0.2 + 0.5 * x1 + 0.5 * x2
p = 1 / (1 + np.exp(-lp))
t = (rng.uniform(0, 1, n) < p).astype(float)
y = 2.0 + 1.5 * t + 1.0 * x1 + 0.8 * x2 + rng.normal(0, 1, n)

df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
df.to_csv(FIXTURES_DIR / "df_psm.csv", index=False)

n_t = int(t.sum())
n_c = n - n_t
print(f"Wrote {FIXTURES_DIR / 'df_psm.csv'} ({n} rows)")
print(f"Treated: {n_t}, Control: {n_c}")
print(f"PS range: [{p.min():.4f}, {p.max():.4f}]")
