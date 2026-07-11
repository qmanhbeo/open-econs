import numpy as np
import pandas as pd

rng = np.random.default_rng(123)
n_below = 1500
x_below = rng.uniform(-1.0, 0.0, n_below)
n_pile = 600
x_pile = rng.uniform(0.0, 0.04, n_pile)
n_above = 900
x_above = rng.uniform(0.15, 1.0, n_above)
x = np.concatenate([x_below, x_pile, x_above])
df = pd.DataFrame({"x": x, "y": x})
df.to_csv(r"tests/stata/fixtures/df_rdd_density.csv", index=False)
print("wrote", len(df), "rows")
