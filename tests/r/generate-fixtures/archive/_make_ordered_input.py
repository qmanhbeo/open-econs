import numpy as np
import pandas as pd

rng = np.random.default_rng(20240717)
n = 600

# Covariates
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
x3 = rng.normal(0, 1, n)
# latent linear predictor with known coefficients
beta = np.array([0.7, -0.5, 0.3])
eta = x1 * beta[0] + x2 * beta[1] + x3 * beta[2]
# fixed cutpoints -> 4 ordered categories (0,1,2,3)
err = rng.normal(0, 1, n)
y_star = eta + err
# cutpoints at -0.8, 0.2, 1.0
y = np.digitize(y_star, [-0.8, 0.2, 1.0]).astype(int)

df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
df.to_csv("tests/r/fixtures/inputs/ordered_input.csv", index=False)
print(df["y"].value_counts().sort_index())
print("rows", len(df))
