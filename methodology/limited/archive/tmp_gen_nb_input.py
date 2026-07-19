import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
firm = rng.integers(0, 30, n)
year = rng.integers(2020, 2024, n)
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
alpha_fe = rng.normal(0, 0.5, 30)[firm]
year_fe = rng.normal(0, 0.3, 4)[year - 2020]
eta = 0.5 * x1 - 0.3 * x2 + alpha_fe + year_fe
mu = np.exp(eta)
# NB2 overdispersion: y ~ NegBin(mean=mu, var=mu + alpha*mu^2)
alpha_true = 0.7
p = mu / (mu + alpha_true)
y = rng.negative_binomial(n=1, p=p)  # wait, scipy param differs
# Use manual: NB2 with size=1/alpha
g = rng.gamma(1.0/alpha_true, alpha_true*mu)  # shape=1/alpha, scale=alpha*mu
y = rng.poisson(g).astype(float)
df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})
df.to_csv("tests/r/fixtures/inputs/nbreg_input.csv", index=False)
print("y mean", y.mean(), "var", y.var(), "alpha_true", alpha_true)
print(df.head())
