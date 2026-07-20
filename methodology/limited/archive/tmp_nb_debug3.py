import pandas as pd
import numpy as np
from scipy.special import gammaln

df = pd.read_csv("tests/r/fixtures/inputs/nbreg_input.csv")
y = df["y"].values.astype(float)
X = np.column_stack([np.ones(len(df)), df["x1"].values, df["x2"].values])
b2 = np.array([0.01797124, 0.4145353, -0.14962152])  # Stata NB2 coefs
b1 = np.array([-0.0244143, 0.49289614, -0.20775441])  # Stata NB1 coefs


# Candidate A: standard NB2 gamma mixture (Var = mu + alpha*mu^2)
def llA(beta, alpha):
    mu = np.exp(X @ beta)
    a = 1 / alpha
    return (
        a * np.log(a / (a + mu))
        + y * np.log(mu / (a + mu))
        + gammaln(y + a)
        - gammaln(a)
        - gammaln(y + 1.0)
    ).sum()


# Candidate B: NB1-style (Var = mu(1+alpha)) log-likelihood
def llB(beta, alpha):
    mu = np.exp(X @ beta)
    a = 1 / alpha
    return (
        y * np.log(mu)
        - (y + a) * np.log(mu + alpha)
        + a * np.log(alpha)
        + gammaln(y + a)
        - gammaln(a)
        - gammaln(y + 1.0)
    ).sum()


# Candidate C: Var = mu + alpha*mu^2 but NB1-style log with (mu + alpha*mu^2)
def llC(beta, alpha):
    mu = np.exp(X @ beta)
    a = 1 / alpha
    V = mu + alpha * mu**2
    return (
        y * np.log(mu)
        - (y + a) * np.log(V)
        + a * np.log(alpha)
        + gammaln(y + a)
        - gammaln(a)
        - gammaln(y + 1.0)
    ).sum()


# Candidate D: Var = mu + alpha*mu (true NB1 linear) gamma mixture style
def llD(beta, alpha):
    mu = np.exp(X @ beta)
    a = 1 / alpha
    return (
        a * np.log(a / (a + mu))
        + y * np.log(mu / (a + mu))
        + gammaln(y + a)
        - gammaln(a)
        - gammaln(y + 1.0)
    ).sum()  # same as A


print("At Stata NB2 coefs (b2), alpha=1.263565:")
print("  llA", llA(b2, 1.263565), "(Stata ll -842.20281)")
print("  llB", llB(b2, 1.263565))
print("  llC", llC(b2, 1.263565))
print("At Stata NB2 coefs, alpha=1.0563:")
print("  llA", llA(b2, 1.0563))
print("At Stata NB1 coefs (b1), alpha=1.0563:")
print("  llA", llA(b1, 1.0563), "(Stata NB1 ll -836.53808)")
print("  llB", llB(b1, 1.0563))
print("  llC", llC(b1, 1.0563))
