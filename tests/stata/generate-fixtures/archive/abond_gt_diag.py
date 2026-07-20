"""Step 2/3 diagnostic: compute GMM intermediates from Stata's REAL extracted
matrices (per-entity 5x5 H block, 150-row Z/X/Y) and from oe's current pipeline.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\manhn\Desktop\open-econs")
import open_econs as oe

DO = r"C:\Users\manhn\Desktop\open-econs\tests\stata\do"


def load(name):
    return pd.read_csv(f"{DO}\\abond_gt_{name}.csv")


Xdf = load("X")
Ydf = load("Y")
Zdf = load("Z")
Hdf = load("H")


# Matrix columns are appended AFTER the original y,x,z,entity,time cols.
def mat_cols(df, prefix, n):
    return [c for c in df.columns if c.startswith(prefix)][:n]


Z = Zdf[mat_cols(Zdf, "Zmat", 5)].to_numpy(dtype=float)  # 150 x 5 (incl. 1 zero col)
X = Xdf[mat_cols(Xdf, "Xmat", 3)].to_numpy(dtype=float)  # 150 x 3
Y = Ydf[mat_cols(Ydf, "Ymat", 1)].to_numpy(dtype=float).ravel()  # 150
H = Hdf[mat_cols(Hdf, "Hmat", 5)].to_numpy(dtype=float)[
    :5, :
]  # real 5x5 block (rest NaN)
ent = Xdf["entity"].to_numpy()
tvals = Xdf["time"].to_numpy()
entities = np.unique(ent)
T = 5

print(f"X={X.shape} Y={Y.shape} Z={Z.shape} H={H.shape}")
print(f"H (5x5) =\n{H}")
print(f"H diagonal = {np.diag(H)}")

# Entity 0 full block
m0 = ent == entities[0]
print(f"\n=== Entity {int(entities[0])}: rows t=0..4 ===")
order0 = np.where(m0)[0]
for i in order0:
    print(f"  t={int(tvals[i])}: Z={Z[i]}  Y={Y[i]:.6f}  X={X[i]}")

# ---- Accumulate per-entity Z'HZ, Z'X, Z'Y using Stata's H (5x5) ----
ZtHZ = np.zeros((5, 5))
ZtX = np.zeros((5, 3))
ZtY = np.zeros(5)
for e_val in entities:
    mm = ent == e_val
    Ze = Z[mm]  # 5 x 5
    Xe = X[mm]  # 5 x 3
    Ye = Y[mm]  # 5
    ZtHZ += Ze.T @ H @ Ze
    ZtX += Ze.T @ Xe
    ZtY += Ze.T @ Ye

print("\n=== From Stata's REAL X/Y/Z/H (per-entity 5x5 H) ===")
print(f"Z'X =\n{ZtX}")
print(f"Z'Y = {ZtY}")
print(f"Z'HZ (with saved e(H)) =\n{ZtHZ}")

p = X.shape[1]
n_eq = int((tvals >= 2).sum())
df = n_eq - p


def build_H_true(T):
    """Unnormalized M'M: t=0 row/col zeroed; diag 2, off -1 for t>=1."""
    H = np.zeros((T, T))
    for t in range(1, T):
        H[t, t] = 2.0
    for t in range(1, T - 1):
        H[t, t + 1] = H[t + 1, t] = -1.0
    return H


def build_H_true_t0(T):
    """M'M including t=0: diag 1 at t=0, then 2, off -1 throughout."""
    H = np.zeros((T, T))
    for t in range(T):
        H[t, t] = 1.0 if t == 0 else 2.0
    for t in range(T - 1):
        H[t, t + 1] = H[t + 1, t] = -1.0
    return H


for label, Htrue in [
    ("H_true(t0 zeroed)", build_H_true(T)),
    ("H_true(t0 included)", build_H_true_t0(T)),
]:
    ZtHZ_t = np.zeros((5, 5))
    for e_val in entities:
        mm = ent == e_val
        ZtHZ_t += Z[mm].T @ Htrue @ Z[mm]
    Wt = np.linalg.pinv(ZtHZ_t)
    Gt = ZtX.T @ Wt @ ZtX
    Ginv_t = np.linalg.inv(Gt)
    bt = Ginv_t @ (ZtX.T @ Wt @ ZtY)
    et = Y - X @ bt
    sig2t = float(et @ et) / (2.0 * df)
    Vt = sig2t * Ginv_t
    print(f"\n--- {label} ---")
    print(f"b = {bt}")
    print(f"V diag = {np.diag(Vt)}")
    print(f"SE = {np.sqrt(np.diag(Vt))}")
    print(f"Stata SE = {np.sqrt([0.06085416, 0.03142457, 0.01086979])}")

L = 5
W = np.linalg.pinv(ZtHZ)
G = ZtX.T @ W @ ZtX
G_inv = np.linalg.inv(G)
b = G_inv @ (ZtX.T @ W @ ZtY)
e = Y - X @ b
sig2 = float(e @ e) / (2.0 * df)
V = sig2 * G_inv
print(f"\nb (from Stata matrices) = {b}")
print("Stata e(b)             = [-0.11984163, 1.1258209, -0.28974145]")
print(f"sig2 = {sig2:.8f}   Stata e(sig2)=0.19753252   (df={df}, n_eq={n_eq}, p={p})")
print(f"\nV (from Stata matrices) =\n{V}")
print(f"\nStata e(V) diag = {np.sqrt([0.06085416, 0.03142457, 0.01086979])}")
print(f"V from Stata matrices diag = {np.sqrt(np.diag(V))}")

# ---- Run oe current pipeline ----
dfp = pd.read_csv(
    r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv"
)
res = oe.abond(
    "y ~ x + z",
    data=dfp,
    entity="entity",
    time="time",
    step="one-step",
    lags=1,
    exogenous=["x", "z"],
    collapse=True,
    robust=False,
)
print("\n=== oe.abond current pipeline ===")
print(f"coefficients = {res.coefficients.values}")
print(f"std_errors    = {res.std_errors.values}")
print(f"n_obs = {res.n_obs}, n_instruments = {res.n_instruments}")
