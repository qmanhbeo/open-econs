"""Verify sig2 formula: test various Stata sig2 conventions."""

import sys
import os

ROOT = r"C:\Users\manhn\Desktop\open-econs"
sys.path.insert(0, os.path.join(ROOT, "tests", "stata", "generate-fixtures"))
sys.path.insert(0, os.path.join(ROOT, "tests", "stata"))
import numpy as np
import pandas as pd
from build_sys_Z import build_Z_from_raw
from build_H_verify import build_full_H
from stata_runner import read_stata

S = read_stata("sysgmm")
df = pd.read_csv(
    os.path.join(ROOT, "tests", "stata", "fixtures", "inputs", "df_panel.csv")
)

T = 5
N_ENT = 30
N_ROW_PER = 10

Z = build_Z_from_raw(df)
H = build_full_H()
Y = np.zeros(N_ENT * 2 * T)
X = np.zeros((N_ENT * 2 * T, 4))
for k in range(N_ENT):
    mask = df["entity"] == k
    sub = df.loc[mask].sort_values("time")
    y = sub["y"].values
    xv = sub["x"].values
    zv = sub["z"].values
    base = k * 2 * T
    for t in range(T):
        dr = base + t
        if t >= 1:
            Y[dr] = y[t] - y[t - 1]
        if t >= 2:
            X[dr, 0] = y[t - 1] - y[t - 2]
            X[dr, 1] = xv[t] - xv[t - 1]
            X[dr, 2] = zv[t] - zv[t - 1]
        lr = base + T + t
        Y[lr] = y[t]
        X[lr, 0] = y[t - 1] if t >= 1 else 0.0
        X[lr, 1] = xv[t]
        X[lr, 2] = zv[t]
        X[lr, 3] = 1.0

ZtHZ = Z.T @ H @ Z
A1 = np.linalg.pinv(ZtHZ)
XZ = X.T @ Z
Zy = Z.T @ Y
b_1s = np.linalg.inv(XZ @ A1 @ XZ.T) @ (XZ @ A1 @ Zy)
e_1s = Y - X @ b_1s

# Extract diff-only residuals (first T of each 10-row block)
diff_resid = np.zeros(N_ENT * (T - 1))
idx = 0
for k in range(N_ENT):
    base = k * N_ROW_PER
    for t in range(1, T):
        diff_resid[idx] = e_1s[base + t]
        idx += 1

# Level residuals (last T of each 10-row block)
lev_resid = np.zeros(N_ENT * T)
idx = 0
for k in range(N_ENT):
    base = k * N_ROW_PER + T
    for t in range(T):
        lev_resid[idx] = e_1s[base + t]
        idx += 1

ee = float(e_1s @ e_1s)
edd = float(diff_resid @ diff_resid)
ell = float(lev_resid @ lev_resid)

print(f"Total e'e  = {ee:.4f}  (N={len(e_1s)})")
print(f"Diff e'e   = {edd:.4f}  (N={len(diff_resid)})")
print(f"Level e'e  = {ell:.4f}  (N={len(lev_resid)})")
print()

for label, e2, n in [
    ("diff /120/2", edd, 120),
    ("diff /120/1", edd, 120),
    ("diff /300/2", edd, 300),
    ("lev  /150/2", ell, 150),
    ("lev  /150/1", ell, 150),
    ("lev  /120/2", ell, 120),
    ("total/300/2", ee, 300),
    ("total/300/1", ee, 300),
]:
    raw = e2 / n / (1 if "/1" in label else 2)
    sc = raw * n / (n - 4)
    print(f"  {label}: raw={raw:.6f}  sc={sc:.6f}")

print(f"\nStata targets: 1s_nr={S['sig2_c_1s_nr']:.6f}  2s_nr={S['sig2_c_2s_nr']:.6f}")
