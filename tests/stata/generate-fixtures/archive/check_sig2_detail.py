"""Detailed sig2 investigation."""
import sys, os
ROOT = r"C:\Users\manhn\Desktop\open-econs"
sys.path.insert(0, os.path.join(ROOT, "tests", "stata", "generate-fixtures"))
sys.path.insert(0, os.path.join(ROOT, "tests", "stata"))
import numpy as np
import pandas as pd
from build_sys_Z import build_Z_from_raw
from build_H_verify import build_full_H
from stata_runner import read_stata

S = read_stata("sysgmm")
df = pd.read_csv(os.path.join(ROOT, "tests", "stata", "fixtures", "inputs", "df_panel.csv"))

T = 5; N_ENT = 30
Z = build_Z_from_raw(df)
H = build_full_H()
Y = np.zeros(N_ENT * 2 * T)
X = np.zeros((N_ENT * 2 * T, 4))
for k in range(N_ENT):
    mask = df["entity"] == k
    sub = df.loc[mask].sort_values("time")
    y = sub["y"].values; xv = sub["x"].values; zv = sub["z"].values
    base = k * 2 * T
    for t in range(T):
        dr = base + t; lr = base + T + t
        if t >= 1: Y[dr] = y[t] - y[t-1]
        if t >= 2: X[dr, 0] = y[t-1] - y[t-2]; X[dr, 1] = xv[t] - xv[t-1]; X[dr, 2] = zv[t] - zv[t-1]
        Y[lr] = y[t]
        X[lr, 0] = y[t-1] if t >= 1 else 0.0
        X[lr, 1] = xv[t]; X[lr, 2] = zv[t]; X[lr, 3] = 1.0

ZtHZ = Z.T @ H @ Z
A1 = np.linalg.pinv(ZtHZ)
XZ = X.T @ Z; Zy = Z.T @ Y
b_1s = np.linalg.inv(XZ @ A1 @ XZ.T) @ (XZ @ A1 @ Zy)
e_1s = Y - X @ b_1s  # 300 residuals

# Try different diff-residual subsets
for label, t_range in [("t>=1", range(1, T)), ("t>=2", range(2, T)), ("t>=0", range(T)), ("t>=3", range(3, T))]:
    diff_resid = np.zeros(N_ENT * len(list(t_range)))
    idx = 0
    for k in range(N_ENT):
        base = k * 2 * T
        for t in t_range:
            diff_resid[idx] = e_1s[base + t]; idx += 1
    N = len(diff_resid)
    ee = float(diff_resid @ diff_resid)
    for denom in [1, 2]:
        raw = ee / N / denom
        sc = raw * N / (N - 4)
        print(f"  {label} N={N} e'e={ee:.2f} /{N}/{denom}: raw={raw:.6f} sc={sc:.6f}")

# Try level residuals only
for label, t_range in [("lev t>=0", range(T)), ("lev t>=1", range(1, T))]:
    lev_resid = np.zeros(N_ENT * len(list(t_range)))
    idx = 0
    for k in range(N_ENT):
        base = k * 2 * T + T
        for t in t_range:
            lev_resid[idx] = e_1s[base + t]; idx += 1
    N = len(lev_resid)
    ee = float(lev_resid @ lev_resid)
    raw = ee / N
    sc = raw * N / (N - 4)
    print(f"  {label} N={N}: raw={raw:.6f} sc={sc:.6f}")

print(f"Target 1s_nr sig2 = {S['sig2_c_1s_nr']:.6f}")
