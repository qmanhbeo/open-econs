"""Debug: compare Z'HZ with corrected H (diagonal=3)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from collections import Counter
from formulaic import Formula
from open_econs.models.linear.abond import _build_h

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

formula_obj = Formula("y ~ x + z")
mm = formula_obj.get_model_matrix(df, na_action="drop")
y_name = mm.lhs.columns[0]
x_cols = [c for c in mm.rhs.columns if c != "Intercept"]

_df = df.loc[mm.rhs.index].copy()
_df["__y"] = mm.lhs[y_name].values.ravel()
for c in x_cols:
    _df["__x__" + c] = mm.rhs[c].values

ent_vals = _df["entity"].values
time_vals = _df["time"].values
order = np.lexsort((time_vals, ent_vals))
ent_sorted = ent_vals[order]
y_sorted = _df["__y"].values[order]
x_sorted = {c: _df["__x__" + c].values[order] for c in x_cols}

entities = []
y_by_e = {}
x_by_e = {}
for e_val in pd.unique(ent_sorted):
    mask = ent_sorted == e_val
    entities.append(e_val)
    y_by_e[e_val] = y_sorted[mask]
    x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

exogenous = ["x", "z"]
exo_set = set(exogenous)
gmm_cols = [c for c in x_cols if c not in exo_set]
iv_cols = [c for c in x_cols if c in exo_set]
min_j = 2
maxL = 4
depths = [2, 3]
n_endog = 1 + len(gmm_cols)
n_instr = len(depths) * n_endog + len(iv_cols)

Y_list, X_list, Z_list, eq_entity_list = [], [], [], []
for e_val in entities:
    y = y_by_e[e_val]
    xs = x_by_e[e_val]
    T = len(y)
    for j in range(min_j, T):
        dep = y[j] - y[j - 1]
        dyn_regs = [y[j - lag] - y[j - lag - 1] for lag in range(1, 2)]
        x_regs = [xs[c][j] - xs[c][j - 1] for c in x_cols]
        X_list.append(dyn_regs + x_regs)
        zrow = np.zeros(n_instr)
        col = 0
        for lag in depths:
            if j - lag >= 0:
                zrow[col] = y[j - lag]
            col += 1
        for gmm_c in gmm_cols:
            for lag in depths:
                if j - lag >= 0:
                    zrow[col] = xs[gmm_c][j - lag]
                col += 1
        for iv_c in iv_cols:
            zrow[col] = xs[iv_c][j] - xs[iv_c][j - 1]
            col += 1
        Z_list.append(zrow)
        Y_list.append(dep)
        eq_entity_list.append(e_val)

Y = np.array(Y_list, dtype=float)
X = np.array(X_list, dtype=float)
Z = np.array(Z_list, dtype=float)
eq_entity = np.array(eq_entity_list)

n_eq = Y.shape[0]
N_eq = float(len(entities))
p = X.shape[1]
L = Z.shape[1]

entity_counts = dict(Counter(eq_entity.tolist()))

# Build H with corrected diagonal=3
H_diag, H_off = _build_h(entity_counts, n_eq, eq_entity)
print(f"H_diag (first 12) = {H_diag[:12]}")
print(f"H_off  (first 12) = {H_off[:12]}")
print()

# Compute Z'HZ with corrected H
ZtHZ = 3.0 * (Z.T @ Z)
ZH_off = Z[:-1] * H_off[:, None]
ZtHZ += ZH_off.T @ Z[1:]
ZtHZ += Z[1:].T @ ZH_off

print("Z'HZ (corrected H, diagonal=3):")
print(np.array2string(ZtHZ, precision=6))
print()

# Compute G and V
ZtX = Z.T @ X
W = np.linalg.pinv(ZtHZ)
G = ZtX.T @ W @ ZtX
G_inv = np.linalg.inv(G)
g_sum = ZtX.T @ W @ (Z.T @ Y)
b = G_inv @ g_sum
e = Y - X @ b

# Classical VCV
h_factor = 3.0  # diagonal value for usable equations
df = float(n_eq - p)
sig2 = float(e @ e) / (h_factor * df)
V = sig2 * G_inv
se = np.sqrt(np.maximum(np.diag(V), 0.0))

print(f"b = {b}")
print(f"e'e = {e @ e:.6f}")
print(f"sig2 = e'e / ({h_factor} * {df:.0f}) = {sig2:.8f}")
print(f"se = {se}")
print()

stata_se = np.array([0.24668636, 0.17726977, 0.10425827])
print(f"Stata SE = {stata_se}")
print(f"oe / Stata = {se / stata_se}")
print()

# Small correction
small_corr = n_eq / (n_eq - p)
se_small = se * np.sqrt(small_corr)
print(f"se with small correction = {se_small}")
print(f"oe(small) / Stata = {se_small / stata_se}")
print()

# What if Stata uses different sig2? Try sig2 = e'e / (N * (2 - (h==1)))
sig2_alt = float(e @ e) / (n_eq * 2.0)
V_alt = sig2_alt * G_inv
se_alt = np.sqrt(np.maximum(np.diag(V_alt), 0.0))
print(f"sig2_alt (e'e / (2*N)) = {sig2_alt:.8f}")
print(f"se_alt = {se_alt}")
print(f"oe_alt / Stata = {se_alt / stata_se}")
print()

# What if the H matrix should be MM' not M'M+I?
# MM' for 3x3 case: [2,-1,0; -1,2,-1; 0,-1,2]
# M'M+I for 3x3 case: [3,-1,0; -1,3,-1; 0,-1,3]
# Check: maybe Stata uses MM' directly for the usable equations?
ZtHZ_mm = 2.0 * (Z.T @ Z)
ZtHZ_mm += ZH_off.T @ Z[1:]  # off-diag from H_off=-1
ZtHZ_mm += Z[1:].T @ ZH_off
W_mm = np.linalg.pinv(ZtHZ_mm)
G_mm = ZtX.T @ W_mm @ ZtX
G_mm_inv = np.linalg.inv(G_mm)
b_mm = G_mm_inv @ (ZtX.T @ W_mm @ (Z.T @ Y))
e_mm = Y - X @ b_mm
sig2_mm = float(e_mm @ e_mm) / (2.0 * df)
V_mm = sig2_mm * G_mm_inv
se_mm = np.sqrt(np.maximum(np.diag(V_mm), 0.0))
print(f"se with MM' (diag=2) = {se_mm}")
print(f"oe(MM') / Stata = {se_mm / stata_se}")
