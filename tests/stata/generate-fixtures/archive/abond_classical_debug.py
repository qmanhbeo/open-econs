"""Debug: trace classical VCV computation step by step."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from collections import Counter
from formulaic import Formula
from open_econs.models.linear.abond import _build_h

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

# Reproduce Z matrix construction — collapsed Run B
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
N = float(len(entities))
p = X.shape[1]
L = Z.shape[1]

entity_counts = dict(Counter(eq_entity.tolist()))
H_diag, H_off = _build_h(entity_counts, n_eq, eq_entity)
ZtHZ = 2.0 * (Z.T @ Z)
ZH_off = Z[:-1] * H_off[:, None]
ZtHZ += ZH_off.T @ Z[1:]
ZtHZ += Z[1:].T @ ZH_off

W = np.linalg.pinv(ZtHZ)
ZtX = Z.T @ X
ZtY = Z.T @ Y
G = ZtX.T @ W @ ZtX
g_sum = ZtX.T @ W @ ZtY
G_inv = np.linalg.inv(G)
b = G_inv @ g_sum
e = Y - X @ b

print(f"N (equations) = {n_eq}")
print(f"p (regressors) = {p}")
print(f"N - p = {n_eq - p}")
print(f"b = {b}")
print(f"e' e = {e @ e:.6f}")
print(f"mean(e) = {np.mean(e):.6f}")
print(f"var(e) = {np.var(e, ddof=1):.6f}")
print()

# Classical VCV: sig2 * G^{-1}
h_factor = 2.0
df = float(n_eq - p)
sig2 = float(e @ e) / (h_factor * df)
V_classical = sig2 * G_inv
se_classical = np.sqrt(np.maximum(np.diag(V_classical), 0.0))

print(f"sig2 = e'e / (2 * (N-p)) = {e @ e:.6f} / (2 * {df:.0f}) = {sig2:.8f}")
print(f"se_classical = {se_classical}")
print()

# Compare with Stata reference
stata_se = np.array([0.24668636, 0.17726977, 0.10425827])
print(f"Stata SE = {stata_se}")
print(f"oe / Stata = {se_classical / stata_se}")
print()

# Check G^{-1} diagonal
print(f"G^{-1} diagonal = {np.diag(G_inv)}")
print(f"G diagonal = {np.diag(G)}")
print(f"det(G) = {np.linalg.det(G):.6f}")
print()

# Small-sample correction: N/(N-k)
small_corr = n_eq / (n_eq - p)
print(f"Small correction N/(N-k) = {n_eq}/{n_eq-p} = {small_corr:.6f}")
print(f"se with small correction = {se_classical * np.sqrt(small_corr)}")
print(f"oe(small) / Stata = {se_classical * np.sqrt(small_corr) / stata_se}")
print()

# Stata's exact formula: sig2 = (Var*(N-1) + N*mean^2) / N / 2
# This equals e'e / (2N), NOT e'e / (2*(N-k))
sig2_stata_pre_small = (np.var(e, ddof=1) * (n_eq - 1) + n_eq * np.mean(e)**2) / n_eq / 2
print(f"Stata pre-small sig2 = (Var*(N-1) + N*mean^2) / N / 2 = {sig2_stata_pre_small:.8f}")
print(f"  = e'e / (2*N) = {float(e @ e) / (2 * n_eq):.8f}")
print()

# What if Stata uses N instead of N-k for df?
sig2_N = float(e @ e) / (h_factor * n_eq)
V_N = sig2_N * G_inv
se_N = np.sqrt(np.maximum(np.diag(V_N), 0.0))
print(f"se with N (not N-p) df = {se_N}")
print(f"oe(N) / Stata = {se_N / stata_se}")
print()

# Try small correction on top of N-based sig2
se_N_small = se_N * np.sqrt(small_corr)
print(f"se with N df + small correction = {se_N_small}")
print(f"oe(N+small) / Stata = {se_N_small / stata_se}")
