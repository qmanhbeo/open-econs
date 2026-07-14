"""Debug: verify Z'HZ computation and H matrix structure."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from collections import Counter
from formulaic import Formula
from open_econs.models.linear.abond import _build_h, _tridiag_h_inv_block

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

# Reproduce the Z matrix construction from abond() — collapsed Run B
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
depths = [2, 3]  # collapsed filtered
n_endog = 1 + len(gmm_cols)
n_instr = len(depths) * n_endog + len(iv_cols)

Y_list = []
X_list = []
Z_list = []
eq_entity_list = []

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
entity_counts = dict(Counter(eq_entity.tolist()))

print(f"n_eq = {n_eq}, N = {len(entities)}")
print(f"Z shape = {Z.shape}")
print(f"entity_counts = {entity_counts}")
print()

# Build H
H_diag, H_off = _build_h(entity_counts, n_eq, eq_entity)
print(f"H_diag (first 12) = {H_diag[:12]}")
print(f"H_off  (first 12) = {H_off[:12]}")
print(f"H_off at entity boundaries:")
for k in range(n_eq - 1):
    if H_off[k] == 0.0:
        print(f"  H_off[{k}] = 0 (boundary between {eq_entity[k]} and {eq_entity[k+1]})")
print()

# Compute Z'HZ
ZtHZ = 2.0 * (Z.T @ Z)
ZH_off = Z[:-1] * H_off[:, None]
ZtHZ += ZH_off.T @ Z[1:]
ZtHZ += Z[1:].T @ ZH_off

# Compute Z'Z for comparison
ZtZ = Z.T @ Z

print("Z'Z =")
print(np.array2string(ZtZ, precision=6))
print()
print("Z'HZ =")
print(np.array2string(ZtHZ, precision=6))
print()

# Eigenvalue comparison
evals_ZtZ = np.linalg.eigvalsh(ZtZ)
evals_ZtHZ = np.linalg.eigvalsh(ZtHZ)
print(f"Eigenvalues of Z'Z:     {evals_ZtZ}")
print(f"Eigenvalues of Z'HZ:    {evals_ZtHZ}")
print(f"Ratio Z'HZ/Z'Z diag:    {np.diag(ZtHZ) / np.diag(ZtZ)}")
print()

# Now compute W and the sandwich with both
W_old = np.linalg.pinv(ZtZ)
W_new = np.linalg.pinv(ZtHZ)

print(f"W_old (pinv(Z'Z)) diagonal = {np.diag(W_old)}")
print(f"W_new (pinv(Z'HZ)) diagonal = {np.diag(W_new)}")
print(f"W_new/W_old diagonal ratio = {np.diag(W_new) / np.diag(W_old)}")
print()

# Compute G and compare
ZtX = Z.T @ X
G_old = ZtX.T @ W_old @ ZtX
G_new = ZtX.T @ W_new @ ZtX
print("G_old =")
print(np.array2string(G_old, precision=6))
print("G_new =")
print(np.array2string(G_new, precision=6))
print(f"det(G_old) = {np.linalg.det(G_old):.8f}")
print(f"det(G_new) = {np.linalg.det(G_new):.8f}")
print()

# Compute S_g and V for both
from collections import Counter

def sandwich(Y, X, Z, eq_entity, W):
    p = X.shape[1]
    ZtX = Z.T @ X
    ZtY = Z.T @ Y
    G = ZtX.T @ W @ ZtX
    g_sum = ZtX.T @ W @ ZtY
    G_inv = np.linalg.inv(G)
    b = G_inv @ g_sum
    e = Y - X @ b

    S_g = np.zeros((p, p))
    for ent in np.unique(eq_entity):
        mask = eq_entity == ent
        Zc = Z[mask]
        Xc = X[mask]
        ec = e[mask]
        Zte = Zc.T @ ec
        XtZ = Xc.T @ Zc
        gi = XtZ @ W @ Zte
        S_g += np.outer(gi, gi)

    V = G_inv @ S_g @ G_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return b, se, G, S_g, V, G_inv

b_old, se_old, G_old2, Sg_old, V_old, Ginv_old = sandwich(Y, X, Z, eq_entity, W_old)
b_new, se_new, G_new2, Sg_new, V_new, Ginv_new = sandwich(Y, X, Z, eq_entity, W_new)

print("=== COEFFICIENTS ===")
print(f"  b_old = {b_old}")
print(f"  b_new = {b_new}")
print(f"  ratio = {b_new / b_old}")
print()

print("=== STANDARD ERRORS ===")
print(f"  se_old = {se_old}")
print(f"  se_new = {se_new}")
print(f"  ratio  = {se_new / se_old}")
print(f"  Stata  = [0.24668636, 0.17726977, 0.10425827]")
print(f"  oe/Stata (old) = {se_old / np.array([0.24668636, 0.17726977, 0.10425827])}")
print(f"  oe/Stata (new) = {se_new / np.array([0.24668636, 0.17726977, 0.10425827])}")
print()

print("=== S_g MATRICES ===")
print("Sg_old =")
print(np.array2string(Sg_old, precision=8))
print("Sg_new =")
print(np.array2string(Sg_new, precision=8))
print(f"Sg_new/Sg_old diag ratio = {np.diag(Sg_new) / np.diag(Sg_old)}")
print()

print("=== G^{-1} MATRICES ===")
print("Ginv_old =")
print(np.array2string(Ginv_old, precision=8))
print("Ginv_new =")
print(np.array2string(Ginv_new, precision=8))
print()

print("=== V MATRICES ===")
print("V_old =")
print(np.array2string(V_old, precision=8))
print("V_new =")
print(np.array2string(V_new, precision=8))
