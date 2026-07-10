"""Diagnose non-collapsed Z matrix for singularity."""
import pandas as pd
import numpy as np
import open_econs.models.linear.abond as abond_mod

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

# Manually replicate the non-collapsed Z construction to inspect.
formula = "y ~ x + z"
data = df
entity = "entity"
time = "time"
lags = 1
exogenous = ["x", "z"]
collapse = False
robust = False

from formulaic import Formula

formula_obj = Formula(formula)
mm = formula_obj.get_model_matrix(data, na_action="drop")
y_name = mm.lhs.columns[0]
x_cols = [c for c in mm.rhs.columns if c != "Intercept"]

_df = data.loc[mm.rhs.index].copy()
_df["__y"] = mm.lhs[y_name].values.ravel()
for c in x_cols:
    _df["__x__" + c] = mm.rhs[c].values
ent_vals = _df[entity].values
time_vals = _df[time].values

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

exo_set = set(exogenous) if exogenous else set()
gmm_cols = [c for c in x_cols if c not in exo_set]
iv_cols = [c for c in x_cols if c in exo_set]

min_j = max(lags + 1, 2)
max_T = max(len(y_by_e[e]) for e in entities)
maxL = max_T - 1
depths = list(range(2, maxL + 1))
n_endog = 1 + len(gmm_cols)

# Build per-entity full Z and count
for e_val in entities[:1]:  # first entity only
    y = y_by_e[e_val]
    xs = x_by_e[e_val]
    T_i = len(y)
    n_gmm_i = n_endog * sum(T_i - d for d in depths if T_i > d)
    n_iv_i = len(iv_cols)
    n_instr_i = n_gmm_i + n_iv_i

    Z_i = np.zeros((T_i, n_instr_i))
    col = 0
    for d in depths:
        n_cols_d = T_i - d
        if n_cols_d <= 0:
            continue
        # L.y block
        blk = abond_mod._build_noncollapsed_gmm_block(y, d, T_i, lag_offset=lags)
        Z_i[:, col:col + n_cols_d] = blk
        col += n_cols_d
        for gmm_c in gmm_cols:
            blk = abond_mod._build_noncollapsed_gmm_block(xs[gmm_c], d, T_i, lag_offset=0)
            Z_i[:, col:col + n_cols_d] = blk
            col += n_cols_d
    for iv_c in iv_cols:
        for j in range(1, T_i):
            Z_i[j, col] = xs[iv_c][j] - xs[iv_c][j - 1]
        col += 1

    print(f"Entity {e_val}: T={T_i}")
    print(f"  n_gmm_i = {n_gmm_i}, n_iv_i = {n_iv_i}, total = {n_instr_i}")
    print(f"  Z_i shape = {Z_i.shape}")
    
    # Check for zero columns
    zero_cols = np.where(np.all(np.abs(Z_i) < 1e-15, axis=0))[0]
    print(f"  Zero columns (all-zero): {zero_cols}")
    
    # Check column rank
    _, s, _ = np.linalg.svd(Z_i[min_j:], full_matrices=False)
    print(f"  Singular values of usable slice [{min_j}:]:")
    for k in range(min(len(s), 10)):
        print(f"    [{k}] = {s[k]:.6e}")
    print(f"  Rank (count > 1e-10): {np.sum(s > 1e-10)}")
    print(f"  Columns: {Z_i.shape[1]}")
    
    # Build the estimation Z
    Z_list = []
    Y_list = []
    X_list = []
    eq_entity_list = []
    for j in range(min_j, T_i):
        dep = y[j] - y[j - 1]
        dyn_regs = [y[j - lag] - y[j - lag - 1] for lag in range(1, lags + 1)]
        x_regs = [xs[c][j] - xs[c][j - 1] for c in x_cols]
        X_list.append(dyn_regs + x_regs)
        Z_list.append(Z_i[j, :])
        Y_list.append(dep)
        eq_entity_list.append(e_val)
    
    Z_est = np.array(Z_list)
    X_est = np.array(X_list)
    print(f"\n  Estimation Z shape: {Z_est.shape}")
    print(f"  Estimation Z rank: {np.linalg.matrix_rank(Z_est)}")
    
    # Check ZtHZ
    n_eq_est = Z_est.shape[0]
    entity_counts = {e_val: n_eq_est}
    eq_entity_arr = np.array(eq_entity_list)
    H_diag, H_off = abond_mod._build_h(entity_counts, n_eq_est, eq_entity_arr)
    ZtHZ = 2.0 * (Z_est.T @ Z_est)
    ZH_off = Z_est[:-1] * H_off[:, None]
    ZtHZ += ZH_off.T @ Z_est[1:]
    ZtHZ += Z_est[1:].T @ ZH_off
    print(f"  ZtHZ shape: {ZtHZ.shape}")
    print(f"  ZtHZ rank: {np.linalg.matrix_rank(ZtHZ)}")
    try:
        np.linalg.inv(ZtHZ)
        print("  ZtHZ is invertible.")
    except np.linalg.LinAlgError:
        print("  ZtHZ is SINGULAR!")
        
        # Find linearly dependent columns
        U, s, Vt = np.linalg.svd(ZtHZ)
        null_mask = s < 1e-10
        print(f"  Zero singular values: {np.where(null_mask)[0]}")
        print(f"  Number of zero singular values: {np.sum(null_mask)}")
        
        # Check which columns of Z are linearly dependent
        U_z, s_z, Vt_z = np.linalg.svd(Z_est, full_matrices=False)
        null_mask_z = s_z < 1e-10
        print(f"  Z null singular values: {np.where(null_mask_z)[0]}")
        print(f"  Z rank deficiency: {np.sum(null_mask_z)}")
