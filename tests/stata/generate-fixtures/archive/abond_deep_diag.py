"""Deep diagnostic: Z matrix structure, per-entity g_i, sandwich dimensions."""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from open_econs.models.linear.abond import abond as _abond, _estimate_gmm
from formulaic import Formula

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

SEP = "=" * 72

# =====================================================================
# PART 1: Z matrix structure for collapsed and non-collapsed
# =====================================================================
print(SEP)
print("PART 1: Z matrix structure")
print(SEP)

for collapse_flag in [True, False]:
    label = "COLLAPSED" if collapse_flag else "NONCOLLAPSED"
    r = _abond(
        "y ~ x + z", data=df, entity="entity", time="time",
        step="one-step", lags=1, max_iv_lag=4, collapse=collapse_flag,
        exogenous=["x", "z"],
    )
    # Access internals by re-running
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

    entities_list = []
    y_by_e = {}
    x_by_e = {}
    for e_val in pd.unique(ent_sorted):
        mask = ent_sorted == e_val
        entities_list.append(e_val)
        y_by_e[e_val] = y_sorted[mask]
        x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

    depths = [2, 3, 4]
    min_j = 2
    iv_cols = ["x", "z"]
    n_endog = 1  # only L.y is GMM-endogenous

    n_instr = len(depths) * n_endog + len(iv_cols)

    # Build Z for first entity
    e0 = entities_list[0]
    y0 = y_by_e[e0]
    xs0 = x_by_e[e0]
    T0 = len(y0)

    print(f"\n--- {label} ---")
    print(f"  n_instr formula: {len(depths)} depths * {n_endog} GMM + {len(iv_cols)} std = {n_instr}")
    print(f"  Z shape: ({r.n_obs}, {r.n_instruments})")
    print(f"  Entity 0 (T={T0}), equations at t={min_j}..{T0-1}:")

    for j in range(min_j, T0):
        zrow = np.zeros(n_instr)
        col = 0
        for lag in depths:
            if j - lag >= 0:
                zrow[col] = y0[j - lag]
            col += 1
        for iv_c in iv_cols:
            zrow[col] = xs0[iv_c][j] - xs0[iv_c][j - 1]
            col += 1
        print(f"    t={j}: Z=[{', '.join(f'{v:.4f}' for v in zrow)}]")

# =====================================================================
# PART 2: Per-entity g_i dimensions and sandwich matrix dims
# =====================================================================
print(f"\n{SEP}")
print("PART 2: Sandwich VCV dimensions and per-entity g_i")
print(SEP)

# Run non-collapsed to match Stata baseline
r = _abond(
    "y ~ x + z", data=df, entity="entity", time="time",
    step="one-step", lags=1, max_iv_lag=4, collapse=False,
    exogenous=["x", "z"],
)

# Rebuild internals to inspect sandwich components
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

entities_list = []
y_by_e = {}
x_by_e = {}
for e_val in pd.unique(ent_sorted):
    mask = ent_sorted == e_val
    entities_list.append(e_val)
    y_by_e[e_val] = y_sorted[mask]
    x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

depths = [2, 3, 4]
min_j = 2
iv_cols = ["x", "z"]
n_instr = len(depths) * 1 + len(iv_cols)  # 5

Y_list, X_list, Z_list, eq_entity_list = [], [], [], []
for e_val in entities_list:
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
        for iv_c in iv_cols:
            zrow[col] = xs[iv_c][j] - xs[iv_c][j - 1]
            col += 1

        Z_list.append(zrow)
        Y_list.append(dep)
        eq_entity_list.append(e_val)

Y = np.array(Y_list, dtype=float)
X = np.array(X_list, dtype=float)
Z = np.array(Z_list, dtype=float)
eq_ent = np.array(eq_entity_list)

print(f"  Y shape: {Y.shape}  (n_obs_equations,)")
print(f"  X shape: {X.shape}  (n_obs_equations, p)")
print(f"  Z shape: {Z.shape}  (n_obs_equations, L)")
print(f"  N entities: {len(entities_list)}")
print(f"  Equations per entity: {Y.shape[0] / len(entities_list):.1f}")
print(f"  Total equations: {Y.shape[0]}")

# Compute sandwich components step by step
L = Z.shape[1]
p = X.shape[1]
N = float(len(entities_list))

ZtZ = Z.T @ Z
W = np.linalg.pinv(ZtZ)
ZtX = Z.T @ X
ZtY = Z.T @ Y

G = ZtX.T @ W @ ZtX
G_inv = np.linalg.inv(G)
b = G_inv @ (ZtX.T @ W @ ZtY)
e = Y - X @ b

print(f"\n  G (cross-product): shape={G.shape}")
print(f"    G = Z'X W X'Z")
print(f"    G_inv: shape={G_inv.shape}")

# Per-entity g_i
print(f"\n  Per-entity moment vectors g_i:")
S_g = np.zeros((p, p))
for i, ent in enumerate(entities_list):
    mask = eq_ent == ent
    Zc = Z[mask]
    Xc = X[mask]
    ec = e[mask]
    n_eq_ent = mask.sum()

    Zte = Zc.T @ ec
    XtZ = Xc.T @ Zc
    gi = XtZ @ W @ Zte

    print(f"    Entity {ent}: Zc={Zc.shape}, Xc={Xc.shape}, ec={ec.shape}, "
          f"g_i={gi.shape}  (sum|g_i|={np.sum(np.abs(gi)):.6f})")
    S_g += np.outer(gi, gi)

print(f"\n  S_g (sandwich meat): shape={S_g.shape}")
print(f"    S_g = Sum_i g_i g_i'  (sum over {len(entities_list)} entities)")
print(f"    Note: g_i has shape (p,) = ({p},), outer product gives ({p},{p})")

V_sandwich = G_inv @ S_g @ G_inv
print(f"\n  V_sandwich = G^-1 S_g G^-1: shape={V_sandwich.shape}")
print(f"    diag(V_sandwich) = {np.diag(V_sandwich)}")
print(f"    se from V_sandwich = {np.sqrt(np.maximum(np.diag(V_sandwich), 0))}")

# Compare with oe output
print(f"\n  oe output std_errors = {r.std_errors.values}")
print(f"  Ratio oe/se_sandwich = {r.std_errors.values / np.sqrt(np.maximum(np.diag(V_sandwich), 0))}")

# Stata reference
print(f"\n  Stata 1-step non-collapsed SE = [0.24521319, 0.17680462, 0.10368879]")
print(f"  Stata VCV diag = [0.06012951, 0.03125987, 0.01075136]")

# Check: is S_g being summed over N_obs instead of N_entities?
print(f"\n  S_g trace = {np.trace(S_g):.10f}")
print(f"  Stata VCV diag sums to = {0.06012951 + 0.03125987 + 0.01075136:.10f}")
