"""Deep diagnostic: compare Z'HZ and V for different H constructions.

Key insight: Stata's Zi has 5 rows (T=5) with GMM instruments zeroed
for non-usable rows, but IV instruments NOT zeroed. The H matrix is 5×5
with H=M'M where M has diagonal=2 for rows 1..4 (M[0,0]=0).

oe's Z has only 3 rows (usable equations t=2,3,4) with H diagonal=3.
This misses the off-diagonal H contributions involving row 1 (t=1).
"""
import sys
sys.path.insert(0, r"C:\Users\manhn\Desktop\open-econs")

import numpy as np
import pandas as pd
from collections import Counter
from formulaic import Formula

df = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv")

mm = Formula("y ~ x + z").get_model_matrix(df, na_action="drop")
y_name = mm.lhs.columns[0]
x_cols = [c for c in mm.rhs.columns if c != "Intercept"]

df2 = df.loc[mm.rhs.index].copy()
df2["__y"] = mm.lhs[y_name].values.ravel()
for c in x_cols:
    df2["__x__" + c] = mm.rhs[c].values
ent_vals = df2["entity"].values
time_vals = df2["time"].values

order = np.lexsort((time_vals, ent_vals))
ent_sorted = ent_vals[order]
y_sorted = df2["__y"].values[order]
x_sorted = {c: df2["__x__" + c].values[order] for c in x_cols}

entities = []
y_by_e = {}
x_by_e = {}
for e_val in pd.unique(ent_sorted):
    mask = ent_sorted == e_val
    entities.append(e_val)
    y_by_e[e_val] = y_sorted[mask]
    x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

exo_set = {"x", "z"}
gmm_cols = [c for c in x_cols if c not in exo_set]
iv_cols = [c for c in x_cols if c in exo_set]
lags = 1
min_j = max(lags + 1, 2)
T = 5
maxL = T - 1
depths = list(range(2, maxL + 1))
valid_depths = [d for d in depths if T - max(min_j, d) >= 2]
depths = valid_depths

n_endog = 1 + len(gmm_cols)
n_gmm = len(depths) * n_endog
n_iv = len(iv_cols)
n_instr = n_gmm + n_iv

print(f"T={T}, min_j={min_j}, depths={depths}")
print(f"n_gmm={n_gmm}, n_iv={n_iv}, n_instr={n_instr}")

# ── Build Stata-style Z (full T rows per entity, usable rows extracted) ──
# Stata: Zi is T×j, with GMM cols zeroed for non-usable rows,
#        IV cols present for ALL rows (including non-usable).
# Then Z'HZ = sum_i Zi' H Zi where H is T×T.

# For comparison, also build oe-style Z (only usable rows).

stata_V = np.array([
    [0.06085416, 0.04108388, -0.02125832],
    [0.04108388, 0.03142457, -0.01432837],
    [-0.02125832, -0.01432837, 0.01086979],
])
stata_b = np.array([-0.11984163, 1.1258209, -0.28974145])
stata_se = np.sqrt(np.diag(stata_V))

# ── Build M and H for T=5 ────────────────────────────────────────────────
# _xform(xform=0, T=5): M = I(5) - lag(I(5), -1), M[0,0] = 0
M = np.eye(T)
M_lag = np.eye(T)
# _lag(M, -1): shift columns left by 1 (column j becomes column j-1)
M_lag_shifted = np.zeros((T, T))
M_lag_shifted[:, :-1] = M_lag[:, 1:]  # shift left
M = M - M_lag_shifted
M[0, 0] = 0.0

print("\nM (first-difference operator):")
print(M)

H = M.T @ M
print("\nH = M'M:")
print(H)
print(f"H diagonal: {np.diag(H)}")

# H_sub = H[min_j:, min_j:] = H[2:, 2:] for usable equations
H_sub = H[min_j:, min_j:]
print(f"\nH_sub = H[{min_j}:, {min_j}:] (usable equations):")
print(H_sub)
print(f"H_sub diagonal: {np.diag(H_sub)}")


def build_ZtHZ_stata_style(entities, y_by_e, x_by_e, depths, min_j, T,
                            n_endog, n_gmm, n_iv, H, exo_set, gmm_cols, iv_cols):
    """Build Z'HZ using Stata's approach: Zi is T×j, H is T×T."""
    n_total = n_gmm + n_iv
    ZtHZ = np.zeros((n_total, n_total))
    ZtX = np.zeros((n_total, len(x_cols) + 1))  # +1 for L.y
    ZtY = np.zeros(n_total)
    Y_all = []
    X_all = []
    Z_usable = []
    eq_ent = []

    x_col_names = ["L1.y"] + x_cols
    p = len(x_col_names)

    for e_val in entities:
        y = y_by_e[e_val]
        xs = x_by_e[e_val]

        # Build Zi as T×j
        Zi = np.zeros((T, n_total))

        # IV instruments for ALL time periods
        for t in range(T):
            col = n_gmm
            for iv_c in iv_cols:
                if t > 0:
                    Zi[t, col] = xs[iv_c][t] - xs[iv_c][t - 1]
                col += 1

        # GMM instruments for usable equations only
        for t in range(min_j, T):
            col = 0
            for lag in depths:
                if t - lag >= 0:
                    Zi[t, col] = y[t - lag]
                col += 1
            for gmm_c in gmm_cols:
                for lag in depths:
                    if t - lag >= 0:
                        Zi[t, col] = xs[gmm_c][t - lag]
                    col += 1

        # Accumulate Z'HZ = Zi' H Zi
        HZ = H @ Zi  # (T, j)
        ZtHZ += Zi.T @ HZ  # (j, j)

        # Accumulate Z'X and Z'Y for usable equations
        Xi = np.zeros((T, p))
        Yi = np.zeros(T)
        for t in range(min_j, T):
            dep = y[t] - y[t - 1]
            Yi[t] = dep
            Xi[t, 0] = y[t - 1] - y[t - 2]  # L.y
            for k, c in enumerate(x_cols):
                Xi[t, k + 1] = xs[c][t] - xs[c][t - 1]

        ZtX += Zi.T @ Xi  # (j, p)
        ZtY += Zi.T @ Yi  # (j,)

        # Collect usable rows of Z for residual computation
        for t in range(min_j, T):
            Z_usable.append(Zi[t, :])
            X_all.append(Xi[t, :])
            Y_all.append(Yi[t])
            eq_ent.append(e_val)

    Z_usable = np.array(Z_usable)
    X_all = np.array(X_all)
    Y_all = np.array(Y_all)
    eq_ent = np.array(eq_ent)

    return ZtHZ, ZtX, ZtY, Z_usable, X_all, Y_all, eq_ent


def solve_gmm(ZtHZ, ZtX, ZtY, Y, X, eq_entity, p):
    """Solve GMM and return b, V, sig2."""
    W = np.linalg.pinv(ZtHZ)
    G = ZtX.T @ W @ ZtX
    G_inv = np.linalg.inv(G)
    b = G_inv @ (ZtX.T @ W @ ZtY)
    e = Y - X @ b
    n_eq = len(Y)
    h_factor = 2.0
    df = float(n_eq - p)
    sig2 = float(e @ e) / (h_factor * df)
    V = sig2 * G_inv
    return b, V, sig2


# ── Test 1: Stata-style H (M'M, diagonal=2) with Stata-style Z ──────────
ZtHZ_1, ZtX_1, ZtY_1, Z_us_1, X_1, Y_1, eq_1 = build_ZtHZ_stata_style(
    entities, y_by_e, x_by_e, depths, min_j, T, n_endog, n_gmm, n_iv,
    H, exo_set, gmm_cols, iv_cols
)
b1, V1, sig2_1 = solve_gmm(ZtHZ_1, ZtX_1, ZtY_1, Y_1, X_1, eq_1, X_1.shape[1])

print("\n" + "=" * 72)
print("TEST 1: Stata-style H (M'M, diag=2) + Stata-style Z")
print("=" * 72)
print(f"b = {b1}")
print(f"se = {np.sqrt(np.diag(V1))}")
print(f"sig2 = {sig2_1}")
print("V:")
for row in V1:
    print(f"  [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}]")

# ── Test 2: oe-style Z with Stata H (diag=2) ─────────────────────────────
# oe's Z only has usable rows. Need to build Z'HZ using H_sub = H[min_j:, min_j:]
Z_oe_list = []
X_oe_list = []
Y_oe_list = []
eq_oe_list = []

for e_val in entities:
    y = y_by_e[e_val]
    xs = x_by_e[e_val]
    for j in range(min_j, T):
        dep = y[j] - y[j - 1]
        dyn_regs = [y[j - lag] - y[j - lag - 1] for lag in range(1, lags + 1)]
        x_regs = [xs[c][j] - xs[c][j - 1] for c in x_cols]
        X_oe_list.append(dyn_regs + x_regs)

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

        Z_oe_list.append(zrow)
        Y_oe_list.append(dep)
        eq_oe_list.append(e_val)

Z_oe = np.array(Z_oe_list, dtype=float)
X_oe = np.array(X_oe_list, dtype=float)
Y_oe = np.array(Y_oe_list, dtype=float)
eq_oe = np.array(eq_oe_list)

# Build Z'HZ using H_sub (the sub-matrix for usable equations)
entity_counts = dict(Counter(eq_oe.tolist()))

def build_ZtHZ_oe_style(Z, eq_entity, H_sub):
    """Build Z'HZ using oe's Z (usable rows only) and H_sub."""
    L = Z.shape[1]
    ZtHZ = np.zeros((L, L))
    pos = 0
    for ent, n_i in entity_counts.items():
        Zi = Z[pos:pos + n_i]  # (n_i, L)
        HZ = H_sub @ Zi  # (n_i, L) — H_sub is n_i × n_i
        ZtHZ += Zi.T @ HZ  # (L, L)
        pos += n_i
    return ZtHZ


ZtHZ_2 = build_ZtHZ_oe_style(Z_oe, eq_oe, H_sub)
ZtX_2 = Z_oe.T @ X_oe
ZtY_2 = Z_oe.T @ Y_oe
b2, V2, sig2_2 = solve_gmm(ZtHZ_2, ZtX_2, ZtY_2, Y_oe, X_oe, eq_oe, X_oe.shape[1])

print("\n" + "=" * 72)
print("TEST 2: oe-style Z (usable only) + H_sub (diag=2)")
print("=" * 72)
print(f"b = {b2}")
print(f"se = {np.sqrt(np.diag(V2))}")
print(f"sig2 = {sig2_2}")
print("V:")
for row in V2:
    print(f"  [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}]")

# ── Test 3: oe's current approach (diag=3) ───────────────────────────────
def build_ZtHZ_oe_current(Z, eq_entity, h_diag):
    n_eq = Z.shape[0]
    H_off = np.full(n_eq - 1, -1.0)
    ent_arr = np.asarray(eq_entity)
    for k in range(n_eq - 1):
        if ent_arr[k] != ent_arr[k + 1]:
            H_off[k] = 0.0
    ZtHZ = h_diag * (Z.T @ Z)
    ZH_off = Z[:-1] * H_off[:, None]
    ZtHZ += ZH_off.T @ Z[1:]
    ZtHZ += Z[1:].T @ ZH_off
    return ZtHZ


ZtHZ_3 = build_ZtHZ_oe_current(Z_oe, eq_oe, 3.0)
b3, V3, sig2_3 = solve_gmm(ZtHZ_3, ZtX_2, ZtY_2, Y_oe, X_oe, eq_oe, X_oe.shape[1])

print("\n" + "=" * 72)
print("TEST 3: oe current (diag=3)")
print("=" * 72)
print(f"b = {b3}")
print(f"se = {np.sqrt(np.diag(V3))}")
print(f"sig2 = {sig2_3}")

# ── Test 4: What if the H diagonal should be LARGER? ──────────────────────
# The Stata Z'HZ has the IV instrument contribution from non-usable rows.
# Let's check what Z'HZ[iv,iv] looks like in each case.
print("\n" + "=" * 72)
print("Z'HZ STRUCTURE COMPARISON")
print("=" * 72)
print("\nZ'HZ (Stata-style, full 5-row Zi):")
print("  GMM×GMM block:")
print(f"    {ZtHZ_1[:n_gmm, :n_gmm]}")
print("  IV×IV block:")
print(f"    {ZtHZ_1[n_gmm:, n_gmm:]}")
print("  GMM×IV block:")
print(f"    {ZtHZ_1[:n_gmm, n_gmm:]}")

print("\nZ'HZ (oe-style, H_sub diag=2):")
print("  GMM×GMM block:")
print(f"    {ZtHZ_2[:n_gmm, :n_gmm]}")
print("  IV×IV block:")
print(f"    {ZtHZ_2[n_gmm:, n_gmm:]}")

print("\nZ'HZ (oe current, diag=3):")
print("  GMM×GMM block:")
print(f"    {ZtHZ_3[:n_gmm, :n_gmm]}")
print("  IV×IV block:")
print(f"    {ZtHZ_3[n_gmm:, n_gmm:]}")

print("\nDifference (Stata-style - oe diag=2):")
print(f"  GMM×GMM: {np.max(np.abs(ZtHZ_1[:n_gmm, :n_gmm] - ZtHZ_2[:n_gmm, :n_gmm])):.6e}")
print(f"  IV×IV:   {np.max(np.abs(ZtHZ_1[n_gmm:, n_gmm:] - ZtHZ_2[n_gmm:, n_gmm:])):.6e}")
print(f"  GMM×IV:  {np.max(np.abs(ZtHZ_1[:n_gmm, n_gmm:] - ZtHZ_2[:n_gmm, n_gmm:])):.6e}")

print("\nDifference (Stata-style - oe diag=3):")
print(f"  GMM×GMM: {np.max(np.abs(ZtHZ_1[:n_gmm, :n_gmm] - ZtHZ_3[:n_gmm, :n_gmm])):.6e}")
print(f"  IV×IV:   {np.max(np.abs(ZtHZ_1[n_gmm:, n_gmm:] - ZtHZ_3[n_gmm:, n_gmm:])):.6e}")
print(f"  GMM×IV:  {np.max(np.abs(ZtHZ_1[:n_gmm, n_gmm:] - ZtHZ_3[:n_gmm, n_gmm:])):.6e}")

# ── Full V comparison ──────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("FULL V-MATRIX COMPARISON vs STATA")
print("=" * 72)
labels = ["L1.y", "x", "z"]
print(f"\n{'Element':<12} {'Stata':>12} {'T1_full':>12} {'T2_sub':>12} {'T3_cur':>12}")
print("-" * 64)

for i in range(3):
    for j in range(3):
        s = stata_V[i, j]
        v1 = V1[i, j]
        v2 = V2[i, j]
        v3 = V3[i, j]
        print(f"V[{labels[i]},{labels[j]:<4}] {s:12.8f} {v1:12.8f} {v2:12.8f} {v3:12.8f}")

print(f"\n{'SE':<12} {'Stata':>12} {'T1_full':>12} {'T2_sub':>12} {'T3_cur':>12}")
print("-" * 64)
for i in range(3):
    s = stata_se[i]
    se1 = np.sqrt(V1[i, i])
    se2 = np.sqrt(V2[i, i])
    se3 = np.sqrt(V3[i, i])
    print(f"se_{labels[i]:<8} {s:12.8f} {se1:12.8f} {se2:12.8f} {se3:12.8f}")

print(f"\n{'Coef':<12} {'Stata':>12} {'T1_full':>12} {'T2_sub':>12} {'T3_cur':>12}")
print("-" * 64)
for i in range(3):
    s = stata_b[i]
    print(f"b_{labels[i]:<9} {s:12.8f} {b1[i]:12.8f} {b2[i]:12.8f} {b3[i]:12.8f}")

print(f"\n{'sig2':<12} {'Stata':>12} {'T1_full':>12} {'T2_sub':>12} {'T3_cur':>12}")
print("-" * 64)
print(f"{'':12} {0.19753252:12.8f} {sig2_1:12.8f} {sig2_2:12.8f} {sig2_3:12.8f}")
