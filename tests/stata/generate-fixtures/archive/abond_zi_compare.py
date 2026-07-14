"""Compare per-entity Zi matrices between oe and Stata style.

Stata builds Zi as T×j (5 rows for T=5), with:
  - Rows for non-usable equations: GMM cols zeroed, IV cols have Δx_t, Δz_t
  - Rows for usable equations: all cols present

oe builds Z as (T-min_j)×j (3 rows), only usable equations.

Also: dump the first entity's Zi in both formats and show Z'Z and Z'HZ.
"""
import sys
sys.path.insert(0, r"C:\Users\manhn\Desktop\open-econs")

import numpy as np
import pandas as pd
from formulaic import Formula

df = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\df_panel.csv")
mm = Formula("y ~ x + z").get_model_matrix(df, na_action="drop")
y_name = mm.lhs.columns[0]
x_cols = [c for c in mm.rhs.columns if c != "Intercept"]

df2 = df.loc[mm.rhs.index].copy()
df2["__y"] = mm.lhs[y_name].values.ravel()
for c in x_cols:
    df2["__x__" + c] = mm.rhs[c].values

# Entity 0 data
e0 = df2[df2["entity"] == 0].sort_values("time")
print("Entity 0 data:")
print(e0[["y", "x", "z", "time"]].to_string())

y = e0["__y"].values
x = e0["__x__x"].values
z = e0["__x__z"].values
T = 5
min_j = 2
depths = [2, 3]
lags = 1

# ── oe-style Z for entity 0 (3 usable rows) ──────────────────────────────
print("\n" + "=" * 72)
print("oe-style Z for entity 0 (rows for t=2,3,4)")
print("=" * 72)
# Columns: GMM for L.y (depth 2, 3), GMM for x (depth 2, 3), IV for x, IV for z
cols = ["GMM:Ly_d2", "GMM:Ly_d3", "GMM:x_d2", "GMM:x_d3", "IV:dx", "IV:dz"]
n_gmm = len(depths) * 2  # L.y + x, each with 2 depths
n_iv = 2  # x, z
n_total = n_gmm + n_iv

print(f"Column order: {cols}")
Z_oe = np.zeros((T - min_j, n_total))
for j_idx, j in enumerate(range(min_j, T)):
    col = 0
    # GMM for L.y
    for lag in depths:
        if j - lag >= 0:
            Z_oe[j_idx, col] = y[j - lag]
        col += 1
    # GMM for x
    for lag in depths:
        if j - lag >= 0:
            Z_oe[j_idx, col] = x[j - lag]
        col += 1
    # IV for x, z (current Δ)
    Z_oe[j_idx, col] = x[j] - x[j - 1]
    col += 1
    Z_oe[j_idx, col] = z[j] - z[j - 1]

for i in range(3):
    t = i + min_j
    print(f"  t={t}: {Z_oe[i]}")

# ── Stata-style Zi for entity 0 (5 rows, including non-usable) ────────────
print("\n" + "=" * 72)
print("Stata-style Zi for entity 0 (5 rows, t=0..4)")
print("=" * 72)
Zi_stata = np.zeros((T, n_total))
# IV instruments for ALL time periods
for t in range(T):
    col = n_gmm  # IV cols start after GMM
    if t > 0:
        Zi_stata[t, col] = x[t] - x[t - 1]
    col += 1
    if t > 0:
        Zi_stata[t, col] = z[t] - z[t - 1]

# GMM instruments for usable equations only
for t in range(min_j, T):
    col = 0
    for lag in depths:
        if t - lag >= 0:
            Zi_stata[t, col] = y[t - lag]
        col += 1
    for lag in depths:
        if t - lag >= 0:
            Zi_stata[t, col] = x[t - lag]
        col += 1

for t in range(T):
    print(f"  t={t}: {Zi_stata[t]}")

# ── Show the difference ──────────────────────────────────────────────────
print("\n" + "=" * 72)
print("DIFFERENCE (Stata rows 2,3,4 - oe rows 0,1,2)")
print("=" * 72)
diff = Zi_stata[min_j:] - Z_oe
print(f"Max abs diff: {np.max(np.abs(diff)):.2e}")
print(f"Diff:\n{diff}")

# ── Check: what are the Δx values for t=0,1? ─────────────────────────────
print("\n" + "=" * 72)
print("IV instruments for non-usable rows (t=0,1)")
print("=" * 72)
for t in [0, 1]:
    dx = x[t] - x[t - 1] if t > 0 else 0
    dz = z[t] - z[t - 1] if t > 0 else 0
    print(f"  t={t}: dx={dx:.6f}, dz={dz:.6f}")

# ── Show H for T=5 ───────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("H matrix for T=5 (M'M)")
print("=" * 72)
M = np.eye(T)
M_lag_shifted = np.zeros((T, T))
M_lag_shifted[:, :-1] = np.eye(T)[:, 1:]
M = M - M_lag_shifted
M[0, 0] = 0.0
H = M.T @ M
print(f"H:\n{H}")
print(f"H diagonal: {np.diag(H)}")

# ── Compute Zi' H Zi for entity 0 (Stata style) ──────────────────────────
print("\n" + "=" * 72)
print("Entity 0: Zi' H Zi (Stata style, 5×5 H)")
print("=" * 72)
HZ = H @ Zi_stata
ZtHZ_e0 = Zi_stata.T @ HZ
print(f"Zi' H Zi (4×4):\n{ZtHZ_e0}")

# ── Compare with oe: Z0' H_sub Z0 ────────────────────────────────────────
print("\n" + "=" * 72)
print("Entity 0: Z0' H_sub Z0 (oe style, 3×3 H_sub)")
print("=" * 72)
H_sub = H[min_j:, min_j:]
HZ_sub = H_sub @ Z_oe.T  # (3, 3) @ (3, 4) = (3, 4)
ZtHZ_e0_oe = Z_oe @ H_sub @ Z_oe.T  # (3, 4) @ (3, 3) @ (4, 3) -- wrong shape

# Actually: Z_oe is (3, 4), H_sub is (3, 3)
# Z_oe.T @ H_sub @ Z_oe = (4, 3) @ (3, 3) @ (3, 4) = (4, 4)
ZtHZ_e0_oe = Z_oe.T @ H_sub @ Z_oe
print(f"Z0' H_sub Z0 (4×4):\n{ZtHZ_e0_oe}")

print(f"\nDifference (Stata - oe):\n{ZtHZ_e0 - ZtHZ_e0_oe}")

# ── Now the key question: what does Stata's accum loop actually compute? ──
# S = S + quadcross(Zi, _wt, quadcross(H, _wt, Zi))
# For unweighted (_wt=1): S += Zi' H Zi
# But Zi is RowsPerGroup × j where RowsPerGroup includes ALL time periods
# And H is RowsPerGroup × RowsPerGroup

# But wait -- for the IV columns, does Stata have one IV column per time period
# (like Stata's Z_IV), or one IV column total (like oe)?

# In Stata, Z_IV for exogenous regressors has one column per regressor
# placed at the equation's time period. So Z_IV[t, col] = Δx_t.
# This means the IV column for x has Δx_0, Δx_1, Δx_2, Δx_3, Δx_4
# in rows 0-4 of Zi.

# In oe, the IV column for x has Δx_2, Δx_3, Δx_4 in rows 0-2 of Z.

# The key difference: Stata's IV column has 5 values (including Δx_0=0, Δx_1),
# while oe's IV column has 3 values (Δx_2, Δx_3, Δx_4).

# The H matrix connects consecutive rows. So H[1,2] = -1 connects
# the IV value at t=1 (Δx_1) with the IV value at t=2 (Δx_2).
# This cross-term contributes to Z'HZ.

# But oe's Z only has rows for t=2,3,4. So oe misses the cross-term
# between t=1 and t=2.

# Let me check: what is Δx_1 for entity 0?
print("\n" + "=" * 72)
print("IV cross-term analysis")
print("=" * 72)
print(f"Δx_1 = {x[1] - x[0]:.6f}")
print(f"Δx_2 = {x[2] - x[1]:.6f}")
print(f"H[1,2] = {H[1,2]:.1f}")
print(f"Cross-term H[1,2] * dx_1 * dx_2 = {H[1,2] * (x[1]-x[0]) * (x[2]-x[1]):.6f}")

# In oe, this term is missing because there's no row for t=1.
# In Stata, this term is present because Zi has a row for t=1.

# But wait, in Stata, the IV columns are:
# Zi[t, iv_col] = Δx_t for ALL t (including non-usable)
# And the GMM columns are zeroed for non-usable t.

# So the IV×IV block of Z'HZ includes contributions from all time periods,
# while the GMM×GMM block only includes contributions from usable periods.

# In oe, both blocks only include contributions from usable periods.

# This explains why the IV×IV block is different between Test 1 and Test 2.
# But the GMM×GMM block is the same.

# The question is: does this explain the 15× gap in V?

# Let me check by computing V with the Stata-style Z'HZ
# (including IV contributions from non-usable rows)
# but using the same ZtX and ZtY as oe (only usable rows).

# Actually, I already did this in Test 1 of the previous script.
# And Test 1 gave V[L1.y,L1.y] = 0.00394, which is still 15× too small.

# So the IV cross-terms don't explain the gap.

# Let me check if the Z matrix values are correct.
# Maybe the GMM instruments are wrong.

# For entity 0, t=2, lag=2:
# oe: Z_oe[0, 0] = y[2-2] = y[0] = 3.243281
# Stata: Zi_stata[2, 0] = y[2-2] = y[0] = 3.243281

# For entity 0, t=2, lag=3:
# oe: Z_oe[0, 1] = y[2-3] = y[-1] (not available, so 0)
# Stata: Zi_stata[2, 1] = y[2-3] = y[-1] (not available, so 0)

# These match. So the GMM instruments are correct.

# Let me check the IV instruments.
# For entity 0, t=2:
# oe: Z_oe[0, 4] = x[2] - x[1] = -0.274138 - 0.298746 = -0.572884
# Stata: Zi_stata[2, 4] = x[2] - x[1] = -0.572884

# These match too.

# So the Z matrix is correct. The issue must be elsewhere.

# Let me check the Z'Z matrix to see the instrument magnitudes.
print("\n" + "=" * 72)
print("Z'Z matrix (oe style, all 90 equations)")
print("=" * 72)

# Rebuild full Z for all entities
from collections import Counter
entities = pd.unique(df2["entity"].values)
y_by_e = {}
x_by_e = {}
ent_vals = df2["entity"].values
time_vals = df2["time"].values
order = np.lexsort((time_vals, ent_vals))
y_sorted = df2["__y"].values[order]
x_sorted = {c: df2["__x__" + c].values[order] for c in x_cols}
for e_val in entities:
    mask = ent_vals[order] == e_val
    y_by_e[e_val] = y_sorted[mask]
    x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

exo_set = {"x", "z"}
gmm_cols = [c for c in x_cols if c not in exo_set]
iv_cols = [c for c in x_cols if c in exo_set]

n_endog = 1 + len(gmm_cols)
n_gmm = len(depths) * n_endog
n_iv = len(iv_cols)
n_total = n_gmm + n_iv

Z_full = []
for e_val in entities:
    y_e = y_by_e[e_val]
    xs_e = x_by_e[e_val]
    for j in range(min_j, T):
        zrow = np.zeros(n_total)
        col = 0
        for lag in depths:
            if j - lag >= 0:
                zrow[col] = y_e[j - lag]
            col += 1
        for gmm_c in gmm_cols:
            for lag in depths:
                if j - lag >= 0:
                    zrow[col] = xs_e[gmm_c][j - lag]
                col += 1
        for iv_c in iv_cols:
            zrow[col] = xs_e[iv_c][j] - xs_e[iv_c][j - 1]
            col += 1
        Z_full.append(zrow)

Z_full = np.array(Z_full)
print(f"Z shape: {Z_full.shape}")
ZtZ = Z_full.T @ Z_full
print(f"Z'Z:\n{ZtZ}")
print(f"Z'Z diagonal: {np.diag(ZtZ)}")
