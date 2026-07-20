import pandas as pd
import numpy as np

csv = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv")
entities = sorted(csv["entity"].unique())
Ye = {e: csv[csv["entity"]==e].sort_values("time")["y"].values for e in entities}
Xe = {e: csv[csv["entity"]==e].sort_values("time")["x"].values for e in entities}
Ze = {e: csv[csv["entity"]==e].sort_values("time")["z"].values for e in entities}

Zdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Z.csv")
Xdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_X.csv")
Ydf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Y.csv")
Z = Zdf[[c for c in Zdf.columns if c.startswith('Zmat')]].values
X = Xdf[[c for c in Xdf.columns if c.startswith('Xmat')]].values
Y = Ydf[[c for c in Ydf.columns if c.startswith('Ymat')]].values
A1 = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\A1.csv").values

print("=== Z COLUMN DEFINITIONS (from entity 0 data) ===")
e0 = csv[csv["entity"]==0].sort_values("time")
y0, x0, z0 = e0["y"].values, e0["x"].values, e0["z"].values
print(f"Entity 0: y={np.round(y0,4)} x={np.round(x0,4)} z={np.round(z0,4)}")

# Entity 0 rows in CSV: 0-4. Row 0=t0, 1=t1, 2=t2, 3=t3, 4=t4
# These are ALL diff rows (const=0)
# Z columns at diff rows:
for t in range(2, 5):
    print(f"  DIFF t={t}: Z[t]={np.round(Z[t], 5)}")
    print(f"    Expected: D.x={x0[t]-x0[t-1]:.4f}, D.z={z0[t]-z0[t-1]:.4f}")
    print(f"    y[t-2]={y0[t-2]:.4f}, y[t-3]={y0[t-3] if t-3>=0 else 0:.4f}, y[t-4]={y0[t-4] if t-4>=0 else 0:.4f}")

# Now check entity 0 level rows. They should be at rows 150-154.
# But we need to figure out which rows in the level block correspond to entity 0.
# Level block: rows 150-299 (150 rows = 30 entities x 5 periods)
# Entity 0 level rows should be at rows 150-154

print("\n=== ENTITY 0 LEVEL ROWS (rows 150-154) ===")
for i in range(150, 155):
    t = i - 150  # time index 0-4
    print(f"  row {i} (entity0 t={t}): X={np.round(X[i],4)} Y={Y[i,0]:.4f}")
    print(f"    Z={np.round(Z[i],5)}")
    if t >= 1:
        print(f"    Expected IV: x[{t}]={x0[t]:.4f}, z[{t}]={z0[t]:.4f}, const=1")
        print(f"    Expected GMM: D.L.y={y0[t-1]-y0[t-2] if t>=2 else 0:.4f}")
        if t >= 3:
            print(f"    Expected GMM2: DL.L.y={y0[t-2]-y0[t-3]:.4f}")

# Verify: one-step b
ZtX = Z.T @ X
ZtY = Z.T @ Y.flatten()
A1_inv = np.linalg.pinv(A1)
G = ZtX.T @ A1 @ ZtX
c = ZtX.T @ A1 @ ZtY
b1 = np.linalg.lstsq(G, c, rcond=None)[0].flatten()
print("\n=== ONE-STEP VERIFICATION ===")
print(f"Computed b: {np.round(b1, 6)}")
print("Stata 1s:   [0.110421, 1.156291, -0.603776, 0.061418]")

# Two-step
e1 = Y.flatten() - X @ b1
ents = Zdf['entity'].values
L = Z.shape[1]
S = np.zeros((L, L))
# Build per-entity moment contributions
# Need to figure out entity boundaries in the 300-row matrix
# From the data: entity 0 = rows 0-4 (diff) + rows 150-154 (level)?
# Or: entity 0 = rows 0-4 (diff) only, and level block rows 150-299 are separate?
# Let me just use ALL rows and group by some criterion
# Actually, the moment conditions for two-step use Z_i' e_i per entity
# Since the row ordering is unknown, let me just compute S naively (all rows together)
S = Z.T @ np.diag(e1**2) @ Z  # WRONG but diagnostic
try:
    A2 = np.linalg.pinv(S)
    G2 = ZtX.T @ A2 @ ZtX
    c2 = ZtX.T @ A2 @ ZtY
    b2 = np.linalg.lstsq(G2, c2, rcond=None)[0].flatten()
    print("\n=== TWO-STEP (naive S) ===")
    print(f"Computed b2: {np.round(b2, 6)}")
    print("Stata 2s:    [0.009464, 1.134976, -0.442064, 0.090758]")
except:
    print("Singular S")
