import pandas as pd
import numpy as np

csv = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv")
entities = sorted(csv["entity"].unique())

# Load Stata-exported matrices
Zdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Z.csv")
Xdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_X.csv")
Ydf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Y.csv")

Z = Zdf[[c for c in Zdf.columns if c.startswith('Zmat')]].values
X = Xdf[[c for c in Xdf.columns if c.startswith('Xmat')]].values
Y = Ydf[[c for c in Ydf.columns if c.startswith('Ymat')]].values
ents = Zdf['entity'].values
# Handle NaN entity (from svmat padding)
ents_clean = np.where(np.isnan(ents.astype(float)), -1, ents.astype(int))
times_vals = Zdf['time'].values

print("=== Stata e(Z), e(X), e(Y) ===")
print(f"Z: {Z.shape}, X: {X.shape}, Y: {Y.shape}")
print(f"Entities: {np.unique(ents)}, times: {np.unique(times_vals)}")

# Verify: can we reproduce b from these matrices?
# Need the weight matrix. A1 = (Z'HZ)^{-1} is known.
# For one-step: b = (X'Z A1 Z'X)^{-1} X'Z A1 Z'Y
A1 = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\A1.csv").values
print(f"\nA1 shape: {A1.shape}, rank: {np.linalg.matrix_rank(A1)}")

# One-step b
ZtX = Z.T @ X  # (11, 4)
ZtY = Z.T @ Y.flatten()  # (11,)
print(f"Z'X: {ZtX.shape}")
print(f"Z'Y: {ZtY.shape}")

A1_inv = np.linalg.pinv(A1)  # Z'HZ
print(f"A1_inv (Z'HZ) shape: {A1_inv.shape}")

# b = (ZtX' A1 ZtX)^{-1} ZtX' A1 ZtY
G = ZtX.T @ A1 @ ZtX
c = ZtX.T @ A1 @ ZtY
b = np.linalg.lstsq(G, c, rcond=None)[0].flatten()
print("\nOne-step b from Stata's e(Z) + e(A1):")
print(f"  b_Ly={b[0]:.6f} b_x={b[1]:.6f} b_z={b[2]:.6f} b_cons={b[3]:.6f}")
print("TARGET:  b_Ly=0.110421 b_x=1.156291 b_z=-0.603776 b_cons=0.061418")

# Now try: can we compute A1 from Z and H directly?
# Stata's e(H) was exported but only 10 columns (truncated?)
Hdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_H.csv")
hcols = [c for c in Hdf.columns if c.startswith('Hmat')]
print(f"\ne(H) from Stata: {len(hcols)} columns (expected 300 for 300x300)")
print("H was truncated by export. Need to reconstruct H from A1.")

# What we know: A1 = (Z'HZ)^{-1}, so Z'HZ = A1^{-1}
ZtHZ = np.linalg.pinv(A1)
print(f"\nZ'HZ (from A1^-1): {ZtHZ.shape}")

# Check rank
sv = np.linalg.svd(ZtHZ, compute_uv=False)
print(f"Z'HZ singular values: {np.round(sv, 4)}")
print(f"Z'HZ rank: {np.sum(sv > 1e-8)}")
# One zero singular value → rank 10, consistent with zrank=11 and 1 degenerate col

# Z col 10 is zero → this is the degenerate column
print(f"\nZ column 10 (zero col): all zero? {np.all(np.abs(Z[:,10]) < 1e-10)}")
print(f"Z column 9 (near zero?): norm = {np.linalg.norm(Z[:,9]):.8f}")

# Now: the key question is what the Z column definitions are
# Let me decode by examining each column against entity data
print("\n=== Z COLUMN DECODING (entity 1, non-zero rows) ===")
Ye = {e: csv[csv["entity"]==e].sort_values("time")["y"].values for e in entities}
Xe = {e: csv[csv["entity"]==e].sort_values("time")["x"].values for e in entities}
Ze = {e: csv[csv["entity"]==e].sort_values("time")["z"].values for e in entities}

e1_mask = ents_clean == 1
e1_rows = np.where(e1_mask)[0]
print(f"Entity 1 rows: {e1_rows.tolist()}")
print(f"Entity 1 times: {times_vals[e1_rows].tolist()}")
print(f"Entity 1 y: {np.round(Ye[1], 4)}")
print(f"Entity 1 x: {np.round(Xe[1], 4)}")
print(f"Entity 1 z: {np.round(Ze[1], 4)}")

for i in e1_rows:
    t = int(times_vals[i])
    eq = "DIFF" if X[i,3]==0 else "LEVEL"
    zstr = " ".join(f"{v:.4f}" for v in Z[i])
    print(f"  row {i} t={t} {eq}: Z=[{zstr}]")
