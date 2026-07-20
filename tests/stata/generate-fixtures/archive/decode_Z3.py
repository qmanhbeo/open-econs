import pandas as pd, numpy as np

csv = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv")
entities = sorted(csv["entity"].unique())

# Entity-keyed data
Ye = {e: csv[csv["entity"]==e].sort_values("time")["y"].values for e in entities}
Xe = {e: csv[csv["entity"]==e].sort_values("time")["x"].values for e in entities}
Ze = {e: csv[csv["entity"]==e].sort_values("time")["z"].values for e in entities}

# Stata matrices
Zdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Z.csv")
Xdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_X.csv")
Ydf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Y.csv")
Z = Zdf[[c for c in Zdf.columns if c.startswith('Zmat')]].values
X = Xdf[[c for c in Xdf.columns if c.startswith('Xmat')]].values
Y = Ydf[[c for c in Ydf.columns if c.startswith('Ymat')]].values
ents = Zdf['entity'].values
times_vals = Zdf['time'].values

print("=== ROW ORDERING DIAGNOSIS ===")
# The CSV has 300 rows. Entity column has values 0-29 (each appearing 5 times) + NaN.
# Let me check: are rows 0-149 = DIFF block and rows 150-299 = LEVEL block?

# Check: for entity 1 (rows 5-9 in CSV), what are the X values?
e1_mask = np.array([e == 1 for e in ents])  # use float comparison
print(f"Entity 1 rows in CSV: {np.where(e1_mask)[0].tolist()}")
print(f"Entity 1 times: {times_vals[e1_mask].tolist()}")
print(f"Entity 1 X:\n{np.round(X[e1_mask], 4)}")
print(f"Entity 1 Y: {np.round(Y[e1_mask].flatten(), 4)}")
print(f"Entity 1 Z col norms per row:")
for i, row_idx in enumerate(np.where(e1_mask)[0]):
    print(f"  row {row_idx} t={times_vals[row_idx]:.0f}: Z norms per col = {np.round(np.abs(Z[row_idx]), 4)}")

# Now check entity 0 (rows 0-4)
e0_mask = np.array([e == 0 for e in ents])
print(f"\nEntity 0 X:\n{np.round(X[e0_mask], 4)}")
print(f"Entity 0 Y: {np.round(Y[e0_mask].flatten(), 4)}")

# Check the LEVEL block (rows 150-299)
print(f"\n=== LEVEL BLOCK (rows 150-154) ===")
print(f"Entity col (should be NaN): {ents[150:155]}")
print(f"Times: {times_vals[150:155]}")
print(f"X:\n{np.round(X[150:155], 4)}")
print(f"Y: {np.round(Y[150:155].flatten(), 4)}")
print(f"Z:\n{np.round(Z[150:155], 4)}")

# Cross-check: entity 0 y values
e0 = csv[csv["entity"]==0].sort_values("time")
print(f"\nEntity 0 raw data:")
print(f"  y: {np.round(e0['y'].values, 4)}")
print(f"  x: {np.round(e0['x'].values, 4)}")
print(f"  z: {np.round(e0['z'].values, 4)}")

# Diff eq at t=1: Δy_1 = y_1 - y_0, L.y_1 = y_0, Dx_1 = x_1-x_0, Dz_1 = z_1-z_0
y0 = e0['y'].values
x0 = e0['x'].values
z0 = e0['z'].values
print(f"\nEntity 0 diff eq expected (t=1):")
print(f"  Dy = y[1]-y[0] = {y0[1]-y0[0]:.4f}")
print(f"  L.y = y[0] = {y0[0]:.4f}")
print(f"  Dx = x[1]-x[0] = {x0[1]-x0[0]:.4f}")
print(f"  Dz = z[1]-z[0] = {z0[1]-z0[0]:.4f}")

# Check: X row for entity 0 at t=1
# Entity 0 is rows 0-4. Row 1 is t=1.
print(f"\nEntity 0 row 1 (t=1): X = {np.round(X[1], 4)}, Y = {Y[1,0]:.4f}")

# Also check entity 1 level block
print(f"\n=== Entity 1 LEVEL (rows 155-159) ===")
print(f"X:\n{np.round(X[155:160], 4)}")
print(f"Y: {np.round(Y[155:160].flatten(), 4)}")
print(f"Z:\n{np.round(Z[155:160], 4)}")
