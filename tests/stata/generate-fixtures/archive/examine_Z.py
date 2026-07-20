import pandas as pd, numpy as np

Zdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Z.csv")
Z = Zdf[[c for c in Zdf.columns if c.startswith('Zmat')]].values
Xdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_X.csv")
X = Xdf[[c for c in Xdf.columns if c.startswith('Xmat')]].values
Ydf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Y.csv")
Y = Ydf[[c for c in Ydf.columns if c.startswith('Ymat')]].values
ents = Zdf['entity'].values
times = Zdf['time'].values

# Focus on entity 1 (rows 0-9)
print("=== Entity 1: Z columns for non-zero rows ===")
for i in [2,3,4,6,7,8,9]:
    eq = "DIFF" if X[i,3]==0 else "LEVEL"
    zstr = " ".join(f"{z:.4f}" for z in Z[i])
    xstr = " ".join(f"{x:.4f}" for x in X[i])
    print(f"  row {i} t={times[i]:.0f} {eq}: Z=[{zstr}]")
    print(f"         X=[{xstr}] Y={Y[i,0]:.4f}")

# Check zero columns
print()
print("Z col norms:", np.round(np.linalg.norm(Z, axis=0), 6))

# Check which Z columns are GMM vs IV by correlation with X columns on level rows
lvl_idx = np.where(X[:,3] > 0.5)[0]
print()
print("Level-row Z correlations with X columns:")
for zi in range(11):
    z_vals = Z[lvl_idx[:10], zi]
    if np.std(z_vals) < 1e-10:
        print(f"  Zcol{zi}: CONSTANT/DEGENERATE")
        continue
    for xi, name in [(1,"x"), (2,"z"), (0,"L.y")]:
        x_vals = X[lvl_idx[:10], xi]
        corr = np.corrcoef(z_vals, x_vals)[0,1]
        if abs(corr) > 0.99:
            print(f"  Zcol{zi}: matches {name} (corr={corr:.4f})")

# Check diff-row Z columns
diff_idx = np.where((X[:,3] == 0) & (np.any(np.abs(X) > 1e-10, axis=1)))[0]
print()
print("Diff-row Z columns (entity 1):")
for i in diff_idx[:3]:
    print(f"  row {i}: Z=[{' '.join(f'{z:.4f}' for z in Z[i])}]")
    print(f"         X=[{' '.join(f'{x:.4f}' for x in X[i])}]")

# Check if diff GMM instruments are y levels or y diffs
# Diff eq should instrument L.y with lags of y (levels, not diffs)
# Read actual y data
csv = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv")
e1 = csv[csv["entity"]==1].sort_values("time")
y_vals = e1["y"].values
print()
print("Entity 1 y values:", np.round(y_vals, 4))
print("Entity 1 y lags: y[t-1],y[t-2],y[t-3],y[t-4],y[t-5]")
for t in range(5):
    lags = [y_vals[t-k] if t-k >= 0 else 0 for k in range(1,6)]
    print(f"  t={t}: L.y={lags[0]:.4f} L2.y={lags[1]:.4f} L3.y={lags[2]:.4f} L4.y={lags[3]:.4f} L5.y={lags[4]:.4f}")
