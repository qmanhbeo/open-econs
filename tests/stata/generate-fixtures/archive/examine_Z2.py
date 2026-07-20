import pandas as pd, numpy as np

Zdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Z.csv")
Z = Zdf[[c for c in Zdf.columns if c.startswith('Zmat')]].values
Xdf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_X.csv")
X = Xdf[[c for c in Xdf.columns if c.startswith('Xmat')]].values
Ydf = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\sys_Y.csv")
Y = Ydf[[c for c in Ydf.columns if c.startswith('Ymat')]].values
ents = Zdf['entity'].values
times = Zdf['time'].values

csv = pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv")

# Build entity-keyed y arrays
entities = sorted(csv["entity"].unique())
Ye = {e: csv[csv["entity"]==e].sort_values("time")["y"].values for e in entities}

# Row ordering: 300 rows, 10 per entity (5 diff + 5 level)
# rows 0-4 = diff for entity 1; rows 5-9 = level for entity 1
# times: rows 0-4 have times 0,1,2,3,4; rows 5-9 have times 0,1,2,3,4

# For entity 1, row 2 (diff, t=2): X=[0.2008,-0.5729,-0.2335,0]
# This means L.y = 0.2008? But y[t-1] at t=2 = y[1] = 2.5127
# Actually: y values for entity 1 are [-0.4821, 2.5127, 2.6703, 0.2344, 1.4646]
# y[1] = 2.5127, but X row 2 col 0 = 0.2008
# Wait, X is e(X) from Stata, not just the regressor matrix

# Actually, X row 2: [0.2008, -0.5729, -0.2335, 0]
# These are: D.L.y = y[2]-y[1] = 2.6703-2.5127 = 0.1576? No.
# Or maybe: X = the moment conditions matrix, not the regressor matrix?

# Let me check: Stata e(X) for system GMM might be the "moment condition" matrix
# or the stacked regressors. Let me check what e(X) actually is.
# For system GMM, the diff eq is: Δy = β Δy_{t-1} + ... so X_diff = [Δy_{t-1}, Δx, Δz, 0]
# But X[2,0] = 0.2008 and Δy_{t-1} = y[2]-y[1] = 2.6703-2.5127 = 0.1576. Doesn't match.

# Wait, what if X is not the regressor matrix but something else?
# In xtabond2, e(X) might be the full moment condition derivatives.
# Actually, let me re-examine: the column 0 values.
# For level rows: X[6,0] = 3.2433. Entity 1 y[1] = 2.5127. Doesn't match.
# Entity 1 y[1] = 2.5127. But X[6,0] = 3.2433. 
# Hmm, maybe the row ordering is different.

# Let me check entity 2
e2 = csv[csv["entity"]==2].sort_values("time")
y2 = e2["y"].values
print("Entity 2 y:", np.round(y2, 4))
print("Entity 1 y:", np.round(Ye[1], 4))

# Rows 10-19 should be entity 2
print("\nEntity 2 rows (10-19):")
for i in range(10, 20):
    eq = "DIFF" if X[i,3]==0 else "LEVEL"
    print(f"  row {i} t={times[i]:.0f} {eq}: X=[{X[i,0]:.4f},{X[i,1]:.4f},{X[i,2]:.4f},{X[i,3]:.4f}]")

# Let me check if the times for entity 2 are shifted
print("\nAll entities times:")
for e in [1,2,3]:
    mask = ents == e
    print(f"  Entity {e}: times={times[mask].tolist()}")
