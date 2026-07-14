#!/usr/bin/env python3
"""Debug design matrix construction for did_sun_abraham."""
import sys
sys.path.insert(0, r"C:\Users\manhn\Desktop\open-econs")

import numpy as np
import pandas as pd
from open_econs.models.causal.did_sun_abraham import _build_sunab_dummies

# Load input data
df = pd.read_csv(r"tests\r\fixtures\inputs\did_sun_abraham_input.csv")

# Extract arrays
cohort_arr = df["cohort"].values.astype(float)
period_arr = df["time"].values.astype(float)

# Build interaction dummies
dummies, dummy_names, keep_mask = _build_sunab_dummies(cohort_arr, period_arr, ref_period=-1)

print(f"Keep mask: {keep_mask.sum()} / {len(keep_mask)} observations")
print(f"Dummies shape: {dummies.shape}")
print(f"Dummy names: {dummy_names}")
print(f"First 5 dummy rows:\n{dummies[:5]}")

# Build design matrix parts
parts = [dummies[keep_mask]]
part_names = list(dummy_names)

# Add entity FE
entity_arr = df["entity"].values[keep_mask]
unique_entities = np.unique(entity_arr)
for e in unique_entities[1:]:
    col = (entity_arr == e).astype(float)
    parts.append(col.reshape(-1, 1))
    part_names.append(f"entity::{e}")

# Add time FE
time_arr = df["time"].values[keep_mask]
unique_times = np.unique(time_arr)
for t_val in unique_times[1:]:
    col = (time_arr == t_val).astype(float)
    parts.append(col.reshape(-1, 1))
    part_names.append(f"time FE::{t_val}")

# Stack
X_full = np.column_stack(parts)
print(f"\nDesign matrix shape: {X_full.shape}")
print(f"Number of parts: {len(part_names)}")
print(f"Part names: {part_names}")

# Check collinearity
from scipy.linalg import qr
Q, R_mat, P = qr(X_full, pivoting=True)
tol = max(X_full.shape[0], X_full.shape[1]) * np.finfo(float).eps * np.abs(R_mat[0, 0]) if R_mat.shape[0] > 0 else 0.0
rank = np.sum(np.abs(np.diag(R_mat)) > tol)
print(f"\nQR rank: {rank} / {X_full.shape[1]}")
print(f"Diagonal of R (first 20): {np.diag(R_mat)[:20]}")
print(f"Tolerance: {tol}")

# Check which columns are dropped
if rank < X_full.shape[1]:
    keep_cols = sorted(P[:rank])
    dropped = [i for i in range(X_full.shape[1]) if i not in keep_cols]
    print(f"\nDropped columns: {dropped}")
    print(f"Dropped names: {[part_names[i] for i in dropped]}")
    print(f"Kept columns: {keep_cols}")
    print(f"Kept names: {[part_names[i] for i in keep_cols]}")
