"""Validate non-collapsed GMM block-builder before integration."""
import numpy as np


def _build_noncollapsed_gmm_block(var, depth, T, lag_offset):
    n_cols = T - depth
    block = np.zeros((T, n_cols))
    for k in range(n_cols):
        j = k + depth + lag_offset
        if j < T:
            block[j, k] = var[k]
    return block


# Toy: T=5, lags=1, depths=[2,3,4]
y = np.array([10, 20, 30, 40, 50])
x = np.array([1, 2, 3, 4, 5])

print("=== L.y blocks (lag_offset=1) ===")
for d in [2, 3, 4]:
    blk = _build_noncollapsed_gmm_block(y, d, 5, lag_offset=1)
    print(f"Depth {d}: shape {blk.shape}")
    print(blk)

print("\n=== gmm_c blocks (lag_offset=0) ===")
for d in [2, 3, 4]:
    blk = _build_noncollapsed_gmm_block(x, d, 5, lag_offset=0)
    print(f"Depth {d}: shape {blk.shape}")
    print(blk)

# Column count
T = 5
depths = [2, 3, 4]
n_endog = 3
n_gmm = n_endog * sum(T - d for d in depths)
print(f"\nn_gmm_cols = {n_endog} * {sum(T - d for d in depths)} = {n_gmm}")

# Row-sum check: L.y depth 2 has non-zero only at rows 3,4 (0-indexed)
blk = _build_noncollapsed_gmm_block(y, 2, 5, lag_offset=1)
print(f"\nL.y depth 2 row sums (expect 0 at [0,1,2]): {blk.sum(axis=1)}")

blk = _build_noncollapsed_gmm_block(x, 2, 5, lag_offset=0)
print(f"gmm_c depth 2 row sums (expect 0 at [0,1]): {blk.sum(axis=1)}")

# Verify hand derivation for L.y depth 2:
# Column k=0 -> j=3, val=y[0]=10 -> Z[3,0]=10
# Column k=1 -> j=4, val=y[1]=20 -> Z[4,1]=20
# Column k=2 -> j=5 >= T -> all-zero col
blk = _build_noncollapsed_gmm_block(y, 2, 5, lag_offset=1)
assert blk[3, 0] == 10, f"Expected blk[3,0]=10, got {blk[3,0]}"
assert blk[4, 1] == 20, f"Expected blk[4,1]=20, got {blk[4,1]}"
assert np.allclose(blk[:, 2], 0), "Column 2 should be all-zero"
print("\nAll hand-checks passed!")
