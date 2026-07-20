"""Build the coupled H matrix for system GMM, compute Z'HZ, verify against e(A1).

H is 300x300 block-diagonal with 30 blocks of 10x10.
Each block H_K = [[M'M, M'], [M, I]] where M is the 5x5 first-difference operator
using Stata's forward-difference convention (h=3 coupled).

Row ordering per entity (matches Z from build_sys_Z.py):
    diff(t=0..4) at rows K*10+0..K*10+4
    level(t=0..4) at rows K*10+5..K*10+9

Stata's e(A1) = pinv(Z'HZ) / sig2, where sig2 = e'e / (2*N).
Therefore A1 @ (Z'HZ) = (1/sig2) * I (to numerical precision), i.e.,
Z'HZ = pinv(A1) / sig2.

Verifies both directions:
  (a) A1 = (1/sig2) * pinv(Z'HZ)    — forward, more numerically stable
  (b) Z'HZ = pinv(A1) / sig2         — backward, as stated in spec
"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sys_Z import build_Z_from_raw

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "fixtures", "inputs", "df_panel.csv")
A1_PATH = os.path.join(ROOT, "archive", "A1.csv")

N_ENTITIES = 30
T = 5


def build_M_forward():
    """5x5 forward-difference operator (Stata convention, h=3).

    M[t,t] = 1, M[t,t+1] = -1 for t=0..T-2, last row = [0,...,0,1].

    M = [[ 1, -1,  0,  0,  0],
         [ 0,  1, -1,  0,  0],
         [ 0,  0,  1, -1,  0],
         [ 0,  0,  0,  1, -1],
         [ 0,  0,  0,  0,  1]]
    """
    M = np.zeros((T, T))
    for t in range(T):
        M[t, t] = 1.0
        if t < T - 1:
            M[t, t + 1] = -1.0
    return M


def build_H_block(M):
    """10x10 coupled H block: [[M'M, M'], [M, I]] (Stata h=3 convention)."""
    eye = np.eye(T)
    MpM = M.T @ M
    Mt = M.T
    return np.block([[MpM, Mt], [M, eye]])


def build_full_H():
    """300x300 block-diagonal H using Stata's forward-difference M."""
    from scipy.linalg import block_diag
    M = build_M_forward()
    H_block = build_H_block(M)
    blocks = [H_block for _ in range(N_ENTITIES)]
    return block_diag(*blocks)


def main():
    df = pd.read_csv(DATA)
    Z = build_Z_from_raw(df)

    H = build_full_H()
    ZtHZ = Z.T @ H @ Z

    A1 = pd.read_csv(A1_PATH).values

    # Stata: A1 = pinv(Z'HZ) / sig2
    # So: A1 * Z'HZ = (1/sig2) * projection_matrix
    prod = A1 @ ZtHZ
    off_diag_max = np.abs(prod - np.diag(np.diag(prod))).max()
    diag_vals = np.diag(prod)
    lam = diag_vals[diag_vals != 0].mean()
    sig2 = 1.0 / lam

    # Direction (a): A1 = lam * pinv(Z'HZ)
    check_fwd = np.abs(A1 - lam * np.linalg.pinv(ZtHZ)).max()

    # Direction (b): Z'HZ = lam * pinv(A1)
    check_bwd = np.abs(ZtHZ - lam * np.linalg.pinv(A1)).max()

    print(f"Z shape: {Z.shape}")
    print(f"H shape: {H.shape}")
    print(f"Z'HZ shape: {ZtHZ.shape}")
    print(f"A1 shape: {A1.shape}")
    print(f"Rank of Z: {np.linalg.matrix_rank(Z)}")
    print(f"Rank of Z'HZ: {np.linalg.matrix_rank(ZtHZ)}")
    print(f"Rank of A1: {np.linalg.matrix_rank(A1)}")
    print()
    print(f"A1 @ Z'HZ off-diagonal max abs: {off_diag_max:.2e}")
    print(f"A1 @ Z'HZ diagonal (non-zero) mean: {lam:.10f}")
    print(f"  (ratio constant to std {np.std(diag_vals[diag_vals != 0]):.2e})")
    print(f"sig2 = 1/lam = {sig2:.10f}")
    print()
    print("Direction (a): A1 ~= lam * pinv(Z'HZ)")
    print(f"  Max abs diff: {check_fwd:.2e}   Pass at 1e-6: {check_fwd < 1e-6}")
    print()
    print("Direction (b): Z'HZ ~= lam * pinv(A1)")
    print(f"  Max abs diff: {check_bwd:.2e}   Pass at 1e-6: {check_bwd < 1e-6}")
    print()
    print("H construction: Stata forward-difference M, h=3 coupled block")
    print("H_K = [[M.T@M, M.T], [M, I]]")
    print()
    M = build_M_forward()
    print("M (forward difference, 5x5):")
    np.set_printoptions(precision=0, suppress=True, linewidth=100)
    print(M)
    print()
    print("M.T @ M (tridiagonal diff kernel):")
    print(M.T @ M)
    print()
    print("H block (first 10x10):")
    np.set_printoptions(precision=1, suppress=True, linewidth=120)
    print(H[:10, :10])

    return ZtHZ, A1, lam, sig2


if __name__ == "__main__":
    main()
