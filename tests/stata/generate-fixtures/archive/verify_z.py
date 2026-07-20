"""verify_z.py — check candidate Z against Stata e(A1) = (Z'HZ)^{-1}."""

import numpy as np
import pandas as pd

CSV = r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv"
df = pd.read_csv(CSV)
entities = sorted(df["entity"].unique())
T = int(df["time"].nunique())
YD = {
    e: df[df["entity"] == e].sort_values("time")["y"].to_numpy(dtype=float)
    for e in entities
}
XD = {
    e: df[df["entity"] == e].sort_values("time")["x"].to_numpy(dtype=float)
    for e in entities
}
ZD = {
    e: df[df["entity"] == e].sort_values("time")["z"].to_numpy(dtype=float)
    for e in entities
}


def build_system(js, diff_gmm0=1, n_diff_gmm=4, level_gmm_n=2, level_gmm_form="D"):
    nz = n_diff_gmm + 2 + level_gmm_n + 2 + 1
    Yr, Xr, Zr, E = [], [], [], []
    for e in entities:
        y, x, z = Y[e], X[e], Z[e]
        for j in js:
            Yr.append(y[j] - y[j - 1])
            Xr.append([y[j - 1], x[j] - x[j - 1], z[j] - z[j - 1], 0.0])
            zr = np.zeros(nz)
            for d in range(n_diff_gmm):
                idx = j - 1 - diff_gmm0 - d
                zr[d] = y[idx] if idx >= 0 else 0.0
            zr[n_diff_gmm] = x[j] - x[j - 1]
            zr[n_diff_gmm + 1] = z[j] - z[j - 1]
            Zr.append(zr)
            E.append(e)
            Yr.append(y[j])
            Xr.append([y[j - 1], x[j], z[j], 1.0])
            zr = np.zeros(nz)
            base = n_diff_gmm + 2
            if level_gmm_form == "D":
                zr[base] = y[j - 1] - y[j - 2] if (j - 2 >= 0) else 0.0
                if level_gmm_n >= 2:
                    zr[base + 1] = y[j - 2] - y[j - 3] if (j - 3 >= 0) else 0.0
            else:
                zr[base] = y[j - 1] if (j - 1 >= 0) else 0.0
                if level_gmm_n >= 2:
                    zr[base + 1] = y[j - 2] if (j - 2 >= 0) else 0.0
            zr[base + level_gmm_n] = x[j]
            zr[base + level_gmm_n + 1] = z[j]
            zr[base + level_gmm_n + 2] = 1.0
            Zr.append(zr)
            E.append(e)
    return np.array(Yr), np.array(Xr), np.array(Zr), np.array(E)


def build_coupled_H(E, k, couple=True, m0=1.0):
    n = len(E)
    H = np.zeros((n, n))
    ents = E.tolist()
    idx = 0
    while idx < n:
        e = ents[idx]
        j = idx
        while j < n and ents[j] == e:
            j += 1
        kk = (j - idx) // 2
        M = np.eye(kk) - np.eye(kk, k=1)
        if m0 != 1.0:
            M[0, 0] = m0
        MM = M @ M.T
        I = np.eye(kk)
        for a in range(kk):
            for b in range(kk):
                H[idx + 2 * a, idx + 2 * b] = MM[a, b]
                if couple:
                    H[idx + 2 * a, idx + 2 * b + 1] = M[a, b]
                    H[idx + 2 * a + 1, idx + 2 * b] = M.T[a, b]
                H[idx + 2 * a + 1, idx + 2 * b + 1] = I[a, b]
        idx = j
    return H


A1 = pd.read_csv(
    r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\A1.csv"
).values
A1inv = np.linalg.pinv(A1)
print("A1 shape", A1.shape)


def build_system_ni(js, diff_gmm0=1, n_diff_gmm=4, level_gmm_n=2, level_gmm_form="D"):
    """NON-interleaved: all diff rows then all level rows per entity."""
    nz = n_diff_gmm + 2 + level_gmm_n + 2 + 1
    Yd, Xd, Zd, Yl, Xl, Zl, E = [], [], [], [], [], [], []
    for e in entities:
        y, x, z = YD[e], XD[e], ZD[e]
        for j in js:
            Yd.append(y[j] - y[j - 1])
            Xd.append([y[j - 1], x[j] - x[j - 1], z[j] - z[j - 1], 0.0])
            zr = np.zeros(nz)
            for d in range(n_diff_gmm):
                idx = j - 1 - diff_gmm0 - d
                zr[d] = y[idx] if idx >= 0 else 0.0
            zr[n_diff_gmm] = x[j] - x[j - 1]
            zr[n_diff_gmm + 1] = z[j] - z[j - 1]
            Zd.append(zr)
            E.append(e)
        for j in js:
            Yl.append(y[j])
            Xl.append([y[j - 1], x[j], z[j], 1.0])
            zr = np.zeros(nz)
            base = n_diff_gmm + 2
            if level_gmm_form == "D":
                zr[base] = y[j - 1] - y[j - 2] if j - 2 >= 0 else 0.0
                if level_gmm_n >= 2:
                    zr[base + 1] = y[j - 2] - y[j - 3] if j - 3 >= 0 else 0.0
            else:
                zr[base] = y[j - 1] if j - 1 >= 0 else 0.0
                if level_gmm_n >= 2:
                    zr[base + 1] = y[j - 2] if j - 2 >= 0 else 0.0
            zr[base + level_gmm_n] = x[j]
            zr[base + level_gmm_n + 1] = z[j]
            zr[base + level_gmm_n + 2] = 1.0
            Zl.append(zr)
            E.append(e)
    Y = np.array(Yd + Yl)
    X = np.array(Xd + Xl)
    Z = np.array(Zd + Zl)
    E = np.array(E)
    return Y, X, Z, E


def build_H_block(E, k, m0=0.0):
    """Non-interleaved coupled H: per entity block [[M'M, M'],[M, I]] over
    (diff_block k rows, level_block k rows). M = I - shift (m0 at [0,0])."""
    n = len(E)
    H = np.zeros((n, n))
    ents = E.tolist()
    idx = 0
    while idx < n:
        e = ents[idx]
        j = idx
        while j < n and ents[j] == e:
            j += 1
        kk = (j - idx) // 2
        M = np.eye(kk) - np.eye(kk, k=1)
        if m0 != 1.0:
            M[0, 0] = m0
        MM = M @ M.T
        I = np.eye(kk)
        H[idx : idx + kk, idx : idx + kk] = MM  # diff-diff
        H[idx : idx + kk, idx + kk : j] = M.T  # diff-level
        H[idx + kk : j, idx : idx + kk] = M  # level-diff
        H[idx + kk : j, idx + kk : j] = I  # level-level
        idx = j
    return H


for dg0, ndg, lgn, form in [(1, 4, 2, "D"), (2, 3, 2, "D")]:
    Yb, Xb, Zb, Eb = build_system_ni([2, 3, 4], dg0, ndg, lgn, form)
    for m0 in [0.0, 1.0]:
        H = build_H_block(Eb, 3, m0=m0)
        ZHZ = Zb.T @ H @ Zb
        rel = np.max(np.abs(ZHZ - A1inv)) / (np.max(np.abs(A1inv)) + 1e-12)
        print(
            "NI dg0=%d ndg=%d lgn=%d form=%s m0=%.1f ZHZvsA1inv rel=%.5f Zcols=%d"
            % (dg0, ndg, lgn, form, m0, rel, Zb.shape[1])
        )

# diagnostic: are diagonals a permutation (scale-invariant)?
Yb, Xb, Zb, Eb = build_system_ni([2, 3, 4], 1, 4, 2, "D")
H = build_H_block(Eb, 3, m0=0.0)
ZHZ = Zb.T @ H @ Zb
print("diag A1inv (sorted):", np.round(np.sort(np.diag(A1inv)), 5))
print("diag ZHZ   (sorted):", np.round(np.sort(np.diag(ZHZ)), 5))
print("ratio (ZHZ/A1inv) sorted:", np.round(np.sort(np.diag(ZHZ) / np.diag(A1inv)), 4))
