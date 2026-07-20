"""resolve_z.py - find exact Z + coupled H reproducing xtabond2 A1=(Z'HZ)^{-1}.

Ground truth: tests/stata/generate-fixtures/archive/A1.csv  (11x11, rank 10).
True Z has 10 cols; A1 is padded to 11 (c10 all-zero column/row).
Diff eq usable periods t in {2,3,4} (abond convention).
Coupled H (per xtabond2.ado): H = [[M'M, M']; [M, I]] stacked per entity,
  with M the first-difference operator acting on level y for the level eq.
Here we build H at the observation level so that Z'HZ = sum over entities of
  [Zd'; Zl'] @ [[M'M, M']; [M, I]] @ [Zd; Zl]
  = Zd' M'M Zd + Zd' M' Zl + Zl' M' Zd + Zl' I Zl.
"""

import numpy as np
import pandas as pd

DF = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
A1 = np.loadtxt(
    "tests/stata/generate-fixtures/archive/A1.csv", delimiter=",", skiprows=1
)
# True (non-padded) target: drop the all-zero 10th row/col (index 9).
A1_true = A1[:10, :10]

TS = [2, 3, 4]  # usable diff periods (t index)
NT = len(TS)  # 3

ents = sorted(DF.entity.unique())
n = len(ents) * NT  # total stacked rows


def build_M(M00=1.0, full=False):
    """Standard first-difference operator. If full=False: NT x NT (3x3) operating within
    the 3 usable diff periods. If full=True: 3x5 mapping all 5 level y's to diffs t=2,3,4."""
    if not full:
        M = np.zeros((NT, NT))
        for i in range(NT):
            M[i, i] = 1.0
            if i > 0:
                M[i, i - 1] = -1.0
        M[0, 0] = M00
        return M
    else:
        M = np.zeros((NT, 5))
        for i in range(NT):
            M[i, i + 2] = 1.0
            if i + 1 < 5:
                M[i, i + 1] = -1.0
        return M


def coupled_H(M):
    """Block H = [[M'M, M']; [M, I]] for one entity."""
    I = np.eye(M.shape[0])
    top = np.block([[M.T @ M, M.T], [M, I]])
    return top


def build_H(M):
    """Block-diag H across entities. Each entity block coupled_H (2*M.shape[0] rows)."""
    Hblk = coupled_H(M)
    H = np.zeros((2 * M.shape[0] * len(ents), 2 * M.shape[0] * len(ents)))
    for k in range(len(ents)):
        r = slice(k * 2 * M.shape[0], (k + 1) * 2 * M.shape[0])
        H[r, r] = Hblk
    return H


def build_Zfull(Zd, Zl, M):
    """Block-diagonal stacked instrument: per entity [Zd_e, 0; 0, Zl_e]; block-diag across entities.
    Zd/Zl are n x kd / n x kl stacked over entities in stacked_index() order."""
    kd, kl = Zd.shape[1], Zl.shape[1]
    k = kd + kl
    ndiff = M.shape[0]
    Zf = np.zeros((2 * ndiff * len(ents), k))
    for ei, e in enumerate(ents):
        r_diff = slice(2 * ndiff * ei, 2 * ndiff * ei + ndiff)
        r_lvl = slice(2 * ndiff * ei + ndiff, 2 * ndiff * ei + 2 * ndiff)
        g = ei * NT
        rd = slice(g, g + NT)
        Zf[r_diff, :kd] = Zd[rd, :]
        Zf[r_lvl, kd:] = Zl[rd, :]
    return Zf


def coupled_H(M):
    """Block H = [[M'M, M']; [M, I]] for one entity (2*NT rows/cols)."""
    I = np.eye(NT)
    top = np.block([[M.T @ M, M.T], [M, I]])
    return top


def stacked_index():
    """Return list of (entity, t) for stacked rows in diff-then-level order."""
    rows = [(e, t) for e in ents for t in TS]
    return rows


def get_level(series, e, t):
    if t < 0 or t > 4:
        return np.nan
    sub = DF[(DF.entity == e)]
    v = sub[sub.time == t]
    if len(v) == 0:
        return np.nan
    return float(v[series].iloc[0])


def collapsed_lag(e, s):
    """Collapsed (summed over available periods) instrument for lag s of y.
    Available t in TS with t>=s. Returns entity-constant scalar."""
    total = 0.0
    for t in TS:
        if t >= s:
            val = get_level("y", e, t - s)
            if not np.isnan(val):
                total += val
    return total


def build_Z(variant, const_last=True):
    """Build BLOCK-SEPARATED instruments: Z_diff (n x k_diff) for e_diff,
    Z_level (n x k_level) for e_level. Total k = k_diff + k_level (target 11).

    diff GMM (k_diff, 2 cols): 'L.L.y collapsed' (y_{t-2}) + 'L(2/4).L.y collapsed'.
       - variant diff2a: y_{t-2} + collapsed-over-periods of (y_{t-3},y_{t-4},y_{t-5})
       - variant diff2b: y_{t-2} + elementwise (y_{t-3}+y_{t-4}+y_{t-5})
    diff IV: D.x, D.z  (2 cols)
    level GMM (k_level): D.L.y=y_{t-1}-y_{t-2}, DL.L.y=y_{t-2}-y_{t-3}  (2 cols)
    level IV: x, z, _cons  (3 cols)  -> k_level=7, k_diff=4, total 11.
    """

    def col_diff(fun):
        return np.array([fun(e, t) for (e, t) in stacked_index()], dtype=float)

    def dval(series, e, t):
        return get_level(series, e, t) - get_level(series, e, t - 1)

    Zd_cols, Zd_lab = [], []
    Zl_cols, Zl_lab = [], []

    # ---- DIFF GMM ----
    # diff4a: col1 = y_{t-2} (period-varying); cols 2-4 = collapsed lags 2,3,4 (entity-const)
    if variant == "diff4a":
        Zd_cols.append(col_diff(lambda e, t: get_level("y", e, t - 2)))
        Zd_lab.append("dg_yL2")
        for s in (2, 3, 4):
            Zd_cols.append(col_diff(lambda e, t, s=s: collapsed_lag(e, s)))
            Zd_lab.append(f"dg_coll{s}")
    # diff4b: 4 period-varying lags 1,2,3,4 of L.y (y_{t-2}..y_{t-5}); t-5->0
    elif variant == "diff4b":
        for s in (2, 3, 4, 5):
            Zd_cols.append(col_diff(lambda e, t, s=s: get_level("y", e, t - s)))
            Zd_lab.append(f"dg_yL{s}")
    # diff4c: L.L.y collapsed (y_{t-2}) + L(2/4) as 3 separate collapsed lags 2,3,4
    elif variant == "diff4c":
        Zd_cols.append(col_diff(lambda e, t: get_level("y", e, t - 2)))
        Zd_lab.append("dg_yL2")
        for s in (2, 3, 4):
            Zd_cols.append(col_diff(lambda e, t, s=s: collapsed_lag(e, s)))
            Zd_lab.append(f"dg_coll{s}")
    # diff3a: only 3 cols (collapsed lags 2,3,4) to test k=10 path
    elif variant == "diff3a":
        for s in (2, 3, 4):
            Zd_cols.append(col_diff(lambda e, t, s=s: collapsed_lag(e, s)))
            Zd_lab.append(f"dg_coll{s}")
    # diff3b: 3 period-varying lags y_{t-2}, y_{t-3}, y_{t-4}  -> k=10
    elif variant == "diff3b":
        for s in (2, 3, 4):
            Zd_cols.append(col_diff(lambda e, t, s=s: get_level("y", e, t - s)))
            Zd_lab.append(f"dg_yL{s}")
    # diff3c: 3 period-varying y_{t-1}, y_{t-2}, y_{t-3}  (L.L.y, L2.L.y, L3.L.y)
    elif variant == "diff3c":
        for s in (1, 2, 3):
            Zd_cols.append(col_diff(lambda e, t, s=s: get_level("y", e, t - s)))
            Zd_lab.append(f"dg_yL{s}")

    # ---- DIFF IV: D.x, D.z ----
    Zd_cols.append(col_diff(lambda e, t: dval("x", e, t)))
    Zd_lab.append("Dx")
    Zd_cols.append(col_diff(lambda e, t: dval("z", e, t)))
    Zd_lab.append("Dz")

    # ---- LEVEL GMM: D.L.y, DL.L.y ----
    Zl_cols.append(
        col_diff(lambda e, t: get_level("y", e, t - 1) - get_level("y", e, t - 2))
    )
    Zl_lab.append("lvlGMM_DLy")
    Zl_cols.append(
        col_diff(lambda e, t: get_level("y", e, t - 2) - get_level("y", e, t - 3))
    )
    Zl_lab.append("lvlGMM_DLLy")

    # ---- LEVEL IV: x, z, _cons ----
    lx = col_diff(lambda e, t: get_level("x", e, t))
    lz = col_diff(lambda e, t: get_level("z", e, t))
    lc = np.ones(n)
    if const_last:
        Zl_cols += [lx, lz, lc]
        Zl_lab += ["x", "z", "const"]
    else:
        Zl_cols = [lc] + Zl_cols
        Zl_lab = ["const"] + Zl_lab
        Zl_cols += [lx, lz]
        Zl_lab += ["x", "z"]

    Zd = np.nan_to_num(np.column_stack(Zd_cols), nan=0.0)
    Zl = np.nan_to_num(np.column_stack(Zl_cols), nan=0.0)
    return (Zd, Zd_lab), (Zl, Zl_lab)


def build_H(M):
    """Block-diag H across entities. Each entity block coupled_H (2*NT x 2*NT)."""
    Hblk = coupled_H(M)
    H = np.zeros((2 * NT * len(ents), 2 * NT * len(ents)))
    for k in range(len(ents)):
        r = slice(k * 2 * NT, (k + 1) * 2 * NT)
        H[r, r] = Hblk
    return H


def A1_cand(Zd, Zl, M):
    Zf = build_Zfull(Zd, Zl, M)
    H = build_H(M)
    W = Zf.T @ H @ Zf
    A = np.linalg.pinv(W)
    return A


def rel_err(A, B):
    return np.max(np.abs(A - B) / (np.abs(B) + 1e-12))


def main():
    variants = ["diff3b", "diff3c", "diff4a", "diff4b"]
    best = None
    for full in [False, True]:
        for M00 in [1.0, 2.0]:
            M = build_M(M00, full=full)
            for v in variants:
                for cl in [True, False]:
                    (Zd, ld), (Zl, ll) = build_Z(v, const_last=cl)
                    k = Zd.shape[1] + Zl.shape[1]
                    target = 11 if not full else 11
                    if k != target:
                        print(f"SKIP {v} cl={cl} full={full} M00={M00}: k={k}")
                        continue
                    A = A1_cand(Zd, Zl, M)
                    e = rel_err(A, A1)
                    tag = f"v={v} cl={cl} full={full} M00={M00}"
                    print(f"{tag}: k={k} reldiff={e:.2e}")
                    if best is None or e < best[0]:
                        best = (e, tag, (Zd, Zl), M, (ld, ll), A)
    print("\nBEST:", best[1], "reldiff", best[0])
    return best


if __name__ == "__main__":
    main()
