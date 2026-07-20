"""proof_sysgmm.py — System-GMM (Blundell-Bond) parity probe against xtabond2.

Goal: reproduce b_Ly = 0.009464 (two-step non-robust) to 1e-6 using PURE
PYTHON with a COUPLED weight matrix H = [[M'M, M']; [M, I]].

We build, per entity, a stacked (diff, level) system over the SAME params
[L.y, x, z, _cons].  Instrument set mirrors the fixture .do EXACTLY:

  DIFF eq  : gmm(L.y, lag(2 4) collapse)  -> 3 GMM cols (y_{t-3}, y_{t-4}, y_{t-5})
             iv(x z, eq(diff))            -> D.x, D.z
  LEVEL eq : gmm(L.y, lag(1 1) collapse)  -> 2 GMM cols (D.L.y, DL.L.y)
                                          = (y_{t-2}-y_{t-1})?? see note
             iv(x z, eq(level)) + _cons   -> x, z, _cons

Stata e(j0)=11 but the actual instrument COUNT printed by xtabond2 is 10
(e(j0) counts an extra constant-ish moment).  We build 10 Z columns.

NOTE on level-eq GMM from Stata output:
  "GMM-type ... D.L.y collapsed"  and "DL.L.y collapsed"
  These are the GMM instruments used IN the levels equation to instrument
  L.y.  D.L.y  = L.y - L2.y = y_{t-1} - y_{t-2}   (lag-1 difference of L.y)
  DL.L.y = L2.y - L3.y = y_{t-2} - y_{t-3}  (lag-2 difference of L.y)
  So level GMM instruments ARE the first-differences of deeper lags of y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PY = r"C:\Users\manhn\miniconda3\envs\open-econs-windows\python.exe"
CSV = r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv"

# Targets
TGT = dict(b_Ly=0.009464, b_x=1.134976, b_z=-0.442064, b_cons=0.090758, sig2=0.248590)

df = pd.read_csv(CSV)
entities = sorted(df["entity"].unique())
T = int(df["time"].nunique())  # 5
# per-entity arrays indexed by time 0..4
Y = {e: df[df["entity"] == e].sort_values("time")["y"].to_numpy(dtype=float) for e in entities}
X = {e: df[df["entity"] == e].sort_values("time")["x"].to_numpy(dtype=float) for e in entities}
Z = {e: df[df["entity"] == e].sort_values("time")["z"].to_numpy(dtype=float) for e in entities}


def build_system(js, diff_gmm0=1, n_diff_gmm=4, level_gmm_n=2, level_gmm_form="D"):
    """Build INTERLEAVED stacked Y,X,Z over entities (diff_t, level_t per period).

    Diff GMM cols: y_{j-2-d} for d in range(n_diff_gmm), starting at lag diff_gmm0
      (d=0 -> y_{j-2} = lag1 of L.y).  Total Z cols = n_diff_gmm + 2(Dx,Dz)
      + level_gmm_n + 2(x,z) + 1(const).
    Level GMM: form 'D' -> D.L.y,DL.L.y (differences of L.y);
               'L' -> y_{t-1}, y_{t-2} (levels of y).
    Rows interleaved: for each period j in js -> [diff row, level row].
    """
    nz = n_diff_gmm + 2 + level_gmm_n + 2 + 1
    Yr, Xr, Zr, E = [], [], [], []
    for e in entities:
        y, x, z = Y[e], X[e], Z[e]
        for j in js:
            Yr.append(y[j] - y[j - 1])
            Xr.append([y[j - 1], x[j] - x[j - 1], z[j] - z[j - 1], 0.0])
            zr = np.zeros(nz)
            for d in range(n_diff_gmm):
                idx = j - 1 - diff_gmm0 - d   # lag diff_gmm0+d of L.y
                zr[d] = y[idx] if idx >= 0 else 0.0
            zr[n_diff_gmm] = x[j] - x[j - 1]
            zr[n_diff_gmm + 1] = z[j] - z[j - 1]
            Zr.append(zr)
            E.append(e)
            # level row
            Yr.append(y[j])
            Xr.append([y[j - 1], x[j], z[j], 1.0])
            zr = np.zeros(nz)
            base = n_diff_gmm + 2
            if level_gmm_form == "D":
                zr[base] = y[j - 1] - y[j - 2] if (j - 2 >= 0) else 0.0
                if level_gmm_n >= 2:
                    zr[base + 1] = y[j - 2] - y[j - 3] if (j - 3 >= 0) else 0.0
            else:  # levels
                zr[base] = y[j - 1] if (j - 1 >= 0) else 0.0
                if level_gmm_n >= 2:
                    zr[base + 1] = y[j - 2] if (j - 2 >= 0) else 0.0
            zr[base + level_gmm_n] = x[j]
            zr[base + level_gmm_n + 1] = z[j]
            zr[base + level_gmm_n + 2] = 1.0
            Zr.append(zr)
            E.append(e)
    return (np.array(Yr), np.array(Xr), np.array(Zr), np.array(E))


def build_coupled_H(E, k, couple=True, m0=1.0):
    """H for INTERLEAVED (diff_t, level_t) rows, k periods per entity.

    M = first-difference operator with M[t,t]=m0, M[t,t-1]=-1 (t>=1).
    Coupled:  Cov = [[M M', M],[M', I]]   (e^Delta = M e^Level).
    Block-diag (couple=False): diff-diff=M M', level-level=I, no cross.
    m0 = 1.0 (standard) or 0.0 (AB "first period special").
    """
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
                H[idx + 2*a, idx + 2*b] = MM[a, b]
                if couple:
                    H[idx + 2*a, idx + 2*b + 1] = M[a, b]
                    H[idx + 2*a + 1, idx + 2*b] = M.T[a, b]
                H[idx + 2*a + 1, idx + 2*b + 1] = I[a, b]
        idx = j
    return H


def estimate(Y, X, Z, E, H, step):
    # Proven-correct GMM form (matches open_econs _gmm_core):
    #   ZtX = Z'X (L,p);  ZtY = Z'Y (L,)
    #   one-step : A1 = (Z'HZ)^{-1};  b1 = (ZtX' A1 ZtX)^{-1} (ZtX' A1 ZtY)
    #   two-step : S = sum_i (Z_i' e1_i)(Z_i' e1_i)';  A2 = S^{-1}; b2 likewise
    ZtX = Z.T @ X          # (L, p)
    ZtY = Z.T @ Y          # (L,)
    L = Z.shape[1]
    # Stata uses a generalized inverse for the (here singular) weight matrix.
    def gin(A):
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)
    if step == "one-step":
        W = H
        ZtHZ = Z.T @ W @ Z            # (L, L)
        A1 = gin(ZtHZ)
        G1 = ZtX.T @ A1 @ ZtX
        b = np.linalg.lstsq(G1, ZtX.T @ A1 @ ZtY, rcond=None)[0]
        e = Y - X @ b
        return b, e
    else:
        # one-step residuals
        ZtHZ1 = Z.T @ H @ Z
        A1 = gin(ZtHZ1)
        G1 = ZtX.T @ A1 @ ZtX
        b1 = np.linalg.lstsq(G1, ZtX.T @ A1 @ ZtY, rcond=None)[0]
        e1 = Y - X @ b1
        # two-step efficient weight S (L x L)
        S = np.zeros((L, L))
        ents = E.tolist()
        i = 0
        while i < len(E):
            e0 = ents[i]
            j = i
            while j < len(E) and ents[j] == e0:
                j += 1
            ze = Z[i:j, :].T @ e1[i:j]      # (L,)
            S += np.outer(ze, ze)
            i = j
        A2 = gin(S)
        G2 = ZtX.T @ A2 @ ZtX
        b2 = np.linalg.lstsq(G2, ZtX.T @ A2 @ ZtY, rcond=None)[0]
        e2 = Y - X @ b2
        return b2, e2


def sig2_from(Yb, Xb, Eb, b):
    """level-residual variance / 2 (Stata e(sig2) for system GMM)."""
    e = Yb - Xb @ b
    # level rows: indices where entity row is in the level part (2nd half per entity)
    ents = Eb.tolist()
    i = 0
    lvl_idx = []
    while i < len(Eb):
        e0 = ents[i]
        j = i
        while j < len(Eb) and ents[j] == e0:
            j += 1
        half = (j - i) // 2
        lvl_idx.extend(range(i + half, j))
        i = j
    er = e[lvl_idx]
    return float(np.var(er, ddof=0) / 2.0)


# ---- Grid search over Z construction + H, two-step, minimize max|dev| ----
target = TGT
js = [2, 3, 4]   # 3 periods per entity (180 moment eqs)
best = []
for diff_gmm0 in [1, 2]:
    for n_diff_gmm in [3, 4]:
        for level_gmm_n in [1, 2]:
            for form in ["D", "L"]:
                Yb, Xb, Zb, Eb = build_system(js, diff_gmm0, n_diff_gmm, level_gmm_n, form)
                if Zb.shape[1] != 11:
                    continue
                H = build_coupled_H(Eb, 3, couple=True, m0=0.0)
                try:
                    b2, e2 = estimate(Yb, Xb, Zb, Eb, H, "two-step")
                except Exception:
                    continue
                dev = max(abs(b2[0]-target["b_Ly"]), abs(b2[1]-target["b_x"]),
                          abs(b2[2]-target["b_z"]), abs(b2[3]-target["b_cons"]))
                best.append(dict(dg0=diff_gmm0, ndg=n_diff_gmm, lgn=level_gmm_n,
                                 form=form, b=b2, dev=dev))
best.sort(key=lambda c: c["dev"])
print(f"TARGET bLy={target['b_Ly']} bx={target['b_x']} bz={target['b_z']} bc={target['b_cons']}")
print("=== two-step grid (js=[2,3,4], couple, m0=0) sorted by max|dev| ===")
for c in best[:15]:
    b = c["b"]
    print(f"dg0={c['dg0']} ndg={c['ndg']} lgn={c['lgn']} form={c['form']} "
          f"dev={c['dev']:.6f} bLy={b[0]:.6f} bx={b[1]:.6f} bz={b[2]:.6f} bc={b[3]:.6f}")

print("\n=== H-variant sweep for best Z (dg0=1,ndg=4,lgn=2,form=D) ===")
Yb, Xb, Zb, Eb = build_system(js, 1, 4, 2, "D")
for couple in [True, False]:
    for m0 in [1.0, 0.0]:
        H = build_coupled_H(Eb, 3, couple=couple, m0=m0)
        b1, e1 = estimate(Yb, Xb, Zb, Eb, H, "one-step")
        b2, e2 = estimate(Yb, Xb, Zb, Eb, H, "two-step")
        print(f"couple={couple} m0={m0}: 1s(bLy={b1[0]:.5f} bx={b1[1]:.5f} bz={b1[2]:.5f})"
              f"  2s(bLy={b2[0]:.5f} bx={b2[1]:.5f} bz={b2[2]:.5f} bc={b2[3]:.5f})")
print("STATA 1s bLy=0.110421 bx=1.156291 bz=-0.603776 bc=0.061418")
print("STATA 2s bLy=0.009464 bx=1.134976 bz=-0.442064 bc=0.090758")
