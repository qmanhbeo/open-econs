"""Deep dive: test AR test on diff-only re-estimation vs system-estimation."""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from collections import Counter
from tests.stata.stata_runner import read_stata
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.abond import _ar_test
import open_econs as oe

S = read_stata("toy_sysgmm")
df = pd.read_csv("tests/stata/generate-fixtures/toy_sysgmm.csv")
df["entity"] = df["entity"].astype(int)

y_name = "y"; x_cols = ["x", "z"]; lags = 1
exogenous = ["x", "z"]; collapse = True; min_j = 2

df_sorted = df.sort_values(["entity", "time"]).reset_index(drop=True)
ent_sorted = df_sorted["entity"].values
y_sorted = df_sorted[y_name].values
x_sorted = {c: df_sorted[c].values for c in x_cols}
entities = []
y_by_e, x_by_e = {}, {}
for e_val in pd.unique(ent_sorted):
    ev = int(e_val)
    mask = ent_sorted == e_val
    entities.append(ev)
    y_by_e[ev] = y_sorted[mask]
    x_by_e[ev] = {c: x_sorted[c][mask] for c in x_cols}

T = max(len(y_by_e[e]) for e in entities)
n_ent = len(entities)
N_ROW_PER = 2 * T

# Build system Z/X/Y
Y_sys = np.zeros(n_ent * N_ROW_PER)
X_sys = np.zeros((n_ent * N_ROW_PER, 4))
Z_sys = np.zeros((n_ent * N_ROW_PER, 11))
for ei, e_val in enumerate(entities):
    y = y_by_e[e_val]; xs = x_by_e[e_val]; Ti = len(y); base = ei * N_ROW_PER
    for t in range(Ti):
        dr = base + t
        if t >= 1: Y_sys[dr] = y[t] - y[t-1]
        if t >= 2:
            X_sys[dr,0]=y[t-1]-y[t-2]; X_sys[dr,1]=xs[x_cols[0]][t]-xs[x_cols[0]][t-1]; X_sys[dr,2]=xs[x_cols[1]][t]-xs[x_cols[1]][t-1]
            Z_sys[dr,2]=X_sys[dr,1]; Z_sys[dr,3]=X_sys[dr,2]; Z_sys[dr,5]=y[t-2]
        if t >= 3: Z_sys[dr,7]=y[t-3]
        if t >= 4: Z_sys[dr,8]=y[t-4]
        lr = base + Ti + t
        Y_sys[lr] = y[t]
        X_sys[lr,0]=y[t-1] if t>=1 else 0; X_sys[lr,1]=xs[x_cols[0]][t]; X_sys[lr,2]=xs[x_cols[1]][t]; X_sys[lr,3]=1.0
        if t>=1: Z_sys[lr,0]=xs[x_cols[0]][t]; Z_sys[lr,1]=xs[x_cols[1]][t]; Z_sys[lr,4]=1.0
        if t>=2: Z_sys[lr,6]=y[t-1]-y[t-2]
        if t>=3: Z_sys[lr,10]=y[t-2]-y[t-3]

eq_entity_sys = np.array([e for e in entities for _ in range(N_ROW_PER)])
M_fwd = np.eye(T)
for tau in range(T-1): M_fwd[tau,tau+1] = -1.0
H_block = np.block([[M_fwd.T@M_fwd, M_fwd.T], [M_fwd, np.eye(T)]])
W = block_diag(*[H_block for _ in range(n_ent)])

# Build diff-only collapsed Z/X/Y (matching abond.py diff path)
maxL = T - 1
depths = [d for d in range(2, maxL+1) if T - max(min_j, d) >= 2]
L_ar_diff = len(depths) + len(x_cols)
N_eq = n_ent * (T - min_j + 1)  # usable diff eqs

Y_coll = np.zeros(n_ent * T)
X_coll = np.zeros((n_ent * T, 1 + len(x_cols)))  # L.y + x + z
Z_coll = np.zeros((n_ent * T, L_ar_diff))
eq_coll = np.repeat(entities, T)

for ei, e_val in enumerate(entities):
    y = y_by_e[e_val]; xs = x_by_e[e_val]; base = ei * T
    for j in range(min_j, T):
        r = base + j
        Y_coll[r] = y[j] - y[j-1]
        X_coll[r, 0] = y[j-1] - y[j-2]
        X_coll[r, 1] = xs[x_cols[0]][j] - xs[x_cols[0]][j-1]
        X_coll[r, 2] = xs[x_cols[1]][j] - xs[x_cols[1]][j-1]
        col = 0
        for lag in depths:
            idx = j - lags - lag
            if idx >= 0: Z_coll[r, col] = y[idx]; col += 1
        for iv_c in x_cols: Z_coll[r, col] = xs[iv_c][j] - xs[iv_c][j-1]; col += 1

# Diff estimation weight
entity_counts = dict(Counter(eq_coll.tolist()))
H_diag = np.full(len(Y_coll), 2.0)
H_off = np.full(max(len(Y_coll)-1, 0), -1.0)
ent_arr = np.asarray(eq_coll)
for k in range(len(Y_coll)-1):
    if ent_arr[k] != ent_arr[k+1]: H_off[k] = 0.0
W_ab = np.diag(H_diag)
for k in range(len(Y_coll)-1): W_ab[k,k+1]=H_off[k]; W_ab[k+1,k]=H_off[k]

def system_ar_vecs(b, entities, y_by_e, x_by_e, x_cols, min_j, T):
    e_d = {}; X_d = {}; T_d = {}
    for e_val in entities:
        y_e = y_by_e[e_val]; xs = x_by_e[e_val]; Ti = len(y_e)
        Xi = np.zeros((Ti, 4)); Yi = np.zeros(Ti)
        for j in range(1, Ti):
            Yi[j] = y_e[j] - y_e[j-1]
            if j >= 2:
                Xi[j,0] = y_e[j-1]-y_e[j-2]
                Xi[j,1] = xs[x_cols[0]][j]-xs[x_cols[0]][j-1]
                Xi[j,2] = xs[x_cols[1]][j]-xs[x_cols[1]][j-1]
        e_i = Yi - Xi @ b; e_i[:min_j]=0; Xi[:min_j]=0
        e_d[e_val]=e_i; X_d[e_val]=Xi; T_d[e_val]=Ti
    return e_d, X_d, T_d

def diff_ar_vecs(b, entities, y_by_e, x_by_e, x_cols, lags, min_j):
    e_d = {}; X_d = {}; T_d = {}
    for e_val in entities:
        y_e = y_by_e[e_val]; xs = x_by_e[e_val]; Ti = len(y_e)
        Xi = np.zeros((Ti, 1+len(x_cols))); Yi = np.zeros(Ti)
        for j in range(1, Ti):
            col = 0
            for lag in range(1, lags+1):
                if j - lag >= 1: Xi[j,col] = y_e[j-lag]-y_e[j-lag-1]
                elif j - lag == 0: Xi[j,col] = y_e[0]
                col += 1
            for c in x_cols: Xi[j,col] = xs[c][j]-xs[c][j-1]; col += 1
            Yi[j] = y_e[j]-y_e[j-1]
        e_i = Yi - Xi @ b; e_i[:min_j]=0; Xi[:min_j]=0
        e_d[e_val]=e_i; X_d[e_val]=Xi; T_d[e_val]=Ti
    return e_d, X_d, T_d

for label, (step, robust) in [("1s_nr", ("one-step", False)), ("1s_r", ("one-step", True))]:
    onestep = (step == "one-step") and (not robust)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    
    # System estimation
    est_sys = _estimate_gmm(Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W, sig2_scale=1.0, small_sample_correction=True)
    # Diff-only estimation
    est_diff = _estimate_gmm(Y_coll, X_coll, Z_coll, eq_coll, step, robust=robust, W=W_ab, sig2_scale=0.5, small_sample_correction=True)
    
    print(f"  Sys b:  {est_sys['b'][0]:.8f} {est_sys['b'][1]:.8f} {est_sys['b'][2]:.8f} {est_sys['b'][3]:.8f}")
    print(f"  Diff b: {est_diff['b'][0]:.8f} {est_diff['b'][1]:.8f} {est_diff['b'][2]:.8f}")
    
    # AR vectors from system estimation
    e_s, X_s, T_s = system_ar_vecs(est_sys['b'], entities, y_by_e, x_by_e, x_cols, min_j, T)
    # AR vectors from diff-only estimation
    e_d, X_d, T_d = diff_ar_vecs(est_diff['b'], entities, y_by_e, x_by_e, x_cols, lags, min_j)
    
    # Per-entity Z
    Z_full_by_ent = {}
    Z_diff_sys_by_ent = {}
    Z_diff_coll_by_ent = {}
    for ei, e_val in enumerate(entities):
        base = ei * N_ROW_PER
        Z_full_by_ent[e_val] = Z_sys[base:base+2*T, :].copy()
        Z_diff_sys_by_ent[e_val] = Z_sys[base:base+T, :].copy()
        base_c = ei * T
        Z_diff_coll_by_ent[e_val] = Z_coll[base_c:base_c+T, :].copy()
        for Z in (Z_full_by_ent, Z_diff_sys_by_ent, Z_diff_coll_by_ent):
            Z[e_val][:min_j] = 0.0
    
    # System AR with system V
    k = int(est_sys['p'])
    N_d_v = float(n_ent * (T-2))
    dr = np.zeros(int(N_d_v)); idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T): dr[idx] = est_sys['e'][base+t]; idx += 1
    s2 = float(dr @ dr) / N_d_v / 2.0; s2 *= N_d_v / (N_d_v - k)
    if onestep:
        r = s2 / est_sys['sig2']
        Vp = est_sys['pV'] * r; m2 = est_sys['m2VZXA'] * r
    else:
        NObs = float(n_ent*(T-1)); wttot = float(len(Y_sys))
        rv = ((NObs-1)/(NObs-k)) / ((wttot-1)/(wttot-k))
        sm = ((NObs-1)/(NObs-k)) * (n_ent/(n_ent-1))
        Vp = est_sys['pV'] * rv; m2 = est_sys['m2VZXA'] * sm
    (ar1_s, ar2_s) = _ar_test(e_s, Z_diff_sys_by_ent, X_s, T_s, step, robust, m2, Vp, s2)
    print(f"\n  A) Sys-est AR with sys V: AR1={ar1_s[0]:.10f} AR2={ar2_s[0]:.10f}")
    
    # Same but with full ZHw (candidate 1)
    e_full_2t = {}
    for ent_idx, e_val in enumerate(entities):
        base = ent_idx * N_ROW_PER
        e_i = est_sys['e'][base:base+2*T].copy()
        e_i[0:2] = 0.0
        e_full_2t[e_val] = e_i
    # Manual AR with full ZHw
    ar_full1 = _ar_test(e_s, Z_diff_sys_by_ent, X_s, T_s, step, robust, m2, Vp, s2)
    # Check if diff-only _ar_test matches system but full ZHw doesn't
    # (already tested in triangulate_ar.py)
    
    # Diff-only AR with diff V
    kd = int(est_diff['p'])
    N_dd = float(len(Y_coll) - n_ent * min_j)  # usable diff obs
    # sig2 = e'e/(2N) * N/(N-k)
    e_all = est_diff['e']
    s2_d = float(e_all @ e_all) / float(len(Y_coll)) / 2.0
    s2_d *= float(len(Y_coll)) / (float(len(Y_coll)) - kd)
    if onestep:
        rd = s2_d / est_diff['sig2']
        Vp_d = est_diff['pV'] * rd; m2_d = est_diff['m2VZXA'] * rd
    else:
        NObs_d = float(n_ent*(T-1)); wttot_d = float(len(Y_coll))
        rv_d = ((NObs_d-1)/(NObs_d-kd)) / ((wttot_d-1)/(wttot_d-kd))
        sm_d = ((NObs_d-1)/(NObs_d-kd)) * (n_ent/(n_ent-1))
        Vp_d = est_diff['pV'] * rv_d; m2_d = est_diff['m2VZXA'] * sm_d
    (ar1_d, ar2_d) = _ar_test(e_d, Z_diff_coll_by_ent, X_d, T_d, step, robust, m2_d, Vp_d, s2_d if onestep else est_diff['sig2'])
    print(f"  B) Diff-est AR with diff V: AR1={ar1_d[0]:.10f} AR2={ar2_d[0]:.10f}")
    
    # OE diff-only
    oe_diff = oe.abond("y ~ x + z", data=df, entity="entity", time="time",
                       step=step, lags=1, exogenous=["x","z"], collapse=True, robust=robust, system=False)
    print(f"  C) OE diff-only: AR1={oe_diff.ar1_stat:.10f} AR2={oe_diff.ar2_stat:.10f}")
    
    # OE system
    oe_sys = oe.abond("y ~ x + z", data=df, entity="entity", time="time",
                       step=step, lags=1, exogenous=["x","z"], collapse=True, robust=robust, system=True)
    print(f"  D) OE system: AR1={oe_sys.ar1_stat:.10f} AR2={oe_sys.ar2_stat:.10f}")
    
    # Stata
    print(f"  E) Stata sys: AR1={S[f'ar1_{label}']:.10f} AR2={S[f'ar2_{label}']:.10f}")
    
    # If A differs from Stata but B matches Stata's diff-only... wait, diff-only doesn't have system flavor
    # The key: does diff-only AR match for diff-only estimation? (should, 40/40 pass)
    # Then if system-vs-diff difference is the gap, it's a system-specific AR issue

print("\nDone.")
