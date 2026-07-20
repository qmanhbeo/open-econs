"""Test m2VZXA fix + candidate 1 (full ZHw) against real fixture."""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from collections import Counter
from tests.stata.stata_runner import read_stata
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.abond import _ar_test

S = read_stata("sysgmm")
df = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
df["entity"] = df["entity"].astype(int)

y_name = "y"; x_cols = ["x", "z"]; lags = 1; exogenous = ["x", "z"]; collapse = True; min_j = 2

df_sorted = df.sort_values(["entity", "time"]).reset_index(drop=True)
ent_sorted = df_sorted["entity"].values; y_sorted = df_sorted[y_name].values
x_sorted = {c: df_sorted[c].values for c in x_cols}
entities = []
y_by_e, x_by_e = {}, {}
for e_val in pd.unique(ent_sorted):
    ev = int(e_val); mask = ent_sorted == e_val
    entities.append(ev); y_by_e[ev] = y_sorted[mask]; x_by_e[ev] = {c: x_sorted[c][mask] for c in x_cols}

T = max(len(y_by_e[e]) for e in entities); n_ent = len(entities); N_ROW_PER = 2 * T

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
        if t>=3: Z_sys[dr,7]=y[t-3]
        if t>=4: Z_sys[dr,8]=y[t-4]
        lr = base + Ti + t
        Y_sys[lr]=y[t]; X_sys[lr,0]=y[t-1] if t>=1 else 0; X_sys[lr,1]=xs[x_cols[0]][t]; X_sys[lr,2]=xs[x_cols[1]][t]; X_sys[lr,3]=1.0
        if t>=1: Z_sys[lr,0]=xs[x_cols[0]][t]; Z_sys[lr,1]=xs[x_cols[1]][t]; Z_sys[lr,4]=1.0
        if t>=2: Z_sys[lr,6]=y[t-1]-y[t-2]
        if t>=3: Z_sys[lr,10]=y[t-2]-y[t-3]

eq_entity_sys = np.array([e for e in entities for _ in range(N_ROW_PER)])
M_fwd = np.eye(T)
for tau in range(T-1): M_fwd[tau,tau+1] = -1.0
W = block_diag(*[np.block([[M_fwd.T@M_fwd, M_fwd.T], [M_fwd, np.eye(T)]]) for _ in range(n_ent)])

Z_full_by_ent = {}; Z_diff_by_ent = {}
for ei, e_val in enumerate(entities):
    base = ei * N_ROW_PER
    Z_full_by_ent[e_val] = Z_sys[base:base+2*T, :].copy()
    Z_diff_by_ent[e_val] = Z_sys[base:base+T, :].copy()

def build_vecs(b, entities, y_by_e, x_by_e, x_cols, min_j):
    e_d = {}; X_d = {}; T_d = {}
    for e_val in entities:
        y_e = y_by_e[e_val]; xs = x_by_e[e_val]; Ti = len(y_e)
        Xi = np.zeros((Ti, 4)); Yi = np.zeros(Ti)
        for j in range(1, Ti):
            Yi[j] = y_e[j]-y_e[j-1]
            if j >= 2:
                Xi[j,0]=y_e[j-1]-y_e[j-2]; Xi[j,1]=xs[x_cols[0]][j]-xs[x_cols[0]][j-1]; Xi[j,2]=xs[x_cols[1]][j]-xs[x_cols[1]][j-1]
        e_i = Yi - Xi @ b; e_i[:min_j]=0; Xi[:min_j]=0
        e_d[e_val]=e_i; X_d[e_val]=Xi; T_d[e_val]=Ti
    return e_d, X_d, T_d

flavors = {"1s_nr": ("one-step", False), "2s_nr": ("two-step", False), "1s_r": ("one-step", True), "2s_r": ("two-step", True)}

print(f"{'Flavor':8s} {'Method':>15s} {'AR1':>12s} {'AR1_gap':>10s}  {'AR2':>12s} {'AR2_gap':>10s}")
print("-" * 75)

for label, (step, robust) in flavors.items():
    onestep = (step == "one-step") and (not robust)
    
    est = _estimate_gmm(Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W, sig2_scale=1.0, small_sample_correction=True)
    b = est["b"]; k = int(est["p"])
    
    N_d = float(n_ent * (T-2))
    dr = np.zeros(int(N_d)); idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T): dr[idx] = est['e'][base+t]; idx += 1
    s2 = float(dr@dr) / N_d / 2.0; s2 *= N_d / (N_d - k)
    
    NObs = float(n_ent*(T-1)); wttot = float(len(Y_sys))
    
    if onestep:
        ratio = s2 / est['sig2']; V_post = est['pV'] * ratio
        m2 = est['m2VZXA']  # pre-small (scale-invariant)
    else:
        rv = ((NObs-1)/(NObs-k)) / ((wttot-1)/(wttot-k)); sm = ((NObs-1)/(NObs-k)) * (n_ent/(n_ent-1))
        V_post = est['pV'] * rv  # Wait — is this the right V? Let me check
        # Actually est['pV'] is pre-small. The correct V_post for the AR test is
        # est['pV'] * sm = V1 * sm  (the Stata post-small V)
        V_post = est['pV_ar'] * sm
        m2 = est['m2VZXA']  # pre-small (scale-invariant)
    
    e, X, T_ent = build_vecs(b, entities, y_by_e, x_by_e, x_cols, min_j)
    
    # Full 2T residuals
    e_full_2t = {}
    for ent_idx, e_val in enumerate(entities):
        base = ent_idx * N_ROW_PER
        e_i = est['e'][base:base+2*T].copy(); e_i[0:2] = 0.0
        e_full_2t[e_val] = e_i
    
    a1_fix = S[f'ar1_c_{label}']; a2_fix = S[f'ar2_c_{label}']
    
    # Method 1: current OE (diff ZHw)
    (ar1_cur, ar2_cur) = _ar_test(e, Z_diff_by_ent, X, T_ent, step, robust, m2, V_post, s2)
    d1 = abs(ar1_cur[0] - a1_fix); d2 = abs(ar2_cur[0] - a2_fix)
    print(f"{label:8s} {'diff ZHw (curr)':>15s} {ar1_cur[0]:12.8f} {d1:10.2e}  {ar2_cur[0]:12.8f} {d2:10.2e}")
    
    # Method 2: full ZHw (candidate 1)
    # This requires computing ZHw with full 2T residual
    # Can't just call _ar_test — need a custom version. Let me compute manually.
    L = Z_diff_by_ent[next(iter(entities))].shape[1]
    p = X[next(iter(entities))].shape[1]
    
    for method_name, use_full in [("full ZHw", True)]:
        for lag_idx, lag_val in enumerate([1, 2]):
            sum_t = 0.0; whw = 0.0; ZHw = np.zeros(L); tmp = np.zeros(p)
            for ent in e:
                Ti = T_ent[ent]; e_e = e[ent]
                wli = np.zeros(Ti); wli[lag_val:] = e_e[:Ti-lag_val]
                s = float(e_e @ wli)
                if onestep:
                    from open_econs.models.linear.abond import _build_H_ar
                    Hmat = _build_H_ar(Ti); whw += float(wli @ Hmat @ wli) * s2
                    psiw = Hmat @ wli * s2; ZHw += Z_diff_by_ent[ent].T @ psiw
                    tmp += X[ent].T @ wli; sum_t += s
                else:
                    if not use_full:
                        whw += s**2; ZHw += Z_diff_by_ent[ent].T @ e_e * s
                        tmp += X[ent].T @ wli; sum_t += s
                    else:
                        whw += s**2; ZHw += Z_full_by_ent[ent].T @ e_full_2t[ent] * s
                        tmp += X[ent].T @ wli; sum_t += s
                
            if lag_idx == 1:
                denom = np.sqrt(whw + tmp @ (m2 @ ZHw + V_post @ tmp))
                ar1_new = sum_t / denom if denom > 0 else float('nan')
            else:
                denom = np.sqrt(whw + tmp @ (m2 @ ZHw + V_post @ tmp))
                ar2_new = sum_t / denom if denom > 0 else float('nan')
        
        d1n = abs(ar1_new - a1_fix); d2n = abs(ar2_new - a2_fix)
        print(f"{label:8s} {method_name:>15s} {ar1_new:12.8f} {d1n:10.2e}  {ar2_new:12.8f} {d2n:10.2e}")

print("\nDone.")
