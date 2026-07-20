"""Scan different V_post formulas to find which matches Stata."""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from tests.stata.stata_runner import read_stata
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.abond import _build_H_ar, _ar_test

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

def build_vecs(b):
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

flavors = {"1s_nr": ("one-step", False), "2s_nr": ("two-step", False), 
           "1s_r": ("one-step", True), "2s_r": ("two-step", True)}

for label, (step, robust) in flavors.items():
    onestep = (step == "one-step") and (not robust)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    
    est = _estimate_gmm(Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W, sig2_scale=1.0, small_sample_correction=True)
    b = est["b"]; k = int(est["p"])
    
    # sig2_stata (diff-only corrected)
    N_d = float(n_ent * (T-2))
    dr = np.zeros(int(N_d)); idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T): dr[idx] = est['e'][base+t]; idx += 1
    s2_stata = float(dr@dr) / N_d / 2.0; s2_stata *= N_d / (N_d - k)
    
    NObs = float(n_ent*(T-1)); wttot = float(len(Y_sys))
    NG = float(n_ent)
    
    # Pre-small components
    V_pre = est['pV_ar']  # V1 or V2 (no small_mult)
    m2_pre = est['m2VZXA']  # pre-small m2VZXA
    
    # Various small multipliers
    sm_1snr_core = wttot / (wttot - k)  # core 1s_nr small_mult
    sm_stata_1snr = s2_stata / est['sig2']  # sig2 ratio
    sm_core_non1snr = ((wttot-1)/(wttot-k)) * (NG/(NG-1))
    sm_stata_non1snr = ((NObs-1)/(NObs-k)) * (NG/(NG-1))
    
    a1_fix = S[f'ar1_c_{label}']; a2_fix = S[f'ar2_c_{label}']
    e, X, T_ent = build_vecs(b)
    
    # Full 2T residuals
    e_full_2t = {}
    for ent_idx, e_val in enumerate(entities):
        base = ent_idx * N_ROW_PER
        e_i = est['e'][base:base+2*T].copy(); e_i[0:2] = 0.0
        e_full_2t[e_val] = e_i
    
    # Test different V formulas
    if onestep:
        formulas = [
            ("V_pre * sig2_ratio", V_pre * sm_stata_1snr),
            ("V_pre * core_sm", V_pre * sm_1snr_core),
            ("V_pre (raw)", V_pre),
        ]
    else:
        formulas = [
            ("V_pre * stata_sm", V_pre * sm_stata_non1snr),
            ("V_pre * core_sm", V_pre * sm_core_non1snr),
            ("V_pre (raw)", V_pre),
        ]
    
    for vname, V in formulas:
        # diff ZHw
        (ar1_d, ar2_d) = _ar_test(e, Z_diff_by_ent, X, T_ent, step, robust, m2_pre, V, s2_stata)
        d1d = abs(ar1_d[0] - a1_fix); d2d = abs(ar2_d[0] - a2_fix)
        print(f"  {vname:30s} diff-ZHw: AR1_gap={d1d:.2e} AR2_gap={d2d:.2e}")
        
        if not onestep:
            # full ZHw (candidate 1) for non-1-step
            L = Z_diff_by_ent[next(iter(entities))].shape[1]
            p = X[next(iter(entities))].shape[1]
            ar1_f = float('nan'); ar2_f = float('nan')
            for lag_val, res_list in [(1, None), (2, None)]:
                sum_t = 0.0; whw = 0.0; ZHw_v = np.zeros(L); tmp_v = np.zeros(p)
                for ent in e:
                    Ti = T_ent[ent]; e_e = e[ent]
                    wli = np.zeros(Ti); wli[lag_val:] = e_e[:Ti-lag_val]
                    s = float(e_e @ wli)
                    whw += s**2; ZHw_v += Z_full_by_ent[ent].T @ e_full_2t[ent] * s
                    tmp_v += X[ent].T @ wli; sum_t += s
                dnom = np.sqrt(whw + tmp_v @ (m2_pre @ ZHw_v + V @ tmp_v))
                if lag_val == 1: ar1_f = sum_t / dnom if dnom > 0 else float('nan')
                else: ar2_f = sum_t / dnom if dnom > 0 else float('nan')
            d1f = abs(ar1_f - a1_fix); d2f = abs(ar2_f - a2_fix)
            print(f"  {vname:30s} full-ZHw: AR1_gap={d1f:.2e} AR2_gap={d2f:.2e}")

print("\nDone.")
