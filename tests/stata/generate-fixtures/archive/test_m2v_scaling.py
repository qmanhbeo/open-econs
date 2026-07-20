"""Test pre-small vs post-small m2VZXA in AR test denominator."""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from tests.stata.stata_runner import read_stata
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.abond import _build_H_ar, _ar_test

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
    ev = int(e_val); mask = ent_sorted == e_val
    entities.append(ev); y_by_e[ev] = y_sorted[mask]; x_by_e[ev] = {c: x_sorted[c][mask] for c in x_cols}

T = max(len(y_by_e[e]) for e in entities); n_ent = len(entities); N_ROW_PER = 2 * T

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

def build_vectors(b, entities, y_by_e, x_by_e, x_cols, min_j):
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

Z_by_entity = {}
for ei, e_val in enumerate(entities):
    base = ei * N_ROW_PER
    Z_by_entity[e_val] = Z_sys[base:base+T, :].copy()
    Z_by_entity[e_val][:min_j] = 0.0

for label, (step, robust) in [("1s_nr", ("one-step", False)), ("1s_r", ("one-step", True))]:
    onestep = (step == "one-step") and (not robust)
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")
    
    est = _estimate_gmm(Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W, sig2_scale=1.0, small_sample_correction=True)
    b = est["b"]; k = int(est["p"])
    
    # sig2_stata
    N_d = float(n_ent * (T-2))
    dr = np.zeros(int(N_d)); idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T): dr[idx] = est['e'][base+t]; idx += 1
    s2 = float(dr@dr) / N_d / 2.0; s2 *= N_d / (N_d - k)
    
    NObs = float(n_ent*(T-1)); wttot = float(len(Y_sys))
    
    if onestep:
        ratio = s2 / est['sig2']
        V_post = est['pV'] * ratio
        m2_raw = est['m2VZXA']  # pre-small
        m2_scaled = est['m2VZXA'] * ratio  # post-small (current OE)
    else:
        rv = ((NObs-1)/(NObs-k)) / ((wttot-1)/(wttot-k))
        sm = ((NObs-1)/(NObs-k)) * (n_ent/(n_ent-1))
        V_post = est['pV'] * rv
        m2_raw = est['m2VZXA']  # pre-small
        m2_scaled = est['m2VZXA'] * sm  # post-small
    
    e, X, T_ent = build_vectors(b, entities, y_by_e, x_by_e, x_cols, min_j)
    
    # Test with pre-small m2VZXA (Stata internal)
    (ar1_pre, ar2_pre) = _ar_test(e, Z_by_entity, X, T_ent, step, robust, m2_raw, V_post, s2)
    # Test with post-small m2VZXA (current OE)
    (ar1_post, ar2_post) = _ar_test(e, Z_by_entity, X, T_ent, step, robust, m2_scaled, V_post, s2)
    
    s_ar1 = S[f'ar1_{label}']; s_ar2 = S[f'ar2_{label}']
    
    print(f"  Pre-small m2VZXA:  AR1={ar1_pre[0]:.10f} diff={abs(ar1_pre[0]-s_ar1):.2e} | AR2={ar2_pre[0]:.10f} diff={abs(ar2_pre[0]-s_ar2):.2e}")
    print(f"  Post-small m2VZXA: AR1={ar1_post[0]:.10f} diff={abs(ar1_post[0]-s_ar1):.2e} | AR2={ar2_post[0]:.10f} diff={abs(ar2_post[0]-s_ar2):.2e}")
    print(f"  Stata:              AR1={s_ar1:.10f} | AR2={s_ar2:.10f}")

print("\nDone.")
