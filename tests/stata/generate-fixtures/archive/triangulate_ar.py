"""Triangulation of system-GMM AR test — one focused shot.

Builds toy panel, runs OE + independent hand-calc following Mata _ARTests formula,
and prints every intermediate term for AR1 vs AR2 to localize the bug.
"""
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from tests.stata.stata_runner import read_stata
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.abond import _build_H_ar, _ar_test

# ── Load toy fixture and data ─────────────────────────────────────────────────
S = read_stata("toy_sysgmm")
df = pd.read_csv("tests/stata/generate-fixtures/toy_sysgmm.csv")
df["entity"] = df["entity"].astype(int)

# ── Build data structures (same as abond.py) ──────────────────────────────────
y_name = "y"
x_cols = ["x", "z"]
lags = 1
exogenous = ["x", "z"]
collapse = True
min_j = max(lags + 1, 2)

df_sorted = df.sort_values(["entity", "time"]).reset_index(drop=True)
ent_sorted = df_sorted["entity"].values
y_sorted = df_sorted[y_name].values
x_sorted = {c: df_sorted[c].values for c in x_cols}

entities = []
y_by_e, x_by_e = {}, {}
for e_val in pd.unique(ent_sorted):
    mask = ent_sorted == e_val
    entities.append(int(e_val))
    y_by_e[int(e_val)] = y_sorted[mask]
    x_by_e[int(e_val)] = {c: x_sorted[c][mask] for c in x_cols}

T = max(len(y_by_e[e]) for e in entities)
n_ent = len(entities)
N_ROW_PER = 2 * T

# ── Build system Z/X/Y (matching abond.py lines 476-561) ──────────────────────
Y_sys = np.zeros(n_ent * N_ROW_PER)
X_sys = np.zeros((n_ent * N_ROW_PER, 1 + len(x_cols) + 1))
Z_sys = np.zeros((n_ent * N_ROW_PER, 11))

for entity_index, e_val in enumerate(entities):
    y = y_by_e[e_val]
    xs = x_by_e[e_val]
    Ti = len(y)
    base = entity_index * N_ROW_PER
    for t in range(Ti):
        diff_row = base + t
        if t >= 1:
            Y_sys[diff_row] = y[t] - y[t - 1]
        if t >= 2:
            X_sys[diff_row, 0] = y[t - 1] - y[t - 2]
            X_sys[diff_row, 1] = xs[x_cols[0]][t] - xs[x_cols[0]][t - 1]
            X_sys[diff_row, 2] = xs[x_cols[1]][t] - xs[x_cols[1]][t - 1]
            Z_sys[diff_row, 2] = X_sys[diff_row, 1]
            Z_sys[diff_row, 3] = X_sys[diff_row, 2]
            Z_sys[diff_row, 5] = y[t - 2]
        if t >= 3:
            Z_sys[diff_row, 7] = y[t - 3]
        if t >= 4:
            Z_sys[diff_row, 8] = y[t - 4]
        lev_row = base + Ti + t
        Y_sys[lev_row] = y[t]
        X_sys[lev_row, 0] = y[t - 1] if t >= 1 else 0.0
        X_sys[lev_row, 1] = xs[x_cols[0]][t]
        X_sys[lev_row, 2] = xs[x_cols[1]][t]
        X_sys[lev_row, 3] = 1.0
        if t >= 1:
            Z_sys[lev_row, 0] = xs[x_cols[0]][t]
            Z_sys[lev_row, 1] = xs[x_cols[1]][t]
            Z_sys[lev_row, 4] = 1.0
        if t >= 2:
            Z_sys[lev_row, 6] = y[t - 1] - y[t - 2]
        if t >= 3:
            Z_sys[lev_row, 10] = y[t - 2] - y[t - 3]

eq_entity_sys = np.array([e for e in entities for _ in range(N_ROW_PER)])

# Coupled H
M_fwd = np.eye(T)
for tau in range(T - 1):
    M_fwd[tau, tau + 1] = -1.0
I_T = np.eye(T)
H_block = np.block([[M_fwd.T @ M_fwd, M_fwd.T], [M_fwd, I_T]])
W = block_diag(*[H_block for _ in range(n_ent)])

# ── Pre-extract per-entity Z (diff and full) ─────────────────────────────────
Z_full_by_entity = {}
Z_diff_by_entity = {}
for ent_idx, e_val in enumerate(entities):
    base = ent_idx * N_ROW_PER
    Z_full_by_entity[e_val] = Z_sys[base:base + 2 * T, :].copy()
    Z_diff_by_entity[e_val] = Z_sys[base:base + T, :].copy()


def build_system_ar_vectors(y_by_e, x_by_e, entities, x_cols, b, min_j, fix_c2=False):
    """Build diff-only per-entity vectors using system AR path (abond.py 709-717).
    
    If fix_c2=True, applies candidate-2 fix: set X_i[1,0] = y_e[0].
    """
    e_by_entity = {}
    X_by_entity = {}
    for e_val in entities:
        y_e = y_by_e[e_val]
        xs = x_by_e[e_val]
        Ti = len(y_e)
        X_i = np.zeros((Ti, 4))
        Y_i = np.zeros(Ti)
        for j in range(1, Ti):
            Y_i[j] = y_e[j] - y_e[j - 1]
            if j >= 2:
                X_i[j, 0] = y_e[j - 1] - y_e[j - 2]
                X_i[j, 1] = xs[x_cols[0]][j] - xs[x_cols[0]][j - 1]
                X_i[j, 2] = xs[x_cols[1]][j] - xs[x_cols[1]][j - 1]
            if fix_c2 and j == 1:
                X_i[j, 0] = y_e[0]
        e_full = Y_i - X_i @ b
        e_full[:min_j] = 0.0
        X_i[:min_j] = 0.0
        e_by_entity[e_val] = e_full
        X_by_entity[e_val] = X_i
    T_by_entity = {e: len(y_by_e[e]) for e in entities}
    return e_by_entity, X_by_entity, T_by_entity


def compute_ar_intermediates(
    e_by_entity, Z_diff, Z_full, X_by_entity, T_by_entity, e_full_2t,
    onestepnonrobust, sig2, m2VZXA, pV, lag, use_full_zhw=False
):
    """Compute AR test numerator, denominator, and all intermediates.
    
    When use_full_zhw=True, uses Z_full'·e_full_2t (Stata's approach for non-1-step).
    Otherwise uses Z_diff'·e_diff (current OE approach).
    """
    L = next(iter(Z_diff.values())).shape[1]
    p = next(iter(X_by_entity.values())).shape[1]
    sum_total = 0.0
    wHw = 0.0
    ZHw = np.zeros(L)
    tmp = np.zeros(p)
    
    for ent in e_by_entity:
        T_i = T_by_entity[ent]
        e_ent = e_by_entity[ent]
        wli = np.zeros(T_i)
        wli[lag:] = e_ent[:T_i - lag]
        sum_wwli = float(e_ent @ wli)
        
        if onestepnonrobust:
            H = _build_H_ar(T_i, h=3)
            wHw += float(wli @ H @ wli) * sig2
            psiw = H @ wli * sig2
            ZHw += Z_diff[ent].T @ psiw
        else:
            wHw += sum_wwli ** 2
            if use_full_zhw:
                ZHw += Z_full[ent].T @ e_full_2t[ent] * sum_wwli
            else:
                ZHw += Z_diff[ent].T @ e_ent * sum_wwli
        tmp += X_by_entity[ent].T @ wli
        sum_total += sum_wwli
    
    m2v_zhw = m2VZXA @ ZHw
    pv_t = pV @ tmp
    denom = np.sqrt(wHw + tmp @ (m2v_zhw + pv_t))
    stat = sum_total / denom if denom > 0 else float("nan")
    
    return {
        "sum_total": sum_total, "wHw": wHw, "ZHw": ZHw.copy(),
        "tmp": tmp.copy(), "m2VZXA_ZHw": m2v_zhw.copy(),
        "pV_tmp": pv_t.copy(), "denom": denom, "stat": stat,
    }


# ── Run all 4 flavors ────────────────────────────────────────────────────────
flavors = {
    "1s_nr": ("one-step", False),
    "2s_nr": ("two-step", False),
    "1s_r": ("one-step", True),
    "2s_r": ("two-step", True),
}

for label, (step, robust) in flavors.items():
    onestepnonrobust = (step == "one-step") and (not robust)
    print(f"\n{'='*72}")
    print(f"  FLAVOR: {label}  step={step}  robust={robust}")
    print(f"{'='*72}")
    
    # ── 1. Run full-system estimation ──
    est = _estimate_gmm(
        Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W,
        sig2_scale=1.0, small_sample_correction=True,
    )
    b = est["b"]
    k = int(est["p"])
    
    # ── 2. Build Stata-corrected sig2 (diff-only override, matching abond.py) ──
    N_d = float(n_ent * (T - 2))
    diff_resid_all = np.zeros(int(N_d))
    idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T):
            diff_resid_all[idx] = est["e"][base + t]
            idx += 1
    sig2_stata = float(diff_resid_all @ diff_resid_all) / N_d / 2.0
    sig2_stata *= N_d / (N_d - k)
    
    # ── 3. Build Stata-corrected post-small V and m2VZXA ──
    NObs = float(n_ent * (T - 1))
    wttot = float(len(Y_sys))
    
    if onestepnonrobust:
        ratio = sig2_stata / est["sig2"]
        V_post = est["pV"] * ratio
        m2VZXA_ar = est["m2VZXA"] * ratio
    else:
        ratio_v = ((NObs - 1.0) / (NObs - k)) / ((wttot - 1.0) / (wttot - k))
        V_post = est["pV"] * ratio_v
        sm = ((NObs - 1.0) / (NObs - k)) * (float(n_ent) / (float(n_ent) - 1.0))
        m2VZXA_ar = est["m2VZXA"] * sm
    
    # ── 4. Build per-entity vectors ──
    e_by_entity, X_by_entity, T_by_entity = build_system_ar_vectors(
        y_by_e, x_by_e, entities, x_cols, b, min_j
    )
    
    # Full 2T per-entity residuals from the system estimation
    e_full_2t = {}
    for ent_idx, e_val in enumerate(entities):
        base = ent_idx * N_ROW_PER
        e_i = est["e"][base:base + 2 * T].copy()
        e_i[0:2] = 0.0  # zero diff rows t=0,1 per AR convention
        e_full_2t[e_val] = e_i
    
    # ── 5. OE's current _ar_test ──
    ar1_oe, ar2_oe = _ar_test(
        e_by_entity, Z_diff_by_entity, X_by_entity, T_by_entity,
        step, robust, m2VZXA_ar, V_post, sig2_stata,
    )
    
    # ── 6. Independent intermediates ──
    ar1_fix = S[f"ar1_{label}"]
    ar2_fix = S[f"ar2_{label}"]
    
    print(f"  b = {b[0]:.8f} {b[1]:.8f} {b[2]:.8f} {b[3]:.8f}")
    print(f"  sig2_stata={sig2_stata:.6f}  sig2_core={est['sig2']:.6f}")
    print(f"  NObs={NObs:.0f}  wttot={wttot:.0f}  N_g={n_ent}  k={k}")
    if not onestepnonrobust:
        print(f"  ratio_v = {ratio_v:.8f}  small_mult = {sm:.8f}")
    else:
        print(f"  ratio (sig2_ratio) = {ratio:.8f}")
    
    for lagv, fixv in [(1, ar1_fix), (2, ar2_fix)]:
        lag_s = f"AR{lagv}"
        oe_stat = ar1_oe[0] if lagv == 1 else ar2_oe[0]
        print(f"\n  [{lag_s}] OE current:     stat={oe_stat:.10f}  diff={abs(oe_stat-fixv):.2e}")
        
        if onestepnonrobust:
            r = compute_ar_intermediates(
                e_by_entity, Z_diff_by_entity, Z_full_by_entity, X_by_entity,
                T_by_entity, e_full_2t, True, sig2_stata, m2VZXA_ar, V_post, lagv
            )
            print(f"  [{lag_s}] Indep (1-step): stat={r['stat']:.10f}  diff={abs(r['stat']-fixv):.2e}")
            print(f"        sum_wwli={r['sum_total']:.6f}  wHw={r['wHw']:.6f}  denom={r['denom']:.6f}")
            print(f"        ZHw[0]={r['ZHw'][0]:.6f}  ZHw[5]={r['ZHw'][5]:.6f}")
            print(f"        tmp[0]={r['tmp'][0]:.6f}  tmp[1]={r['tmp'][1]:.6f}")
        else:
            r_diff = compute_ar_intermediates(
                e_by_entity, Z_diff_by_entity, Z_full_by_entity, X_by_entity,
                T_by_entity, e_full_2t, False, sig2_stata, m2VZXA_ar, V_post, lagv,
                use_full_zhw=False
            )
            r_full = compute_ar_intermediates(
                e_by_entity, Z_diff_by_entity, Z_full_by_entity, X_by_entity,
                T_by_entity, e_full_2t, False, sig2_stata, m2VZXA_ar, V_post, lagv,
                use_full_zhw=True
            )
            print(f"  [{lag_s}] Indep (diff ZHw): stat={r_diff['stat']:.10f}  diff={abs(r_diff['stat']-fixv):.2e}")
            print(f"        sum_wwli={r_diff['sum_total']:.6f}  wHw={r_diff['wHw']:.6f}  denom={r_diff['denom']:.6f}")
            print(f"        ZHw[0]={r_diff['ZHw'][0]:.6f}  ZHw[5]={r_diff['ZHw'][5]:.6f}")
            print(f"        tmp[0]={r_diff['tmp'][0]:.6f}  tmp[1]={r_diff['tmp'][1]:.6f}")
            print(f"  [{lag_s}] Indep (full ZHw): stat={r_full['stat']:.10f}  diff={abs(r_full['stat']-fixv):.2e}")
            print(f"        sum_wwli={r_full['sum_total']:.6f}  wHw={r_full['wHw']:.6f}  denom={r_full['denom']:.6f}")
            print(f"        ZHw[0]={r_full['ZHw'][0]:.6f}  ZHw[5]={r_full['ZHw'][5]:.6f}")
            print(f"        tmp[0]={r_full['tmp'][0]:.6f}  tmp[1]={r_full['tmp'][1]:.6f}")
    
    # ── 7. Check: Does replacing ZHw alone fix it? ──
    if not onestepnonrobust:
        for lagv, fixv in [(1, ar1_fix), (2, ar2_fix)]:
            r_diff = compute_ar_intermediates(
                e_by_entity, Z_diff_by_entity, Z_full_by_entity, X_by_entity,
                T_by_entity, e_full_2t, False, sig2_stata, m2VZXA_ar, V_post, lagv,
                use_full_zhw=False
            )
            r_full = compute_ar_intermediates(
                e_by_entity, Z_diff_by_entity, Z_full_by_entity, X_by_entity,
                T_by_entity, e_full_2t, False, sig2_stata, m2VZXA_ar, V_post, lagv,
                use_full_zhw=True
            )
            d_diff = abs(r_diff['stat'] - fixv)
            d_full = abs(r_full['stat'] - fixv)
            better = d_full < d_diff
            print(f"\n  CANDIDATE1 AR{lagv}: diff_ZHw gap={d_diff:.2e}  full_ZHw gap={d_full:.2e}  full_improves={better}")

# ── Check candidate 2 (j==1 X_i asymmetry) ──
print(f"\n{'='*72}")
print("  CANDIDATE 2: j==1 DL.y construction asymmetry")
print(f"{'='*72}")
for label, (step, robust) in [("1s_nr", ("one-step", False)), ("2s_nr", ("two-step", False))]:
    onestepnonrobust = (step == "one-step") and (not robust)
    
    est = _estimate_gmm(
        Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W,
        sig2_scale=1.0, small_sample_correction=True,
    )
    b = est["b"]
    k = int(est["p"])
    
    N_d = float(n_ent * (T - 2))
    diff_resid_all = np.zeros(int(N_d))
    idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T):
            diff_resid_all[idx] = est["e"][base + t]
            idx += 1
    sig2_stata = float(diff_resid_all @ diff_resid_all) / N_d / 2.0
    sig2_stata *= N_d / (N_d - k)
    
    if onestepnonrobust:
        r = sig2_stata / est["sig2"]
        Vp = est["pV"] * r
        m2 = est["m2VZXA"] * r
    else:
        rv = ((NObs - 1.0) / (NObs - k)) / ((wttot - 1.0) / (wttot - k))
        Vp = est["pV"] * rv
        sm = ((NObs - 1.0) / (NObs - k)) * (float(n_ent) / (float(n_ent) - 1.0))
        m2 = est["m2VZXA"] * sm
    
    e_cur, X_cur, Tb = build_system_ar_vectors(y_by_e, x_by_e, entities, x_cols, b, min_j, fix_c2=False)
    e_fix, X_fix, _ = build_system_ar_vectors(y_by_e, x_by_e, entities, x_cols, b, min_j, fix_c2=True)
    
    ar1cur, ar2cur = _ar_test(e_cur, Z_diff_by_entity, X_cur, Tb, step, robust, m2, Vp, sig2_stata)
    ar1fix, ar2fix = _ar_test(e_fix, Z_diff_by_entity, X_fix, Tb, step, robust, m2, Vp, sig2_stata)
    
    a1_s = S[f"ar1_{label}"]
    a2_s = S[f"ar2_{label}"]
    print(f"  {label}: current AR1={ar1cur[0]:.10f}  fix AR1={ar1fix[0]:.10f}  Stata={a1_s:.10f}")
    print(f"          current gap={abs(ar1cur[0]-a1_s):.2e}  fix gap={abs(ar1fix[0]-a1_s):.2e}")
    print(f"          current AR2={ar2cur[0]:.10f}  fix AR2={ar2fix[0]:.10f}  Stata={a2_s:.10f}")
    print(f"          current gap={abs(ar2cur[0]-a2_s):.2e}  fix gap={abs(ar2fix[0]-a2_s):.2e}")

# ── Full OE abond() call ──
print(f"\n{'='*72}")
print("  VERIFICATION: OE abond(system=True)")
print(f"{'='*72}")
import open_econs as oe
for label, (step, robust) in flavors.items():
    oe_r = oe.abond("y ~ x + z", data=df, entity="entity", time="time",
                     step=step, lags=1, exogenous=["x", "z"],
                     collapse=True, robust=robust, system=True)
    a1_s = S[f"ar1_{label}"]
    a2_s = S[f"ar2_{label}"]
    print(f"  {label}: OE ar1={oe_r.ar1_stat:.8f}  gap={abs(oe_r.ar1_stat-a1_s):.2e}  | "
          f"OE ar2={oe_r.ar2_stat:.8f}  gap={abs(oe_r.ar2_stat-a2_s):.2e}")

print("\nDone.")
