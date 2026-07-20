"""AR test diagnostic for system GMM — test the diff-only re-estimation hypothesis."""
import numpy as np
import pandas as pd
from tests.stata.stata_runner import read_stata
from open_econs.models._gmm_core import estimate_gmm as _estimate_gmm
from open_econs.models.linear.abond import _ar_test

S = read_stata("sysgmm")
df = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
df["entity"] = df["entity"].astype(int)

# Build data structures matching abond()
y_name = "y"
x_cols = ["x", "z"]
lags = 1
exogenous = ["x", "z"]
collapse = True

df_sorted = df.sort_values(["entity", "time"]).reset_index(drop=True)
ent_sorted = df_sorted["entity"].values
y_sorted = df_sorted[y_name].values
x_sorted = {c: df_sorted[c].values for c in x_cols}

entities = []
y_by_e, x_by_e = {}, {}
for e_val in pd.unique(ent_sorted):
    mask = ent_sorted == e_val
    entities.append(e_val)
    y_by_e[e_val] = y_sorted[mask]
    x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

T = max(len(y_by_e[e]) for e in entities)
n_ent = len(entities)
N_ROW_PER = 2 * T
total_rows = n_ent * N_ROW_PER
min_j = max(lags + 1, 2)

# Build system Z/X/Y (matching abond.py lines 476-561)
Y_sys = np.zeros(total_rows)
X_sys = np.zeros((total_rows, 1 + len(x_cols) + 1))
Z_sys = np.zeros((total_rows, 11))

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
            # col 3 (_cons) stays 0 for diff
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
from scipy.linalg import block_diag
M_fwd = np.eye(T)
for tau in range(T - 1):
    M_fwd[tau, tau + 1] = -1.0
I_T = np.eye(T)
H_block = np.block([[M_fwd.T @ M_fwd, M_fwd.T], [M_fwd, I_T]])
W = block_diag(*[H_block for _ in range(n_ent)])

flavors = {
    "1s_nr": ("one-step", False),
    "2s_nr": ("two-step", False),
    "1s_r": ("one-step", True),
    "2s_r": ("two-step", True),
}

for label, (step, robust) in flavors.items():
    # --- FULL SYSTEM estimation ---
    est = _estimate_gmm(
        Y_sys, X_sys, Z_sys, eq_entity_sys, step, robust=robust, W=W,
        sig2_scale=1.0, small_sample_correction=True,
    )
    b = est["b"]
    m2VZXA_full = est["m2VZXA"]
    pV_ar_full = est["pV_ar"]
    e_full = est["e"]

    # --- DIFF-ONLY per-entity vectors (from full-system estimation) ---
    e_by_diff = {}
    Z_by_diff = {}
    X_by_diff = {}
    for ent_idx, e_val in enumerate(entities):
        base = ent_idx * N_ROW_PER
        ei = e_full[base:base + T].copy()  # diff rows 0..T-1
        Zi = Z_sys[base:base + T, :].copy()
        Xi = X_sys[base:base + T, :].copy()
        # Zero t<min_j (=2): rows 0,1
        ei[:min_j] = 0.0
        Zi[:min_j] = 0.0
        Xi[:min_j] = 0.0
        e_by_diff[e_val] = ei
        Z_by_diff[e_val] = Zi
        X_by_diff[e_val] = Xi

    # --- DIFF-ONLY re-estimation: estimate ONLY on diff rows (no level) ---
    # Build the 150-row diff-only system
    Y_diff = np.zeros(n_ent * T)
    X_diff = np.zeros((n_ent * T, 4))
    Z_diff = np.zeros((n_ent * T, 11))
    eq_diff = np.repeat(entities, T)
    for ent_idx, e_val in enumerate(entities):
        base_out = ent_idx * T
        base_in = ent_idx * N_ROW_PER
        Y_diff[base_out:base_out + T] = Y_sys[base_in:base_in + T]
        X_diff[base_out:base_out + T] = X_sys[base_in:base_in + T]
        Z_diff[base_out:base_out + T] = Z_sys[base_in:base_in + T]

    est_diff = _estimate_gmm(
        Y_diff, X_diff, Z_diff, eq_diff, step, robust=robust, W=None,
        sig2_scale=1.0, small_sample_correction=True,
    )
    m2VZXA_do = est_diff["m2VZXA"]
    pV_ar_do = est_diff["pV_ar"]
    sig2_do = est_diff["sig2"]
    e_diff = est_diff["e"]

    # DIFF-ONLY per-entity vectors from the DIFF-ONLY estimation
    e_by_do = {}
    Z_by_do = {}
    X_by_do = {}
    for ent_idx, e_val in enumerate(entities):
        base = ent_idx * T
        ei = e_diff[base:base + T].copy()
        Zi = Z_diff[base:base + T, :].copy()
        Xi = X_diff[base:base + T, :].copy()
        ei[:min_j] = 0.0
        Zi[:min_j] = 0.0
        Xi[:min_j] = 0.0
        e_by_do[e_val] = ei
        Z_by_do[e_val] = Zi
        X_by_do[e_val] = Xi

    # Compute the Stata-corrected post-small V
    # NObs = n_ent*(T-1) = 120 for system GMM (usable diff obs with valid instruments)
    NObs = float(n_ent * (T - 1))
    wttot = float(len(Y_sys))
    NG = n_ent
    k = X_sys.shape[1]
    # --- Compute sig2_stata (diff-only override, matching abond.py) ---
    N_d = float(n_ent * (T - 2))
    diff_resid_all = np.zeros(int(N_d))
    idx = 0
    for ent_idx in range(n_ent):
        base = ent_idx * N_ROW_PER
        for t in range(2, T):
            diff_resid_all[idx] = e_full[base + t]
            idx += 1
    sig2_stata = float(diff_resid_all @ diff_resid_all) / N_d / 2.0
    sig2_stata *= N_d / (N_d - k)

    # --- Compute Stata-corrected V_post ---
    if step == "one-step" and not robust:
        # 1s_nr: V corrected by sig2_ratio
        V_post_stata = est["pV"] * (sig2_stata / est["sig2"])
    else:
        # Non-1s-nr: V corrected by (NObs-1)/(NObs-k) / ((wttot-1)/(wttot-k))
        ratio_v = ((NObs - 1.0) / (NObs - k)) / ((wttot - 1.0) / (wttot - k))
        V_post_stata = est["pV"] * ratio_v

    # Scale m2VZXA consistently with V
    sm_stata = V_post_stata[0,0] / pV_ar_full[0,0]
    m2VZXA_stata = m2VZXA_full * sm_stata

    # --- Test A: Stata-V + Stata-m2VZXA + sig2_stata (current fix in abond.py) ---
    ar1_A, ar2_A = _ar_test(
        e_by_diff, Z_by_diff, X_by_diff, {e: T for e in entities},
        step, robust, m2VZXA_stata, V_post_stata, sig2_stata,
    )

    # --- Test B: Stata-V + Stata-m2VZXA + FULL sig2 (raw) for 1s_nr only ---
    sig2_full = est["sig2"]
    ar1_B, ar2_B = _ar_test(
        e_by_diff, Z_by_diff, X_by_diff, {e: T for e in entities},
        step, robust, m2VZXA_stata, V_post_stata, sig2_full,
    )

    ar1_fix = S[f"ar1_c_{label}"]
    ar2_fix = S[f"ar2_c_{label}"]

    print(f"\n{label}:  sig2_stata={sig2_stata:.6f}  NObs={NObs:.0f}  wttot={wttot:.0f}")
    if step != "one-step" or robust:
        print(f"  ratio_v={ratio_v:.6f}  sm_stata={sm_stata:.6f}")
    print(f"  Test A (pre-V):  AR1={ar1_A[0]:.6f} diff={abs(ar1_A[0]-ar1_fix):.4f}"
          f"  AR2={ar2_A[0]:.6f} diff={abs(ar2_A[0]-ar2_fix):.4f}")
    print(f"  Test B (Sta-V):  AR1={ar1_B[0]:.6f} diff={abs(ar1_B[0]-ar1_fix):.4f}"
          f"  AR2={ar2_B[0]:.6f} diff={abs(ar2_B[0]-ar2_fix):.4f}")
    print(f"  Fixture:         AR1={ar1_fix:.6f}  AR2={ar2_fix:.6f}")
