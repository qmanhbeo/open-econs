"""Diagnostic: collapsed-vs-collapsed and noncollapsed-vs-noncollapsed."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from open_econs.models.linear.abond import abond as _abond

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")
N_ENT = df["entity"].nunique()
T_PER = df.groupby("entity").size().iloc[0]
print(f"Panel: {N_ENT} entities x {T_PER} periods = {len(df)} obs\n")

SEP = "=" * 72

# =====================================================================
# COLLAPSED COMPARISON (oe default collapse=True vs Stata collapse)
# =====================================================================
print(SEP)
print("COLLAPSED COMPARISON (oe collapse=True vs Stata collapse)")
print(SEP)

for tag, lag_spec, step in [("A", 2, "one-step"), ("B", 4, "one-step"), ("C", 4, "two-step")]:
    r = _abond(
        "y ~ x + z", data=df, entity="entity", time="time",
        step=step, lags=1, max_iv_lag=lag_spec, collapse=True,
        exogenous=["x", "z"],
    )
    print(f"\n  Run {tag} (max_iv_lag={lag_spec}, {step}):")
    print(f"    N={r.n_obs}  N_ent={r.n_entities}  instruments={r.n_instruments}")
    print(f"    b_L1.y={r.coefficients['L1.y']:.10f}")
    print(f"    b_x   ={r.coefficients['x']:.10f}")
    print(f"    b_z   ={r.coefficients['z']:.10f}")
    print(f"    se_L1.y={r.std_errors['L1.y']:.10f}")
    print(f"    se_x   ={r.std_errors['x']:.10f}")
    print(f"    se_z   ={r.std_errors['z']:.10f}")
    print(f"    Hansen J={r.hansen_j:.6f}  dof={r.hansen_j_dof}")

# =====================================================================
# NONCOLLAPSED COMPARISON (oe collapse=False vs Stata default no collapse)
# =====================================================================
print(f"\n{SEP}")
print("NONCOLLAPSED COMPARISON (oe collapse=False vs Stata default)")
print(SEP)

for tag, lag_spec, step in [("A", 2, "one-step"), ("B", 4, "one-step"), ("C", 4, "two-step")]:
    r = _abond(
        "y ~ x + z", data=df, entity="entity", time="time",
        step=step, lags=1, max_iv_lag=lag_spec, collapse=False,
        exogenous=["x", "z"],
    )
    print(f"\n  Run {tag} (max_iv_lag={lag_spec}, {step}):")
    print(f"    N={r.n_obs}  N_ent={r.n_entities}  instruments={r.n_instruments}")
    print(f"    b_L1.y={r.coefficients['L1.y']:.10f}")
    print(f"    b_x   ={r.coefficients['x']:.10f}")
    print(f"    b_z   ={r.coefficients['z']:.10f}")
    print(f"    se_L1.y={r.std_errors['L1.y']:.10f}")
    print(f"    se_x   ={r.std_errors['x']:.10f}")
    print(f"    se_z   ={r.std_errors['z']:.10f}")
    print(f"    Hansen J={r.hansen_j:.6f}  dof={r.hansen_j_dof}")

# =====================================================================
# STATA REFERENCE VALUES (from abond_diag.log, non-collapsed)
# =====================================================================
print(f"\n{SEP}")
print("STATA REFERENCE (non-collapsed, from abond_diag.log)")
print(SEP)

stata = {
    "A": {"b_Ly": -0.06654203, "b_x": 1.1615094, "b_z": -0.3106412,
           "se_Ly": 0.25634289, "se_x": 0.18460575, "se_z": 0.10729435,
           "ninstr": 4, "step": "one-step", "lag": 2},
    "B": {"b_Ly": -0.08671378, "b_x": 1.1472344, "b_z": -0.30353782,
           "se_Ly": 0.24521319, "se_x": 0.17680462, "se_z": 0.10368879,
           "ninstr": 5, "step": "one-step", "lag": 4},
    "C": {"b_Ly": -0.09296598, "b_x": 1.1277868, "b_z": -0.29591272,
           "se_Ly": 0.21085153, "se_x": 0.15348325, "se_z": 0.09469086,
           "ninstr": 5, "step": "two-step", "lag": 4},
}

for tag in ["A", "B", "C"]:
    s = stata[tag]
    print(f"\n  Stata Run {tag} (lag={s['lag']}, {s['step']}, non-collapsed):")
    print(f"    instruments={s['ninstr']}")
    print(f"    b_L1.y={s['b_Ly']:.10f}")
    print(f"    b_x   ={s['b_x']:.10f}")
    print(f"    b_z   ={s['b_z']:.10f}")
    print(f"    se_L1.y={s['se_Ly']:.10f}")
    print(f"    se_x   ={s['se_x']:.10f}")
    print(f"    se_z   ={s['se_z']:.10f}")
