"""Quick smoke test for system GMM implementation."""
import pandas as pd
import numpy as np
from tests.stata.stata_runner import read_stata

S = read_stata("sysgmm")
df = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
df["entity"] = df["entity"].astype(int)
import open_econs as oe

results = {}
for step, robust, label in [
    ("one-step", False, "1s_nr"),
    ("two-step", False, "2s_nr"),
    ("one-step", True, "1s_r"),
    ("two-step", True, "2s_r"),
]:
    r = oe.abond(
        "y ~ x + z", data=df, entity="entity", time="time",
        step=step, lags=1, exogenous=["x", "z"],
        collapse=True, robust=robust, system=True,
    )
    results[label] = r
    print(f"--- {label} ---")
    print(f"  b: {np.round(r.coefficients.values, 6)}")
    print(f"  se: {np.round(r.std_errors.values, 6)}")
    print(f"  sig2={r.sig2:.6f}, n_obs={r.n_obs}, zrank={r.zrank}")
    print(f"  AR1={r.ar1_stat:.4f}, AR2={r.ar2_stat:.4f}")

targets = {
    "1s_nr": {
        "b": [S["b_Ly_c_1s_nr"], S["b_x_c_1s_nr"], S["b_z_c_1s_nr"], S["b_cons_c_1s_nr"]],
        "sig2": S["sig2_c_1s_nr"],
    },
    "2s_nr": {
        "b": [S["b_Ly_c_2s_nr"], S["b_x_c_2s_nr"], S["b_z_c_2s_nr"], S["b_cons_c_2s_nr"]],
        "sig2": S["sig2_c_2s_nr"],
    },
    "1s_r": {
        "b": [S["b_Ly_c_1s_r"], S["b_x_c_1s_r"], S["b_z_c_1s_r"], S["b_cons_c_1s_r"]],
        "sig2": S["sig2_c_1s_r"],
    },
    "2s_r": {
        "b": [S["b_Ly_c_2s_r"], S["b_x_c_2s_r"], S["b_z_c_2s_r"], S["b_cons_c_2s_r"]],
        "sig2": S["sig2_c_2s_r"],
    },
}

print()
print("=== Fixture comparison ===")
for label in ["1s_nr", "2s_nr", "1s_r", "2s_r"]:
    r = results[label]
    t = targets[label]
    b_diff = np.max(np.abs(r.coefficients.values - t["b"]))
    sig2_diff = abs(r.sig2 - t["sig2"])
    print(f"{label}: max coef diff = {b_diff:.2e}, sig2 diff = {sig2_diff:.6f}")
