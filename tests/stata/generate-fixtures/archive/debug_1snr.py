"""Debug 1s_nr V correction."""
import open_econs as oe
import pandas as pd
import numpy as np

df = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
df["entity"] = df["entity"].astype(int)

# Run and capture internals
import open_econs.models.linear.abond as _abond_mod
from unittest.mock import patch

original_gmm = _abond_mod._estimate_gmm
captured = {}

def patched_gmm(*args, **kwargs):
    result = original_gmm(*args, **kwargs)
    captured.update({
        "sig2_core": result["sig2"],
        "V_core": result["V"].copy(),
        "b": result["b"].copy(),
        "p": result["p"],
    })
    return result

with patch.object(_abond_mod, "_estimate_gmm", patched_gmm):
    r = oe.abond(
        "y ~ x + z", df, entity="entity", time="time",
        step="one-step", lags=1, exogenous=["x", "z"],
        collapse=True, robust=False, system=True,
    )

print("Core results:")
print(f"  sig2_core = {captured['sig2_core']:.10f}")
print(f"  V core diag = {np.diag(captured['V_core'])}")
print(f"  b = {captured['b']}")
print(f"  k = {captured['p']}")

# The V matrix should be V1_raw * sig2_core for 1s_nr
# V1_raw = V_core / sig2_core
V1_raw = captured["V_core"] / captured["sig2_core"]
print(f"\nV1_raw diag = {np.diag(V1_raw)}")

# Now compute V_stata = V1_raw * sig2_stata * NObs/(NObs-k)
NObs = 120.0
k = float(captured["p"])
sig2_stata = r.sig2
print(f"\nsig2_stata (from result) = {sig2_stata:.10f}")
print(f"NObs/(NObs-k) = {NObs/(NObs-k):.10f}")

V_stata = V1_raw * sig2_stata * NObs/(NObs - k)
print(f"\nV_stata diag = {np.diag(V_stata)}")
print(f"SE_stata = {np.sqrt(np.maximum(np.diag(V_stata), 0))}")
print(f"OE SE = {r.std_errors.values}")

# What's the actual ratio we need?
print(f"\nActual V ratio needed: {V_stata[0,0] / captured['V_core'][0,0]:.10f}")
print(f"Computed ratio: {sig2_stata / captured['sig2_core'] * NObs/(NObs-k):.10f}")
