"""Check AR statistics vs fixture."""
from tests.stata.stata_runner import read_stata
import open_econs as oe
import pandas as pd

S = read_stata("sysgmm")
df = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
df["entity"] = df["entity"].astype(int)

flavors = {
    "1s_nr": ("one-step", False),
    "2s_nr": ("two-step", False),
    "1s_r": ("one-step", True),
    "2s_r": ("two-step", True),
}

for label, (step, robust) in flavors.items():
    r = oe.abond(
        "y ~ x + z", df, entity="entity", time="time",
        step=step, lags=1, exogenous=["x", "z"],
        collapse=True, robust=robust, system=True,
    )
    ar1_fix = S[f"ar1_c_{label}"]
    ar2_fix = S[f"ar2_c_{label}"]
    print(f"{label}:")
    print(f"  AR1: OE={r.ar1_stat:.6f} fixture={ar1_fix:.6f} diff={abs(r.ar1_stat-ar1_fix):.2e}")
    print(f"  AR2: OE={r.ar2_stat:.6f} fixture={ar2_fix:.6f} diff={abs(r.ar2_stat-ar2_fix):.2e}")
