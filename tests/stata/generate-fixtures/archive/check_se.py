"""Check current SEs against fixture."""
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
    oe_se = r.std_errors.values
    target = [S[f"se_Ly_c_{label}"], S[f"se_x_c_{label}"], S[f"se_z_c_{label}"], S[f"se_cons_c_{label}"]]
    rel = [abs(o - t)/abs(t) if abs(t)>1e-15 else abs(o-t) for o,t in zip(oe_se, target)]
    print(f"{label}: max rel err = {max(rel):.6e}")
