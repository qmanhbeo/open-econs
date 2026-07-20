"""Compare SEs produced by oe.abond with Stata fixtures."""
from tests.stata.stata_runner import read_stata
import pandas as pd
import open_econs as oe

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
    target = [
        S[f"se_Ly_c_{label}"],
        S[f"se_x_c_{label}"],
        S[f"se_z_c_{label}"],
        S[f"se_cons_c_{label}"],
    ]
    oe_se = r.std_errors.values
    diff = [abs(o - t) for o, t in zip(oe_se, target)]
    rel_diff = [abs(o - t) / abs(t) if abs(t) > 1e-10 else abs(o - t) for o, t in zip(oe_se, target)]
    print(f"{label}:")
    print(f"  OE:   {[f'{s:.6f}' for s in oe_se]}")
    print(f"  Stata: {[f'{s:.6f}' for s in target]}")
    print(f"  abs diff: {[f'{d:.2e}' for d in diff]}")
    print(f"  rel diff: {[f'{d:.2e}' for d in rel_diff]}")
    print()
