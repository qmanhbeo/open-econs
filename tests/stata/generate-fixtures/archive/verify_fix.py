"""Verify m2VZXA fix against real sysgmm.dta fixture."""
import open_econs as oe
import pandas as pd
from tests.stata.stata_runner import read_stata

S = read_stata("sysgmm")
df = pd.read_csv("tests/stata/fixtures/inputs/df_panel.csv")
df["entity"] = df["entity"].astype(int)

flavors = {
    "1s_nr": ("one-step", False),
    "2s_nr": ("two-step", False),
    "1s_r": ("one-step", True),
    "2s_r": ("two-step", True),
}

print(f"{'Flavor':8s} {'AR1_OE':>12s} {'AR1_Stata':>12s} {'AR1_gap':>10s}  "
      f"{'AR2_OE':>12s} {'AR2_Stata':>12s} {'AR2_gap':>10s}")
print("-" * 80)

all_ok = True
for label, (step, robust) in flavors.items():
    r = oe.abond("y ~ x + z", data=df, entity="entity", time="time",
                  step=step, lags=1, exogenous=["x", "z"],
                  collapse=True, robust=robust, system=True)
    a1_s = S[f"ar1_c_{label}"]
    a2_s = S[f"ar2_c_{label}"]
    d1 = abs(r.ar1_stat - a1_s)
    d2 = abs(r.ar2_stat - a2_s)
    ok = d1 < 1e-6 and d2 < 1e-6
    all_ok = all_ok and ok
    status = "PASS" if ok else "FAIL"
    print(f"{label:8s} {r.ar1_stat:12.8f} {a1_s:12.8f} {d1:10.2e}  "
          f"{r.ar2_stat:12.8f} {a2_s:12.8f} {d2:10.2e}  {status}")

print(f"\nAll pass at 1e-6: {all_ok}")
