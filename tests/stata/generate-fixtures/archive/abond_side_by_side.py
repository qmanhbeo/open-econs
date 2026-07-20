"""Side-by-side Z column comparison: oe vs Stata."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from open_econs.models.linear.abond import abond as _abond
from formulaic import Formula

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

# Build Z for collapsed Run B
r = _abond("y ~ x + z", data=df, entity="entity", time="time",
           step="one-step", lags=1, max_iv_lag=4, collapse=True,
           exogenous=["x", "z"])

# Rebuild Z internals
formula_obj = Formula("y ~ x + z")
mm = formula_obj.get_model_matrix(df, na_action="drop")
y_name = mm.lhs.columns[0]
x_cols = [c for c in mm.rhs.columns if c != "Intercept"]

_df = df.loc[mm.rhs.index].copy()
_df["__y"] = mm.lhs[y_name].values.ravel()
for c in x_cols:
    _df["__x__" + c] = mm.rhs[c].values

ent_vals = _df["entity"].values
time_vals = _df["time"].values
order = np.lexsort((time_vals, ent_vals))
ent_sorted = ent_vals[order]
y_sorted = _df["__y"].values[order]
x_sorted = {c: _df["__x__" + c].values[order] for c in x_cols}

entities_list = []
y_by_e = {}
x_by_e = {}
for e_val in pd.unique(ent_sorted):
    mask = ent_sorted == e_val
    entities_list.append(e_val)
    y_by_e[e_val] = y_sorted[mask]
    x_by_e[e_val] = {c: x_sorted[c][mask] for c in x_cols}

# After the fix, collapsed depths should be [2, 3] (distance 4 dropped)
depths = [2, 3]
min_j = 2
iv_cols = ["x", "z"]
n_instr = len(depths) * 1 + len(iv_cols)  # should be 4

print("=" * 70)
print("OE COLLAPSED Z STRUCTURE (after fix)")
print("=" * 70)
print(f"  depths = {depths}")
print(f"  n_instr = {len(depths)} GMM + {len(iv_cols)} std = {n_instr}")
print(f"  Z shape = ({r.n_obs}, {r.n_instruments})")
print()
print("  Column labels (oe):")
col = 0
for d in depths:
    print(f"    col {col}: y_{{t-{d}}}  (GMM, distance {d})")
    col += 1
for c in iv_cols:
    print(f"    col {col}: D.{c}  (standard, exogenous)")
    col += 1
print()

# Print Z for entity 0
e0 = entities_list[0]
y0 = y_by_e[e0]
xs0 = x_by_e[e0]
T0 = len(y0)

print(f"  Entity 0 (T={T0}), Z rows:")
print(f"    {'col 0 (y_t-2)':>16s}  {'col 1 (y_t-3)':>16s}  {'col 2 (D.x)':>12s}  {'col 3 (D.z)':>12s}")
for j in range(min_j, T0):
    zrow = np.zeros(n_instr)
    col = 0
    for lag in depths:
        if j - lag >= 0:
            zrow[col] = y0[j - lag]
        col += 1
    for iv_c in iv_cols:
        zrow[col] = xs0[iv_c][j] - xs0[iv_c][j - 1]
        col += 1
    print(f"    t={j}: {zrow[0]:16.4f}  {zrow[1]:16.4f}  {zrow[2]:12.4f}  {zrow[3]:12.4f}")

print()
print("=" * 70)
print("STATA COLLAPSED INSTRUMENT LIST (from abond_collapsed.log)")
print("=" * 70)
print("  Standard:   D.(x z)                    → 2 columns")
print("  GMM-type:   L(2/4).L.y collapsed       → 2 columns (distances 2, 3)")
print("  Total:                                4 instruments")
print()
print("  Note: Stata says 'L(2/4).L.y collapsed' but with T=5,")
print("  distance 4 has only 1 valid time period (t=4), so it is")
print("  silently dropped. The actual GMM columns are distances 2 and 3.")

print()
print("=" * 70)
print("COEFFICIENT COMPARISON: oe collapsed vs Stata collapsed (Run B, 1-step)")
print("=" * 70)
header = f"  {'':16s} {'oe':>14s} {'Stata':>14s} {'Ratio':>10s}"
print(header)
for key, oe_key in [("b_L1.y", "L1.y"), ("b_x", "x"), ("b_z", "z")]:
    sv = {"b_L1.y": -0.11984163, "b_x": 1.1258209, "b_z": -0.28974145}[key]
    ov = r.coefficients[oe_key]
    ratio = ov / sv if sv != 0 else float("inf")
    print(f"  {key:16s} {ov:14.10f} {sv:14.10f} {ratio:10.6f}")
for key, oe_key in [("se_L1.y", "L1.y"), ("se_x", "x"), ("se_z", "z")]:
    sv = {"se_L1.y": 0.24668636, "se_x": 0.17726977, "se_z": 0.10425827}[key]
    ov = r.std_errors[oe_key]
    ratio = ov / sv if sv != 0 else float("inf")
    print(f"  {key:16s} {ov:14.10f} {sv:14.10f} {ratio:10.6f}")
