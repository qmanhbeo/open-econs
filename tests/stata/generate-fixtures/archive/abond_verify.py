"""Verify: for each distance {2,3,4}, how many (entity, t) pairs have valid data."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")
N_ENT = df["entity"].nunique()
T_PER = df.groupby("entity").size().iloc[0]

print(f"Panel: {N_ENT} entities x {T_PER} periods = {len(df)} obs")
print("min_j (first usable differenced equation) = lags+1 = 2")
print(f"Usable equations per entity: t = 2, 3, 4  (T - min_j = {T_PER - 2})")
print()

# For each entity, for each usable equation t, check which distances are valid.
# A distance d is valid at time t if y_{t-d} exists AND the differenced equation
# at t is usable (t >= min_j = 2).
#
# The GMM instrument for L.y at distance d is y_{t-d}.
# For the differenced equation at t, we need y_{t-d} to be non-missing.
# With balanced panel, y_{t-d} exists iff t-d >= 0.
#
# But also: for the differenced regressor L.y_diff = y_{t-1} - y_{t-2},
# we need t-1 >= 0 and t-2 >= 0, i.e. t >= 2.
#
# For the GMM instrument y_{t-d}:
#   t=2: need 2-d >= 0 → d <= 2 → only d=2 is valid
#   t=3: need 3-d >= 0 → d <= 3 → d=2,3 are valid
#   t=4: need 4-d >= 0 → d <= 4 → d=2,3,4 are valid

print("=" * 60)
print("VALID DISTANCES PER EQUATION TIME PERIOD")
print("=" * 60)
for t in [2, 3, 4]:
    valid = [d for d in [2, 3, 4] if t - d >= 0]
    print(f"  t={t}: valid distances = {valid}")

print()
print("=" * 60)
print("TOTAL VALID (entity, t) PAIRS PER DISTANCE")
print("=" * 60)
for d in [2, 3, 4]:
    count = 0
    for e in range(N_ENT):
        for t in [2, 3, 4]:
            if t - d >= 0:
                count += 1
    print(f"  distance {d}: {count} valid pairs across {N_ENT} entities")

print()
print("=" * 60)
print("Z MATRIX COLUMN ANALYSIS (oe current, collapse=True)")
print("=" * 60)
print("oe Z columns (5 total):")
print("  col 0: y_{{t-2}}  (distance 2) — valid at t=2,3,4 → 90 valid rows")
print("  col 1: y_{{t-3}}  (distance 3) — valid at t=3,4   → 60 valid rows")
print("  col 2: y_{{t-4}}  (distance 4) — valid at t=4     → 30 valid rows")
print("  col 3: D.x       (standard)    — valid at t=2,3,4 → 90 valid rows")
print("  col 4: D.z       (standard)    — valid at t=2,3,4 → 90 valid rows")
print()
print("Distance 4 has ONLY 30 valid rows (out of 90 equations).")
print("The other 60 rows are ZERO (instrument not available).")
print()
print("Stata likely drops this column entirely when it detects")
print("insufficient variation, reporting 4 instruments instead of 5.")
