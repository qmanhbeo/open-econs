#!/usr/bin/env python3
"""Quick verification of did_sun_abraham() against R fixture."""
import sys
sys.path.insert(0, r"C:\Users\manhn\Desktop\open-econs")

import json
import numpy as np
import pandas as pd
from open_econs import did_sun_abraham

# Load input data
df = pd.read_csv(r"tests\r\fixtures\inputs\did_sun_abraham_input.csv")
print(f"Data shape: {df.shape}")
print(f"Cohort distribution:\n{df['cohort'].value_counts(dropna=False)}\n")

# Run Python implementation
result = did_sun_abraham(
    data=df,
    y="y",
    cohort="cohort",
    period="time",
    ref_period=-1,
    entity="entity",
    time="time",
    cluster="entity",
    covariates=["x"],
)

# Load R fixture
with open(r"tests\r\fixtures\expected\did_sun_abraham.json") as f:
    r_fix = json.load(f)

print("=== Comparison ===")
print(f"{'Quantity':<25} {'Python':>15} {'R':>15} {'Diff':>15}")
print("-" * 70)

def cmp(name, py_val, r_val):
    diff = abs(py_val - r_val)
    print(f"{name:<25} {py_val:>15.10f} {r_val:>15.10f} {diff:>15.2e}")

cmp("ATT", result.att, r_fix["att"])
cmp("SE", result.att_se, r_fix["se"])
cmp("t-stat", result.att_t_stat, r_fix["t_stat"])
cmp("p-value", result.att_p_value, r_fix["p_value"])
cmp("sigma2", result.sigma2, r_fix["sigma2"])

# Compare coefficients
print("\n=== Coefficients ===")
py_coefs = result.coefficients
r_coefs = pd.Series(r_fix["coefficients"], index=r_fix["coef_names"])
print(f"{'Name':<30} {'Python':>15} {'R':>15} {'Diff':>15}")
print("-" * 75)
for name in r_coefs.index:
    if name in py_coefs.index:
        py_val = py_coefs[name]
        r_val = r_coefs[name]
        diff = abs(py_val - r_val)
        print(f"{name:<30} {py_val:>15.10f} {r_val:>15.10f} {diff:>15.2e}")
    else:
        print(f"{name:<30} {'MISSING':>15} {r_coefs[name]:>15.10f}")

# Compare VCE
print("\n=== VCE (clustered) ===")
py_vce = result.vcov().values
r_vce = np.array(r_fix["vce_clustered"]).reshape(r_fix["vce_nrow"], r_fix["vce_ncol"])
print(f"VCE shape: Python {py_vce.shape}, R {r_vce.shape}")
if py_vce.shape == r_vce.shape:
    diff = np.max(np.abs(py_vce - r_vce))
    print(f"Max VCE element-wise diff: {diff:.2e}")
else:
    print("VCE shapes differ!")
