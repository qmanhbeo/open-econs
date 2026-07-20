"""Quick check: instrument count after depth filtering fix."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from open_econs.models.linear.abond import abond as _abond

df = pd.read_csv("tests/stata/fixtures/df_panel.csv")

print("COLLAPSED Run B (max_iv_lag=4, one-step, exogenous=[x,z]):")
r = _abond("y ~ x + z", data=df, entity="entity", time="time",
           step="one-step", lags=1, max_iv_lag=4, collapse=True,
           exogenous=["x", "z"])
print(f"  instruments = {r.n_instruments}")
print(f"  b_L1.y  = {r.coefficients['L1.y']:.10f}")
print(f"  b_x     = {r.coefficients['x']:.10f}")
print(f"  b_z     = {r.coefficients['z']:.10f}")
print(f"  se_L1.y = {r.std_errors['L1.y']:.10f}")
print(f"  se_x    = {r.std_errors['x']:.10f}")
print(f"  se_z    = {r.std_errors['z']:.10f}")
print(f"  Hansen J = {r.hansen_j:.6f}  dof = {r.hansen_j_dof}")

print("\nCOLLAPSED Run C (max_iv_lag=4, two-step, exogenous=[x,z]):")
r2 = _abond("y ~ x + z", data=df, entity="entity", time="time",
            step="two-step", lags=1, max_iv_lag=4, collapse=True,
            exogenous=["x", "z"])
print(f"  instruments = {r2.n_instruments}")
print(f"  b_L1.y  = {r2.coefficients['L1.y']:.10f}")
print(f"  b_x     = {r2.coefficients['x']:.10f}")
print(f"  b_z     = {r2.coefficients['z']:.10f}")
print(f"  se_L1.y = {r2.std_errors['L1.y']:.10f}")
print(f"  se_x    = {r2.std_errors['x']:.10f}")
print(f"  se_z    = {r2.std_errors['z']:.10f}")

print("\nNONCOLLAPSED Run B (max_iv_lag=4, one-step, exogenous=[x,z]):")
r3 = _abond("y ~ x + z", data=df, entity="entity", time="time",
            step="one-step", lags=1, max_iv_lag=4, collapse=False,
            exogenous=["x", "z"])
print(f"  instruments = {r3.n_instruments}")
print(f"  b_L1.y  = {r3.coefficients['L1.y']:.10f}")
print(f"  se_L1.y = {r3.std_errors['L1.y']:.10f}")

print("\n" + "=" * 60)
print("STATA REFERENCE (collapsed, from abond_collapsed.log):")
print("  Run B: instruments=4, b_L1.y=-0.11984163, se_L1.y=0.24668636")
print("  Run C: instruments=4, b_L1.y=-0.11991842, se_L1.y=0.21366874")
print("STATA REFERENCE (non-collapsed, from abond_diag.log):")
print("  Run B: instruments=5, b_L1.y=-0.08671378, se_L1.y=0.24521319")
