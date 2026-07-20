"""Build the 300x11 system GMM Z matrix from raw panel data.

Verifies against Stata's e(Z) exported in sys_Z.csv column-by-column to 1e-6.

Usage:
    python tests/stata/generate-fixtures/build_sys_Z.py

Column formulas (confirmed from Stata xtabond2 with collapse):
    DIFF rows (t=0..4):
        Z3 = D.x[t] = x[t] - x[t-1],        0 if t < 2
        Z4 = D.z[t] = z[t] - z[t-1],        0 if t < 2
        Z6 = L2.y = y[t-2],                  0 if t < 2
        Z8 = L3.y = y[t-3],                  0 if t < 3
        Z9 = L4.y = y[t-4],                  0 if t < 4
        Z1=Z2=Z5=Z7=Z10=Z11 = 0

    LEVEL rows (t=0..4):
        Z1 = x[t],                            0 if t < 1
        Z2 = z[t],                            0 if t < 1
        Z5 = 1.0 (_cons),                     0 if t < 1
        Z7 = D.L.y = y[t-1] - y[t-2],         0 if t < 2
        Z11 = DL.L.y = y[t-2] - y[t-3],       0 if t < 3
        Z3=Z4=Z6=Z8=Z9=Z10 = 0

    Z10 is always 0 (unused column).
"""
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA = os.path.join(ROOT, "..", "fixtures", "inputs", "df_panel.csv")
REF = os.path.join(ROOT, "sys_Z.csv")

N_ENTITIES = 30
T = 5  # periods per entity

Z_COLS_REF = [f"Zmat{i}" for i in range(1, 12)]


def build_Z_from_raw(df: pd.DataFrame) -> np.ndarray:
    Z = np.zeros((N_ENTITIES * 2 * T, 11))

    for k in range(N_ENTITIES):
        mask = df["entity"] == k
        sub = df.loc[mask].sort_values("time")
        y = sub["y"].values
        x = sub["x"].values
        z = sub["z"].values

        base = k * 2 * T

        for t in range(T):
            # DIFF rows: offset 0..4 within entity block
            diff_row = base + t
            if t >= 2:
                Z[diff_row, 2] = x[t] - x[t - 1]  # Z3 = D.x
                Z[diff_row, 3] = z[t] - z[t - 1]  # Z4 = D.z
                Z[diff_row, 5] = y[t - 2]         # Z6 = L2.y
            if t >= 3:
                Z[diff_row, 7] = y[t - 3]         # Z8 = L3.y
            if t >= 4:
                Z[diff_row, 8] = y[t - 4]         # Z9 = L4.y

            # LEVEL rows: offset 5..9 within entity block
            lev_row = base + T + t
            if t >= 1:
                Z[lev_row, 0] = x[t]              # Z1 = x
                Z[lev_row, 1] = z[t]              # Z2 = z
                Z[lev_row, 4] = 1.0               # Z5 = _cons
            if t >= 2:
                Z[lev_row, 6] = y[t - 1] - y[t - 2]  # Z7 = D.L.y
            if t >= 3:
                Z[lev_row, 10] = y[t - 2] - y[t - 3]  # Z11 = DL.L.y

    return Z


def verify():
    df = pd.read_csv(DATA)
    ref = pd.read_csv(REF)

    Z = build_Z_from_raw(df)
    Z_ref = ref[Z_COLS_REF].values

    ok = True
    for j in range(11):
        colname = f"Zmat{j + 1}"
        diff = np.abs(Z[:, j] - Z_ref[:, j])
        maxdiff = diff.max()
        meandiff = diff.mean()
        if maxdiff > 1e-6:
            worst = np.argmax(diff)
            print(f"  FAIL {colname}: max diff = {maxdiff:.2e}, mean diff = {meandiff:.2e}, worst at row {worst}")
            print(f"    built = {Z[worst, j]:.15f}, ref = {Z_ref[worst, j]:.15f}")
            ok = False
        else:
            print(f"  PASS {colname}: max diff = {maxdiff:.2e}")

    if ok:
        print("\nPASS: All 11 Z columns match Stata e(Z) to 1e-6.")
    else:
        print("\nFAIL: Some columns exceeded tolerance.")


if __name__ == "__main__":
    verify()
