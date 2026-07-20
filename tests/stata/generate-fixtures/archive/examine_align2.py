import pandas as pd

raw = pd.read_csv('tests/stata/fixtures/inputs/df_panel.csv')
z = pd.read_csv('tests/stata/generate-fixtures/sys_Z.csv')

# Build entity data arrays
ents = sorted(raw.entity.unique())
y_by_ent = {int(e): raw[raw.entity==e].y.values for e in ents}
x_by_ent = {int(e): raw[raw.entity==e].x.values for e in ents}
z_by_ent = {int(e): raw[raw.entity==e].z.values for e in ents}

def find_entity_for_val(val, lookup_fn, candidates, tol=1e-6):
    """Find which entity produces this value from the lookup function."""
    for e in candidates:
        if abs(lookup_fn(e) - val) < tol:
            return e
    return None

print("=== Row-by-row entity correspondence (first 30 dataset obs) ===\n")
for i in range(30):
    zr = z.iloc[i]
    ds_ent = int(zr.entity) if not pd.isna(zr.entity) else -1
    ds_t = int(zr.time) if not pd.isna(zr.time) else -1
    
    z3 = zr['Zmat3']
    z4 = zr['Zmat4']
    z6 = zr['Zmat6']
    z8 = zr['Zmat8']
    z9 = zr['Zmat9']
    
    # Determine row type: if Z3 (D.x) is non-zero, it's a diff row
    # If Z1/Z2/Z5 are non-zero, it's a level row
    z1 = zr['Zmat1']
    z2 = zr['Zmat2']
    z5 = zr['Zmat5']
    
    row_type = 'diff' if z3 != 0 else ('level' if z5 == 1.0 else 'zero')
    
    # Try to find entity from z3 (D.x) for diff rows
    inferred_ent = -1
    if row_type == 'diff' and z3 != 0 and ds_t >= 1:
        for e in range(30):
            if ds_t < len(x_by_ent[e]):
                dx = x_by_ent[e][ds_t] - x_by_ent[e][ds_t-1]
                if abs(dx - z3) < 1e-6:
                    inferred_ent = e
                    break
    
    elif row_type == 'level' and z1 != 0:
        for e in range(30):
            if ds_t < len(x_by_ent[e]):
                if abs(x_by_ent[e][ds_t] - z1) < 1e-6:
                    inferred_ent = e
                    break
    
    match_str = 'OK' if inferred_ent == ds_ent else f'MISMATCH(z->ent{inferred_ent})'
    print(f"  Obs{i}: ds(ent={ds_ent}, t={ds_t}) type={row_type} "
          f"Z1={z1:.4f} Z2={z2:.4f} Z3={z3:.4f} Z4={z4:.4f} "
          f"Z5={z5:.4f} Z6={z6:.4f} Z8={z8:.4f} Z9={z9:.4f} "
          f"-> {match_str}")
