import pandas as pd

raw = pd.read_csv('tests/stata/fixtures/inputs/df_panel.csv')
z = pd.read_csv('tests/stata/generate-fixtures/sys_Z.csv')

# Entity data lookup
ents = sorted(raw.entity.unique())
y_e = {int(e): raw[raw.entity==e].y.values for e in ents}
x_e = {int(e): raw[raw.entity==e].x.values for e in ents}
z_e = {int(e): raw[raw.entity==e].z.values for e in ents}

# For level block rows 150-299, identify which entity they correspond to
print("Identifying entity for each level-block row (150-199):")
print("Row  Col6(t=2 only)  Col3(t=2 only)  ->  D.L.y match?  D.x match?  Inferred entity")
print()

for row in range(150, 200):
    zr = z.iloc[row]
    z6 = float(zr['Zmat6'])
    z3 = float(zr['Zmat3'])
    z7 = float(zr['Zmat7'])
    z1 = float(zr['Zmat1'])
    
    if abs(z6) < 1e-10 and abs(z3) < 1e-10 and abs(z1) < 1e-10:
        continue  # all zeros, skip for now
    
    # Try to match by Z6 (y[t-2] for diff, D.L.y for level) at t=2
    # Or Z1 (x[t] for level)
    # Or Z3 (D.x for diff)
    
    # Strategy: try to match each column against all entities
    for t in [0, 1, 2, 3, 4]:
        for e in range(30):
            matches = True
            for j in range(1, 12):
                actual = float(zr['Zmat' + str(j)])
                col_name = 'Zmat' + str(j)
                
                # Compute expected value based on row type and entity
                expected = None
                
                # First determine row type from the row position
                # Rows 150-299: this could be entity E diff or level depending on position
                pos_in_block = row - 150  # 0-149
                entity_in_block = pos_in_block // 10  # 0-14
                is_diff = (pos_in_block % 10) < 5  # 0-4 = diff, 5-9 = level
                t_in_block = pos_in_block % 5 if is_diff else (pos_in_block % 10) - 5
                t_val = t_in_block
                
                e_actual = entity_in_block + 15  # entities 15-29
                
                if is_diff:
                    # Diff row
                    if j == 3:  # D.x
                        if t_val >= 1:
                            expected = x_e[e_actual][t_val] - x_e[e_actual][t_val-1]
                        else:
                            expected = 0.0
                    elif j == 4:  # D.z
                        if t_val >= 1:
                            expected = z_e[e_actual][t_val] - z_e[e_actual][t_val-1]
                        else:
                            expected = 0.0
                    elif j == 6:  # y[t-2]
                        if t_val >= 2:
                            expected = y_e[e_actual][t_val-2]
                        else:
                            expected = 0.0
                    elif j == 8:  # y[t-3]
                        if t_val >= 3:
                            expected = y_e[e_actual][t_val-3]
                        else:
                            expected = 0.0
                    elif j == 9:  # y[t-4]
                        if t_val >= 4:
                            expected = y_e[e_actual][t_val-4]
                        else:
                            expected = 0.0
                    else:
                        expected = 0.0
                else:
                    # Level row
                    if j == 1:  # x[t]
                        expected = x_e[e_actual][t_val] if t_val < 5 else 0.0
                    elif j == 2:
                        expected = z_e[e_actual][t_val] if t_val < 5 else 0.0
                    elif j == 5:
                        expected = 1.0
                    elif j == 7:  # D.L.y
                        if t_val >= 2:
                            expected = y_e[e_actual][t_val-1] - y_e[e_actual][t_val-2]
                        else:
                            expected = 0.0
                    elif j == 11:  # DL.L.y
                        if t_val >= 3:
                            expected = y_e[e_actual][t_val-2] - y_e[e_actual][t_val-3]
                        else:
                            expected = 0.0
                    else:
                        expected = 0.0
                
                if expected is not None and abs(actual - expected) > 1e-6:
                    matches = False
                    break
            
            if matches:
                non_zero = [j for j in range(1,12) if abs(float(zr['Zmat'+str(j)])) > 1e-10]
                print(f"  Row{row}: pos_in_block={pos_in_block} e={e_actual} "
                      f"{'diff' if is_diff else 'level'} t={t_val} "
                      f"non-zero cols={non_zero} -> MATCH!")
                break
    
    if matches:
        break  # Only show first match
