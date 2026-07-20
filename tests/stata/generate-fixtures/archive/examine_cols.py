import pandas as pd, numpy as np
raw = pd.read_csv('tests/stata/fixtures/inputs/df_panel.csv')
z = pd.read_csv('tests/stata/generate-fixtures/sys_Z.csv')

print("All 11 Z columns for dataset obs 5-9:")
for i in range(5, 10):
    zr = z.iloc[i]
    parts = [f'Obs{i}: ds({int(zr.entity)},{int(zr.time)}) ']
    for j in range(1, 12):
        parts.append(f'Z{j}={float(zr["Zmat"+str(j)]):.6f}')
    print(' '.join(parts))

print()
print("All 11 Z columns for dataset obs 150-154:")
for i in range(150, 155):
    zr = z.iloc[i]
    parts = [f'Obs{i}: ']
    for j in range(1, 12):
        parts.append(f'Z{j}={float(zr["Zmat"+str(j)]):.6f}')
    print(' '.join(parts))

print()
print("All 11 Z columns for dataset obs 7 (entity 0 level t=2):")
zr = z.iloc[7]
for j in range(1, 12):
    v = float(zr["Zmat"+str(j)])
    print(f'  Z{j}={v:.10f}')
