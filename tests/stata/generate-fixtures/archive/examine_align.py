import pandas as pd, numpy as np

raw = pd.read_csv('tests/stata/fixtures/inputs/df_panel.csv')
z = pd.read_csv('tests/stata/generate-fixtures/sys_Z.csv')

print('Raw:', raw.shape, 'Z:', z.shape)
print()

for i in range(20):
    r = raw.iloc[i]
    zr = z.iloc[i]
    em = '=' if int(r.entity) == int(zr.entity) else '!'
    tm = '=' if int(r.time) == int(zr.time) else '!'
    vals = []
    for j in range(1, 12):
        v = zr['Zmat' + str(j)]
        vals.append(f'{v:.4f}')
    print(f'  row{i}: raw({int(r.entity)},{int(r.time)}) Z({int(zr.entity)},{int(zr.time)}) {em}{tm}  ' + ' '.join(vals))

# Also check the level block
print()
print('=== Level block (rows 150-169) ===')
for i in range(150, 170):
    zr = z.iloc[i]
    vals = []
    for j in range(1, 12):
        v = zr['Zmat' + str(j)]
        vals.append(f'{v:.4f}')
    ent_str = '' if pd.isna(zr['entity']) else str(int(zr['entity']))
    print(f'  row{i}: entity={ent_str} time={zr["time"]}  ' + ' '.join(vals))
