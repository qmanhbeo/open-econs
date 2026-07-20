"""Read the toy sysgmm fixture."""
import pandas as pd
df = pd.read_stata(r'C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\expected\toy_sysgmm.dta')
for _, row in df.iterrows():
    print(f"{row['name']:30s} = {row['value']:.10f}")
