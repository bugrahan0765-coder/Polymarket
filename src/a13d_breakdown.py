import pandas as pd, numpy as np
t=pd.read_parquet('../hf_data/a13c_t.parquet')
x=t[(t.edge>.02)&(t.px>=.95)].sort_values('T').drop_duplicates('market_id').copy()
x['month']=pd.to_datetime(x['T'],unit='s').dt.to_period('M'); x['lag']=x.ts-x['T']
print(x.groupby('month').agg(n=('r','size'),px=('px','mean'),win=('win','mean'),roi=('r','mean'),losses=('win',lambda w:(w==0).sum())).round(4).to_string())
print(x.groupby(['period','coin']).agg(n=('r','size'),roi=('r','mean')).round(4).unstack(0).to_string())
print(x.groupby(['period',pd.cut(x.left,[0,1,2,4,15])],observed=True).agg(n=('r','size'),roi=('r','mean')).round(4).to_string())
print(x.groupby(['period',pd.cut(x.lag,[0,2,5,10,20,31])],observed=True).agg(n=('r','size'),roi=('r','mean'),win=('win','mean'),px=('px','mean')).round(4).to_string())
x[['market_id','T','ts','tok','px','period']].to_parquet('../hf_data/a13d_sel.parquet',compression='zstd')
