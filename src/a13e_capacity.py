import pandas as pd, numpy as np, duckdb
s=pd.read_parquet('../hf_data/a13d_sel.parquet')
c=duckdb.connect(); c.execute("SET memory_limit='3GB'")
c.register('s',s)
c.execute("create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd, token_amount sh from read_parquet('../hf_data/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from s)") if False else c.execute("create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('../hf_data/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from s)")
cap=c.execute("""select s.market_id, s.period,
 sum(case when f.ts=s.ts then f.usd else 0 end) first_ts_usd,
 sum(case when f.ts<=s.ts+10 and ((s.tok=1 and f.buy1 and f.p1<=s.px+0.003) or (s.tok=2 and not f.buy1 and 1-f.p1<=s.px+0.003)) then f.usd else 0 end) cap10,
 sum(case when ((s.tok=1 and f.buy1 and f.p1<=s.px+0.003) or (s.tok=2 and not f.buy1 and 1-f.p1<=s.px+0.003)) then f.usd else 0 end) cap60
 from s join f on f.market_id=s.market_id and f.ts>=s.ts and f.ts<=s.ts+60 group by 1,2""").df()
for col in ['first_ts_usd','cap10','cap60']:
    print(col, cap.groupby('period')[col].describe(percentiles=[.25,.5,.75,.9]).round(1).to_string())
t=pd.read_parquet('../hf_data/a13c_t.parquet')
x=t[(t.edge>.02)&(t.px>=.95)].sort_values('T').drop_duplicates('market_id')
z=x.merge(cap,on=['market_id','period'])
z['cb']=pd.cut(z.cap10,[-1,1,5,20,100,1e9])
print(z.groupby(['period','cb'],observed=True).agg(n=('r','size'),roi=('r','mean'),win=('win','mean')).round(4).to_string())
for st in [5,20,50,100]:
    z['stake']=z.cap10.clip(upper=st); 
    print(st, z.groupby('period').apply(lambda g: pd.Series(dict(usd=g.stake.sum(), pnl=(g.stake*g.r).sum(), roi=(g.stake*g.r).sum()/g.stake.sum()))).round(3).to_dict('index'))
days={'H1':(pd.Timestamp('2026-06-01')-pd.Timestamp('2025-10-01')).days,'H2':(pd.Timestamp('2026-10-04')-pd.Timestamp('2026-06-01')).days}
print('days',days)
