import pandas as pd, numpy as np, duckdb
Q=pd.read_parquet('../hf_data/a13_q.parquet')
def side(Q):
    up=Q[Q.ask1.notna()&(Q.ask_ts-Q['T']<=60)].assign(tok=1,px=lambda d:d.ask1,win=lambda d:d.win1,P2=lambda d:d.P,ts=lambda d:d.ask_ts)
    dn=Q[Q.bid1.notna()&(Q.bid_ts-Q['T']<=60)].assign(tok=2,px=lambda d:1-d.bid1,win=lambda d:1-d.win1,P2=lambda d:1-d.P,ts=lambda d:d.bid_ts)
    t=pd.concat([up,dn]); t['r']=t.win/t.px-1-.07*(1-t.px); return t
t=side(Q)
sel=t[(t.P2>.99)&(t.px>.97)&(t.px<=.99)]
for D in [2,10]:
    s=sel[sel.delay==D]
    m=s.groupby('market_id').r.mean()
    rng=np.random.default_rng(0); a=m.to_numpy(); bs=[a[rng.integers(0,len(a),len(a))].mean() for _ in range(2000)]
    print(f'delay {D}: snaps {len(s)} markets {len(m)} roi/market {a.mean():.4f} CI [{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}] lossrate {(s.win==0).mean():.4f}')
s=sel[sel.delay==2].copy(); s['month']=pd.to_datetime(s['T'],unit='s').dt.to_period('M')
print(s.groupby('month').agg(n=('r','size'),px=('px','mean'),win=('win','mean'),roi=('r','mean')).round(4).to_string())
print(s.groupby(['coin','w']).agg(n=('r','size'),roi=('r','mean')).round(4).to_string())
print(s.groupby('left').agg(n=('r','size'),px=('px','mean'),win=('win','mean'),roi=('r','mean')).round(4).to_string())
# thresholds grid
for pth in [.98,.99,.995]:
  for lo,hi in [(.95,.97),(.97,.98),(.98,.99),(.99,.995)]:
    x=t[(t.delay==2)&(t.P2>pth)&(t.px>lo)&(t.px<=hi)]
    print(pth,lo,hi, x.groupby('period').agg(n=('r','size'),roi=('r','mean'),se=('r','sem')).round(4).to_dict('index'))
# capacity: taker USD that hit same side at <= entry px within 60s after entry
c=duckdb.connect(); c.execute("SET memory_limit='3GB'")
c.register('s',s[['market_id','tok','T','ts','px']])
c.execute("create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('../hf_data/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from s)")
cap=c.execute("""select s.market_id, s.T, s.tok, sum(case when s.tok=1 and f.buy1 and f.p1<=s.px+0.003 then f.usd*1.0 when s.tok=2 and not f.buy1 and 1-f.p1<=s.px+0.003 then f.usd*1.0 else 0 end) cap
 from s join f on f.market_id=s.market_id and f.ts>=s.ts and f.ts<=s.ts+60 group by 1,2,3""").df()
print('capacity USD within 60s at <= entry+0.3c: ', cap.cap.describe(percentiles=[.25,.5,.75,.9]).round(1).to_dict())
