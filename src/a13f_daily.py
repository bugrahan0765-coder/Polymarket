import pandas as pd, numpy as np, duckdb
t=pd.read_parquet('../hf_data/a13c_t.parquet')
x=t[(t.edge>.02)&(t.px>=.95)].sort_values('T').drop_duplicates('market_id').copy()
c=duckdb.connect(); c.execute("SET memory_limit='3GB'")
c.register('s',x[['market_id','tok','ts','px']])
c.execute("create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('../hf_data/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from s)")
cap=c.execute("""select s.market_id, sum(case when ((s.tok=1 and f.buy1 and f.p1<=s.px+0.003) or (s.tok=2 and not f.buy1 and 1-f.p1<=s.px+0.003)) then f.usd else 0 end) cap10
 from s join f on f.market_id=s.market_id and f.ts>=s.ts and f.ts<=s.ts+10 group by 1""").df()
x=x.merge(cap,on='market_id',how='left').fillna({'cap10':0})
x['day']=pd.to_datetime(x['T'],unit='s').dt.date
h2=x[x.period=='H2']; ndays=(pd.Timestamp('2026-10-04')-pd.Timestamp('2026-06-01')).days
print('H2 days',ndays,'sample trades',len(h2),'-> full-universe trades/day ~',round(len(h2)*20/ndays,1))
print('trades/day by left-minutes: avg minutes left', round(h2.left.mean(),2))
for st in [5,10,20]:
    h=h2.assign(stake=h2.cap10.clip(upper=st)); h=h[h.stake>0]
    d=h.groupby('day').apply(lambda g:(g.stake*g.r).sum()).reindex(pd.date_range('2026-06-01','2026-10-03').date,fill_value=0)*20
    print(f'stake<=${st}: trades with capacity {len(h)*20/ndays:.0f}/day, avg stake ${h.stake.mean():.2f}, turnover ${h.stake.sum()*20/ndays:.0f}/day, '
          f'pnl/day mean ${d.mean():.2f} median ${d.median():.2f}, losing days {(d<0).mean():.0%}, worst day ${d.min():.1f}, 125-day total ${d.sum():.0f}')
