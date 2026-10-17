"""Step 8b: copy-trading with exact fills. a8 showed wallet skill persists in total PnL, but their FIRST BUYS lose at
next-hour VWAP (their profit is not from directional entries). Here wallets are selected on what a copier can copy:
the hold-to-resolution return of their first buy (r = win / their avg buy px - 1) on markets ending before SEL_END.
Test on the 5% fill-level sample (markets ending >= TEST_START): copy entry = first real taker fill of the same token
after t_entry + delay (<= 1 h), fee by category. Baseline: same procedure for all wallets with >= 20 markets.
Run: python src/a8b_copy_exact.py <hf_data_dir>
"""
import sys, json, duckdb, numpy as np, pandas as pd
from analysis_lib import load_meta, FEE_RATE

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SEL_END = pd.Timestamp('2026-04-01', tz='UTC').timestamp()
TEST_START = pd.Timestamp('2026-05-01', tz='UTC').timestamp()
c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3'); c.execute("SET temp_directory='../hf_data/duck_tmp'")
c.execute('SET preserve_insertion_order=false')
c.execute(f"""create view fb as select wallet, market_id, cat, win1, end_ts, bought, taker_share,
  case when coalesce(tb1, 9e18) <= coalesce(tb2, 9e18) then 1 else 2 end tok,
  least(coalesce(tb1, 9e18), coalesce(tb2, 9e18)) t_entry,
  case when coalesce(tb1, 9e18) <= coalesce(tb2, 9e18) then px1 else px2 end px
  from read_parquet('{HF}/wallet_pm.parquet') where bought > 1 and (tb1 is not null or tb2 is not null)""")
# per-wallet selection stats on first-buy hold returns (crypto up/down excluded: 5-15 min markets, not copyable)
c.execute(f"""create table ws as select wallet, count(*) n, avg(r) mu, stddev_samp(r) sd, avg(taker_share) ts, median(bought) med from (
  select *, (case when tok = 1 then win1 else 1 - win1 end) / greatest(least(px, .99), .01) - 1 r from fb
  where end_ts < {SEL_END} and px between .02 and .98 and not (cat = 'crypto' and end_ts - t_entry < 7200))
  group by wallet having count(*) >= 20""")
c.execute("alter table ws add column t double"); c.execute("update ws set t = mu / (sd / sqrt(n))")
print(c.execute("select count(*), sum((t >= 3)::int), sum((t >= 2)::int) from ws").fetchall())
c.execute(f"""create table ent as select fb.*, ws.t from fb join ws using (wallet)
  where fb.end_ts >= {TEST_START} and hash(fb.market_id) is not null and try_cast(fb.market_id as bigint) % 20 = 0 and fb.t_entry < fb.end_ts""")
c.execute("create table mids as select distinct market_id from ent")
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{HF}/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from mids) order by market_id, ts")
print(c.execute("select count(*), count(distinct wallet), count(distinct market_id) from ent").fetchall())
rows = []
for delay in [10, 60, 300]:
    q = c.execute(f"""select e.wallet, e.market_id, e.cat, e.tok, e.win1, e.t, e.px their_px, e.end_ts, e.t_entry, b.p1 b1, b.ts bts, x.p1 x1, x.ts xts from ent e
      asof left join (select * from f where buy1) b on e.market_id = b.market_id and e.t_entry + {delay} <= b.ts
      asof left join (select * from f where not buy1) x on e.market_id = x.market_id and e.t_entry + {delay} <= x.ts""").df()
    q['cpx'] = np.where(q.tok == 1, q.b1, 1 - q.x1); q['cts'] = np.where(q.tok == 1, q.bts, q.xts)
    q['win'] = np.where(q.tok == 1, q.win1, 1 - q.win1)
    q = q[q.cpx.between(.02, .98) & (q.cts - q.t_entry <= 3600) & (q.cts < q.end_ts)]
    q['r'] = q.win / q.cpx - 1 - q.cat.map(FEE_RATE) * (1 - q.cpx)
    q['short'] = (q.cat == 'crypto') & (q.end_ts - q.t_entry < 7200)
    q = q[~q.short]
    for name, sel in [('t>=3', q.t >= 3), ('t>=2', q.t >= 2), ('t>=5', q.t >= 5), ('all', q.t > -99), ('t<=-2', q.t <= -2)]:
        s = q[sel]
        mk = s.groupby('market_id').agg(r=('r', 'mean'), cat=('cat', 'first'))  # one copy per market
        if len(mk) < 30: continue
        a = mk.r.to_numpy(); rng = np.random.default_rng(0); bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)]
        rows.append(dict(delay=delay, sel=name, markets=len(mk), roi=round(a.mean(), 4), lo=round(np.percentile(bs, 2.5), 4), hi=round(np.percentile(bs, 97.5), 4),
                         by_cat={k: (int(v['size']), round(v['mean'], 4)) for k, v in mk.groupby('cat').r.agg(['size', 'mean']).iterrows()}))
        print(rows[-1], flush=True)
json.dump(rows, open('results/a8b_copy_exact.json', 'w'), indent=1, default=str)
