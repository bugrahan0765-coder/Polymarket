"""Step 8: smart-money study. Do some wallets win persistently, and can they be copied profitably?
Per (wallet, market): net shares of each token and cash flow from CLOB fills (hf_extract_wallets.py).
PnL at resolution = net1*win1 + net2*(1-win1) + cash (also correct for split/merge round trips).
Select wallets on markets ending before SEL_END (n >= 20 markets, t-stat >= 3, mostly taker), then in markets ending
after TEST_START copy their first buy with a delay at the next-hour taker VWAP of that token (hourly data), fee incl.
Run: python src/a8_wallets.py <hf_data_dir>
"""
import sys, json, duckdb, numpy as np, pandas as pd
from analysis_lib import load_meta, FEE_RATE

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SEL_END = pd.Timestamp('2026-04-01', tz='UTC').timestamp()
TEST_START = pd.Timestamp('2026-05-01', tz='UTC').timestamp()

meta = load_meta(HF)[['market_id', 'cat', 'win1', 'end_ts']]
c = duckdb.connect(); c.execute("SET memory_limit='9GB'"); c.execute("SET threads=4")
c.register('meta_df', meta)
c.execute(f"""create table pos as
select wallet, market_id,
  sum(case when tok=1 and buy then shares when tok=1 then -shares else 0 end) net1,
  sum(case when tok=2 and buy then shares when tok=2 then -shares else 0 end) net2,
  sum(case when buy then -usd else usd end) cash,
  sum(case when buy then usd else 0 end) bought,
  sum(case when role='T' then usd else 0 end) / sum(usd) taker_share,
  min(case when buy and tok=1 then t_first end) tb1, min(case when buy and tok=2 then t_first end) tb2,
  sum(case when buy and tok=1 then usd end) / nullif(sum(case when buy and tok=1 then shares end), 0) px1,
  sum(case when buy and tok=2 then usd end) / nullif(sum(case when buy and tok=2 then shares end), 0) px2
from read_parquet('{HF}/ext/wallets/*.parquet') group by wallet, market_id""")
c.execute("""create table pm as select p.*, m.cat, m.win1, m.end_ts, p.net1*m.win1 + p.net2*(1-m.win1) + p.cash pnl
             from pos p join meta_df m using (market_id)""")
print(c.execute("select count(*), count(distinct wallet), count(distinct market_id) from pm").fetchall())
# exclude infrastructure addresses (exchange / adapters) by sheer breadth
c.execute("create table infra as select wallet from pm group by wallet having count(*) > 200000")
print('infra wallets excluded:', c.execute('select * from infra').fetchall())

w = c.execute(f"""select wallet, count(*) n, sum(pnl) pnl, sum(bought) bought, avg(pnl) mu, stddev_samp(pnl) sd, avg(taker_share) taker_share,
                  median(bought) med_bet
                  from pm where end_ts < {SEL_END} and bought > 1 and wallet not in (select wallet from infra) group by wallet having count(*) >= 20""").df()
w['t'] = w['mu'] / (w['sd'] / np.sqrt(w['n'])); w['roi'] = w['pnl'] / w['bought']
print(f'{len(w):,} wallets with >= 20 resolved markets before selection cut')

# out-of-sample persistence: did the selected wallets keep winning (their own trades) in the test period?
wt = c.execute(f"""select wallet, count(*) n2, sum(pnl) pnl2, sum(bought) bought2 from pm where end_ts >= {TEST_START} and bought > 1 group by wallet""").df()
w = w.merge(wt, on='wallet', how='left'); w['roi2'] = w['pnl2'] / w['bought2']
for name, sel in [('t>=3 taker>=0.5 roi>=0.05', (w.t >= 3) & (w.taker_share >= .5) & (w.roi >= .05)),
                  ('t>=3 any', w.t >= 3), ('t>=5 any', w.t >= 5), ('top100 pnl', w.pnl.rank(ascending=False) <= 100),
                  ('t<=-3 (losers)', w.t <= -3), ('all', w.t.notna())]:
    s = w[sel & w.n2.notna()]
    print(f'{name:28s} wallets={sel.sum():5d} active_in_test={len(s):5d} test_pnl={s.pnl2.sum():14,.0f} test_roi={s.pnl2.sum() / s.bought2.sum():.4f} '
          f'share_positive={(s.pnl2 > 0).mean():.3f}')

# copy-trading test: selected directional wallets' first buys in test-period markets
smart = w[(w.t >= 3) & (w.taker_share >= .5) & (w.roi >= .05)][['wallet']]
c.register('smart_df', smart)
e = c.execute(f"""select pm.wallet, pm.market_id, pm.cat, pm.win1, pm.end_ts, pm.bought,
                  case when coalesce(pm.tb1, 9e18) <= coalesce(pm.tb2, 9e18) then 1 else 2 end tok,
                  least(coalesce(pm.tb1, 9e18), coalesce(pm.tb2, 9e18)) t_entry,
                  case when coalesce(pm.tb1, 9e18) <= coalesce(pm.tb2, 9e18) then pm.px1 else pm.px2 end their_px
                  from pm join smart_df using (wallet) where pm.end_ts >= {TEST_START} and pm.bought > 1""").df()
e = e[e['t_entry'] < 9e17]
c.register('e_df', e)
c.execute(f"create view h as select * from read_parquet('{HF}/ext/hourly/*.parquet')")
rows = []
for delay in [60, 300, 3600]:
    q = c.execute(f"""select e_df.*, h.vwap, h.hour from e_df
                      asof left join (select * from h where buy1) h1 on e_df.market_id = h1.market_id and e_df.t_entry + {delay} <= h1.hour + 3600
                      left join (select market_id, hour, vwap from h where buy1) h on h.market_id = h1.market_id and h.hour = h1.hour""").df()
    q2 = c.execute(f"""select e_df.market_id, e_df.wallet, h2.vwap vwap_s from e_df
                       asof left join (select * from h where not buy1) h2 on e_df.market_id = h2.market_id and e_df.t_entry + {delay} <= h2.hour + 3600""").df()
    q = q.merge(q2, on=['market_id', 'wallet'], how='left')
    q['px'] = np.where(q['tok'] == 1, q['vwap'], 1 - q['vwap_s'])
    q['win'] = np.where(q['tok'] == 1, q['win1'], 1 - q['win1'])
    q = q[q['px'].between(.01, .99) & (q['px'] < 1)]
    q['r'] = q['win'] / q['px'] - 1 - q['cat'].map(FEE_RATE) * (1 - q['px'])
    q['r_theirs'] = q['win'] / q['their_px'].clip(.01, .99) - 1
    # one copy per market (several smart wallets may enter the same market): average
    mk = q.groupby('market_id').agg(r=('r', 'mean'), r_theirs=('r_theirs', 'mean'), cat=('cat', 'first'), px=('px', 'mean'))
    rng = np.random.default_rng(0); arr = mk['r'].to_numpy()
    bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(2000)]
    rows.append(dict(delay_s=delay, markets=len(mk), roi_copy=round(arr.mean(), 4), ci=[round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)],
                     roi_theirs=round(mk['r_theirs'].mean(), 4), by_cat=mk.groupby('cat')['r'].agg(['size', 'mean']).round(4).to_dict('index')))
    print(rows[-1])
json.dump(dict(copy=rows, n_smart=len(smart)), open('results/a8_wallets.json', 'w'), indent=1, default=str)
