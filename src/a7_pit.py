"""Step 7: point-in-time (look-ahead-free) strategy scan over a 5% sample of ALL Polymarket markets.
Decision time T = scheduled end_date - horizon (known in advance). Signal = last fill price before T (token1 terms).
Entry = first real taker fill after T + 30 s: token1 ask (taker buy of token1) or token2 ask (1 - token1 price of a taker sell).
Hold to resolution, taker fee by category. Selection check: same cell with entry = signal + 1c on ALL markets with a signal.
Run: python src/a7_pit.py <hf_data_dir>
"""
import sys, json, duckdb, numpy as np, pandas as pd
from analysis_lib import load_meta, FEE_RATE

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
HORIZONS = {'5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400, '3d': 259200, '7d': 604800}
SPLIT = pd.Timestamp('2026-05-01', tz='UTC').timestamp()
DELAY, MAX_WAIT = 30, 900

meta = load_meta(HF)
meta = meta[pd.to_numeric(meta['market_id'], errors='coerce') % 20 == 0][['market_id', 'cat', 'win1', 'end_ts', 'neg_risk', 'volume']]
c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3'); c.execute("SET temp_directory='../hf_data/duck_tmp'")
c.register('meta_df', meta)
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{HF}/ext/sample_trades/*.parquet', union_by_name=true) order by market_id, ts")
c.execute("create table span as select market_id, min(ts) t0, max(ts) t1 from f group by market_id")
snaps = ' union all '.join(f"select market_id, '{k}' horizon, cast(end_ts as bigint) - {v} T from meta_df" for k, v in HORIZONS.items())
c.execute(f"create table s as select q.* from ({snaps}) q join span using (market_id) where q.T > span.t0")
c.execute("create table buys as select * from f where buy1")
c.execute("create table sells as select * from f where not buy1")
d = c.execute(f"""
select s.market_id, s.horizon, s.T, sig.p1 sig, sig.ts sig_ts,
       b.p1 ask1, b.ts ask_ts, b.usd ask_usd, x.p1 bid1, x.ts bid_ts
from s
asof left join f sig on s.market_id = sig.market_id and s.T > sig.ts
asof left join buys b on s.market_id = b.market_id and s.T + {DELAY} <= b.ts
asof left join sells x on s.market_id = x.market_id and s.T + {DELAY} <= x.ts
""").df()
d = d.merge(meta, on='market_id')
d = d[d['sig'].notna() & (d['T'] - d['sig_ts'] <= 6 * 3600)]
d['ask_ok'] = d['ask1'].notna() & (d['ask_ts'] - d['T'] <= MAX_WAIT)
d['bid_ok'] = d['bid1'].notna() & (d['bid_ts'] - d['T'] <= MAX_WAIT)
d['period'] = np.where(d['T'] < SPLIT, 'H1', 'H2')
d['rate'] = d['cat'].map(FEE_RATE)
print(f'{len(d):,} snapshots with a fresh signal on {d.market_id.nunique():,} markets')


def roi(px, win, rate):
    return win / px - 1 - rate * (1 - px)  # per $1 staked incl. fee rate*p*(1-p) per share


BINS = [0, .03, .1, .2, .3, .4, .5, .6, .7, .8, .9, .97, 1.0001]
d['bin'] = pd.cut(d['sig'], BINS, right=False)
rows = []
for side in ['tok1', 'tok2']:
    if side == 'tok1':
        ok, px, win, naive = d['ask_ok'], d['ask1'], d['win1'], (d['sig'] + .01).clip(upper=.999)
    else:
        ok, px, win, naive = d['bid_ok'], 1 - d['bid1'], 1 - d['win1'], (1 - d['sig'] + .01).clip(upper=.999)
    x = d.assign(px=px, win=win, ok=ok, naive=naive)
    x = x[(x['naive'] > .005)]
    x['r_naive'] = roi(x['naive'], x['win'], x['rate'])
    x['r_real'] = np.where(x['ok'] & x['px'].between(.005, .995), roi(x['px'].clip(.005, .995), x['win'], x['rate']), np.nan)
    g = x.groupby(['cat', 'horizon', 'bin', 'period'], observed=True).agg(
        n=('r_naive', 'size'), filled=('r_real', 'count'), roi_naive_all=('r_naive', 'mean'),
        roi_naive_filled=('r_naive', lambda r: r[x.loc[r.index, 'r_real'].notna()].mean()),
        roi_real=('r_real', 'mean'), se_real=('r_real', 'sem'), slip=('px', lambda p: (p - x.loc[p.index, 'naive'] + .01)[x.loc[p.index, 'ok']].median())).reset_index()
    g['side'] = side; rows.append(g)
res = pd.concat(rows); res['bin'] = res['bin'].astype(str)
res.to_csv('results/a7_pit_cells.csv', index=False)
p = res.pivot_table(index=['side', 'cat', 'horizon', 'bin'], columns='period', values=['filled', 'roi_real', 'se_real', 'roi_naive_all', 'slip']).dropna()
p.columns = [f'{a}_{b}' for a, b in p.columns]
if 'roi_real_H2' not in p: sys.exit('no H2 data yet')
good = p[(p.roi_real_H1 > 0) & (p.roi_real_H2 > 0) & (p.roi_naive_all_H2 > 0) & (p.filled_H1 >= 50) & (p.filled_H2 >= 50)].copy()
good['t_H2'] = good.roi_real_H2 / good.se_real_H2
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200)
print('\n=== cells positive in H1 and H2 at REAL entry, and positive at naive entry on ALL markets in H2 ===\n' +
      good.sort_values('t_H2', ascending=False).round(4).head(50).to_string())
json.dump(good.round(4).reset_index().to_dict('records'), open('results/a7_pit_good.json', 'w'), indent=1, default=str)
