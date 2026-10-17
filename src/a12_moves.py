"""Step 12: over/under-reaction after big price moves, all categories, 5% sample, point-in-time.
Event: hourly last price (token1) moves by >= JUMP vs the previous hour's last price. T = end of that hour (known then).
Trade at the first real taker fill after T+30s (<= 15 min): 'cont' buys the side that rose, 'rev' buys the side that fell.
Hold to resolution, taker fee by category. One event per market per day (first one) to limit overlap.
Run: python src/a12_moves.py <hf_data_dir>
"""
import sys, json, duckdb, numpy as np, pandas as pd
from analysis_lib import load_meta, FEE_RATE

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SPLIT = pd.Timestamp('2026-05-01', tz='UTC').timestamp()
DELAY, MAX_WAIT = 30, 900

meta = load_meta(HF)
meta = meta[pd.to_numeric(meta['market_id'], errors='coerce') % 20 == 0][['market_id', 'cat', 'win1', 'end_ts']]
c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3'); c.execute("SET temp_directory='../hf_data/duck_tmp'")
c.register('meta_df', meta)
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{HF}/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from meta_df) order by market_id, ts")
c.execute("""create table hl as select market_id, ts // 3600 * 3600 h, arg_max(p1, ts) p, sum(usd) usd from f group by 1, 2""")
c.execute("""create table ev as select * from (
  select market_id, h + 3600 T, p, usd, p - lag(p) over w dp, h - lag(h) over w gap from hl window w as (partition by market_id order by h)
) where abs(dp) >= 0.08 and gap <= 3 * 3600 and p between 0.03 and 0.97""")
c.execute("create table ev1 as select * from (select *, row_number() over (partition by market_id, T // 86400 order by T) rn from ev) where rn = 1")
d = c.execute(f"""select e.*, b.p1 ask1, b.ts ask_ts, x.p1 bid1, x.ts bid_ts from ev1 e
  asof left join (select * from f where buy1) b on e.market_id = b.market_id and e.T + {DELAY} <= b.ts
  asof left join (select * from f where not buy1) x on e.market_id = x.market_id and e.T + {DELAY} <= x.ts""").df()
d = d.merge(meta, on='market_id')
d = d[d['end_ts'] > d['T'] + 600]
d['rate'] = d['cat'].map(FEE_RATE)
d['period'] = np.where(d['T'] < SPLIT, 'H1', 'H2')
d['left'] = pd.cut((d['end_ts'] - d['T']) / 3600, [0, 6, 48, 24 * 14, 1e6], labels=['<6h', '6-48h', '2-14d', '>14d'])
d['size'] = pd.cut(d['dp'].abs(), [.08, .15, .3, 1.01], right=False)
print(f'{len(d):,} move events on {d.market_id.nunique():,} markets')

ok1 = d['ask1'].notna() & (d['ask_ts'] - d['T'] <= MAX_WAIT)
ok2 = d['bid1'].notna() & (d['bid_ts'] - d['T'] <= MAX_WAIT)
up = d['dp'] > 0
rows = []
for strat in ['cont', 'rev']:
    buy_tok1 = up if strat == 'cont' else ~up
    px = np.where(buy_tok1, d['ask1'], 1 - d['bid1'])
    ok = np.where(buy_tok1, ok1, ok2)
    win = np.where(buy_tok1, d['win1'], 1 - d['win1'])
    naive = np.where(buy_tok1, d['p'] + .01, 1 - d['p'] + .01)
    x = d.assign(px=px, ok=ok, win=win, naive=naive)
    x['r_naive'] = x['win'] / x['naive'] - 1 - x['rate'] * (1 - x['naive'])
    x = x[x['ok'] & x['px'].between(.01, .99)]
    x['r'] = x['win'] / x['px'] - 1 - x['rate'] * (1 - x['px'])
    g = x.groupby(['cat', 'left', 'size', 'period'], observed=True).agg(n=('r', 'size'), px=('px', 'mean'), win=('win', 'mean'),
                                                                      roi=('r', 'mean'), se=('r', 'sem'), roi_naive=('r_naive', 'mean')).reset_index()
    g['strat'] = strat; rows.append(g)
res = pd.concat(rows)
res.to_csv('results/a12_moves_cells.csv', index=False)
p = res.pivot_table(index=['strat', 'cat', 'left', 'size'], columns='period', values=['n', 'roi', 'se', 'px'], observed=True).dropna()
p.columns = [f'{a}_{b}' for a, b in p.columns]
good = p[(p.roi_H1 > 0) & (p.roi_H2 > 0) & (p.n_H1 >= 40) & (p.n_H2 >= 40)].copy()
good['t_H2'] = good.roi_H2 / good.se_H2
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print('\n=== overall by strategy x category x period ===')
print(res.groupby(['strat', 'cat', 'period']).apply(lambda t: pd.Series(dict(n=t.n.sum(), roi=(t.roi * t.n).sum() / t.n.sum()))).round(4).unstack('period').to_string())
print('\n=== cells positive in both halves (n>=40 each) ===\n' + good.sort_values('t_H2', ascending=False).round(4).to_string())
json.dump(good.round(4).reset_index().astype(str).to_dict('records'), open('results/a12_moves_good.json', 'w'), indent=1)
