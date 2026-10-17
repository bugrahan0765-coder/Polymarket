"""Step 9: weather markets at many decision times (local clock), point-in-time from on-chain fills.
T = local hour on D-1 or D (target day). Signal = last fill before T (token1 = Yes). Entry = first taker fill after T+30s
(within 15 min): YES at the Yes ask, or NO at 1 - (price of a taker Yes-sell). Hold to resolution, weather taker fee.
Cells: decision time x price bin x side x period; also 'favourite only'. Run: python src/a9_weather_intraday.py <ext_dir>
"""
import sys, json, glob, datetime as dt
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
from analysis_lib import load_markets
from common import TZ

EXT = sys.argv[1] if len(sys.argv) > 1 else '../hf_data/ext'
TIMES = {'D-1 12h': (-1, 12), 'D 00h': (0, 0), 'D 06h': (0, 6), 'D 09h': (0, 9), 'D 12h': (0, 12), 'D 14h': (0, 14),
         'D 16h': (0, 16), 'D 18h': (0, 18), 'D 21h': (0, 21)}
SPLIT = pd.Timestamp('2026-06-15', tz='UTC').timestamp()
FEE_START = pd.Timestamp('2026-03-30', tz='UTC').timestamp()

m = load_markets()
m = m[m['yes'].notna()].copy(); m['market_id'] = m['market_id'].astype(str)
ev = m.drop_duplicates('event_id')[['event_id', 'city', 'target_date']]
rows = []
for e in ev.itertuples():
    tz = ZoneInfo(TZ[e.city])
    for k, (dd, hh) in TIMES.items():
        d = e.target_date + dt.timedelta(days=dd)
        rows.append((e.event_id, k, int(dt.datetime(d.year, d.month, d.day, hh, tzinfo=tz).timestamp())))
S = pd.DataFrame(rows, columns=['event_id', 'tname', 'T']).merge(m[['event_id', 'market_id', 'yes', 'bucket']], on='event_id')

c = duckdb.connect()
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{EXT}/weather_trades/*.parquet', union_by_name=true) order by market_id, ts")
c.register('s_df', S)
d = c.execute("""select s.*, sig.p1 sig, sig.ts sig_ts, b.p1 ask1, b.ts ask_ts, x.p1 bid1, x.ts bid_ts from s_df s
  asof left join f sig on s.market_id = sig.market_id and s.T > sig.ts
  asof left join (select * from f where buy1) b on s.market_id = b.market_id and s.T + 30 <= b.ts
  asof left join (select * from f where not buy1) x on s.market_id = x.market_id and s.T + 30 <= x.ts""").df()
d = d[d['sig'].notna() & (d['T'] - d['sig_ts'] <= 12 * 3600)]
d['rank'] = d.groupby(['event_id', 'tname'])['sig'].rank(ascending=False, method='first')
d['period'] = np.where(d['T'] < SPLIT, 'H1', 'H2')
print(f'{len(d):,} bucket snapshots')


def roi(px, win, T):
    rate = np.where(T >= FEE_START, .05, 0.)
    return win / px - 1 - rate * (1 - px)


BINS = [0, .02, .05, .1, .2, .35, .5, .65, .8, .9, .95, .98, 1.0001]
d['bin'] = pd.cut(d['sig'], BINS, right=False)
out = []
for side in ['YES', 'NO']:
    if side == 'YES':
        ok = d['ask1'].notna() & (d['ask_ts'] - d['T'] <= 900); px = d['ask1']; win = d['yes']
    else:
        ok = d['bid1'].notna() & (d['bid_ts'] - d['T'] <= 900); px = 1 - d['bid1']; win = 1 - d['yes']
    x = d[ok & px.between(.01, .995)].assign(px=px, win=win)
    x['r'] = roi(x['px'], x['win'], x['T'])
    for who, xx in [('all', x), ('fav', x[x['rank'] == 1])]:
        g = xx.groupby(['tname', 'bin', 'period'], observed=True).agg(n=('r', 'size'), px=('px', 'mean'), win=('win', 'mean'),
                                                                     roi=('r', 'mean'), se=('r', 'sem')).reset_index()
        g['side'] = side; g['who'] = who; out.append(g)
res = pd.concat(out); res['bin'] = res['bin'].astype(str)
res.to_csv('results/a9_weather_intraday_cells.csv', index=False)
p = res.pivot_table(index=['who', 'side', 'tname', 'bin'], columns='period', values=['n', 'roi', 'se', 'win', 'px']).dropna()
p.columns = [f'{a}_{b}' for a, b in p.columns]
good = p[(p.roi_H1 > 0) & (p.roi_H2 > 0) & (p.n_H1 >= 60) & (p.n_H2 >= 60)].copy()
good['t_H2'] = good.roi_H2 / good.se_H2
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('\n=== weather cells positive in both periods (split 2026-06-15), real entry, fee ===\n' +
      good.sort_values('t_H2', ascending=False).round(4).head(50).to_string())
json.dump(good.round(4).reset_index().to_dict('records'), open('results/a9_weather_intraday.json', 'w'), indent=1, default=str)
