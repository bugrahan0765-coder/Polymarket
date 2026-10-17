"""Step 10: 'evening lock' candidate on weather markets. In the evening of the target day (local time) the daily high is
almost always already reached; buy YES on buckets the market prices >= threshold, hold to resolution.
Point-in-time from on-chain fills: signal = last fill before T, entry = first taker Yes-buy after T+30s (<= 15 min).
Reports per hour x threshold: trades/day, win rate, ROI (fee incl.), by month, every loss, capacity (taker-bought USD
within 30 min at <= entry + 0.5c), capital lock time (T -> market close), and a look-ahead-free entry check.
Run: python src/a10_evening_lock.py <ext_dir>
"""
import sys, json, datetime as dt
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
from analysis_lib import load_markets
from common import TZ

EXT = sys.argv[1] if len(sys.argv) > 1 else '../hf_data/ext'
HOURS = [16, 17, 18, 19, 20, 21, 22]
FEE_START = pd.Timestamp('2026-03-30', tz='UTC').timestamp()

m = load_markets()
m = m[m['yes'].notna()].copy(); m['market_id'] = m['market_id'].astype(str)
ev = m.drop_duplicates('event_id')[['event_id', 'city', 'target_date']]
rows = []
for e in ev.itertuples():
    tz = ZoneInfo(TZ[e.city]); d = e.target_date
    for hh in HOURS:
        rows.append((e.event_id, hh, int(dt.datetime(d.year, d.month, d.day, hh, tzinfo=tz).timestamp())))
S = pd.DataFrame(rows, columns=['event_id', 'hour', 'T']).merge(m[['event_id', 'market_id', 'yes', 'bucket', 'city', 'target_date', 'closed_time']], on='event_id')

c = duckdb.connect()
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{EXT}/weather_trades/*.parquet', union_by_name=true) order by market_id, ts")
c.register('s_df', S[['event_id', 'hour', 'T', 'market_id']])
d = c.execute("""select s.*, sig.p1 sig, sig.ts sig_ts, b.p1 ask1, b.ts ask_ts from s_df s
  asof left join f sig on s.market_id = sig.market_id and s.T > sig.ts
  asof left join (select * from f where buy1) b on s.market_id = b.market_id and s.T + 30 <= b.ts""").df()
d = d.merge(S[['event_id', 'hour', 'market_id', 'yes', 'bucket', 'city', 'target_date', 'closed_time']], on=['event_id', 'hour', 'market_id'])
d = d[d['sig'].notna() & (d['T'] - d['sig_ts'] <= 6 * 3600) & (d['sig'] >= .9)].copy()
# capacity: taker Yes-buy USD within 30 min after T at <= entry + 0.005
c.register('d_df', d[['market_id', 'hour', 'T', 'ask1']])
cap = c.execute("""select d.market_id, d.hour, sum(f.usd) cap_usd from d_df d join f on f.market_id = d.market_id
                   and f.buy1 and f.ts >= d.T + 30 and f.ts <= d.T + 1800 and f.p1 <= d.ask1 + 0.005 group by 1, 2""").df()
d = d.merge(cap, on=['market_id', 'hour'], how='left')
d['close_ts'] = pd.to_datetime(d['closed_time'].str.replace('+00', '+00:00'), errors='coerce', utc=True).map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
d['lock_h'] = (d['close_ts'] - d['T']) / 3600
d['month'] = pd.to_datetime(d['T'], unit='s').dt.to_period('M')
ok = d['ask1'].notna() & (d['ask_ts'] - d['T'] <= 900) & (d['ask1'] < 1)
d['entry'] = np.where(ok, d['ask1'], np.nan)
fee = lambda px, T: np.where(T >= FEE_START, .05 * (1 - px), 0.)
d['r'] = d['yes'] / d['entry'] - 1 - fee(d['entry'], d['T'])
d['r_naive'] = d['yes'] / (d['sig'] + .005).clip(upper=.999) - 1 - fee(d['sig'], d['T'])
days = d['target_date'].nunique()

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
rows = []
for hh in HOURS:
    for th in [.95, .97, .98, .985, .99]:
        x = d[(d['hour'] == hh) & (d['sig'] >= th)]
        xr = x[x['entry'].notna() & (x['entry'] <= .995)]
        if len(xr) < 30: continue
        rows.append(dict(hour=hh, thr=th, n=len(xr), per_day=round(len(xr) / days, 2), win=round(xr['yes'].mean(), 4), losses=int((xr['yes'] == 0).sum()),
                         entry=round(xr['entry'].mean(), 4), roi=round(xr['r'].mean(), 4), se=round(xr['r'].sem(), 4),
                         roi_naive_all=round(x['r_naive'].mean(), 4), cap_med=round(xr['cap_usd'].median(), 1), cap_p25=round(xr['cap_usd'].quantile(.25), 1),
                         lock_h_med=round(xr['lock_h'].median(), 1), roi_per_day=round(xr['r'].mean() / max(xr['lock_h'].median(), 1) * 24, 4)))
g = pd.DataFrame(rows)
print(f'{days} target days\n=== evening lock grid ===\n' + g.to_string(index=False))

best = d[(d['hour'] == 18) & (d['sig'] >= .98) & d['entry'].notna() & (d['entry'] <= .995)]
mm = best.groupby('month').agg(n=('r', 'size'), win=('yes', 'mean'), roi=('r', 'mean'), cap_med=('cap_usd', 'median'))
print('\n=== 18h, sig >= 0.98, by month ===\n' + mm.round(4).to_string())
print('\n=== every loss at hours 17-20, sig >= 0.97 ===\n' + d[(d['hour'].between(17, 20)) & (d['sig'] >= .97) & (d['yes'] == 0) & d['entry'].notna()]
      [['city', 'target_date', 'hour', 'bucket', 'sig', 'entry', 'cap_usd']].sort_values(['target_date', 'city', 'hour']).to_string(index=False))
print('\n=== 18h sig>=0.98 by city (n>=10) ===\n' + best.groupby('city').agg(n=('r', 'size'), win=('yes', 'mean'), roi=('r', 'mean')).query('n>=10').round(4).sort_values('roi').to_string())
json.dump(dict(grid=rows, by_month=mm.round(4).reset_index().astype({'month': str}).to_dict('records')), open('results/a10_evening_lock.json', 'w'), indent=1, default=str)
