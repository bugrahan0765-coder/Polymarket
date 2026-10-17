"""Step 11b: does the weather market lag live METAR observations?
1) Map each city to its resolution station: the candidate whose METAR daily max (local day, rounded in the market unit)
   most often falls in the winning bucket.
2) At local hours T on the target day, observed max-so-far (obs valid <= T - 5 min) kills every bucket entirely below it.
   (A) 'dead bucket' trade: buy NO on buckets whose upper edge < max-so-far, at the real NO ask after T+30s.
   (B) 'lock' model: P(final bucket == bucket of max-so-far) by logistic regression on hour, drop from max, distance to the
       bucket's upper edge, unit; fitted on events before SPLIT, applied after. Trade YES/NO vs the real ask when |edge| > t.
Run: python src/a11_obs_lag.py <ext_dir>
"""
import sys, re, glob, json, gzip, datetime as dt
from zoneinfo import ZoneInfo
import duckdb, numpy as np, pandas as pd
from analysis_lib import load_markets
from common import TZ
from a11_metar_fetch import CANDIDATES

EXT = sys.argv[1] if len(sys.argv) > 1 else '../hf_data/ext'
SPLIT = pd.Timestamp('2026-06-01', tz='UTC').timestamp()
FEE_START = pd.Timestamp('2026-03-30', tz='UTC').timestamp()
HOURS = [10, 11, 12, 13, 14, 15, 16, 17, 18]


def parse_bucket(s):
    unit = 'F' if 'F' in s else 'C'
    nums = [int(x) for x in re.findall(r'-?\d+', s)]
    if 'below' in s: return unit, -999, nums[0]
    if 'higher' in s or 'above' in s: return unit, nums[0], 999
    return unit, nums[0], nums[-1]


m = load_markets(); m = m[m['yes'].notna()].copy(); m['market_id'] = m['market_id'].astype(str)
m[['unit', 'lo', 'hi']] = pd.DataFrame([parse_bucket(b) for b in m['bucket']], index=m.index)
win = m[m['yes'] == 1][['event_id', 'city', 'target_date', 'unit', 'lo', 'hi']]

# --- METAR: local-day max per station, in both units
obs = {}
for p in glob.glob('data/metar/*.csv.gz'):
    icao = p.split('\\')[-1].split('/')[-1][:4]
    o = pd.read_csv(p, comment='#')
    o = o[pd.to_numeric(o['tmpf'], errors='coerce').notna()]
    o['tmpf'] = o['tmpf'].astype(float); o['ts'] = pd.to_datetime(o['valid'], utc=True)
    obs[icao] = o[['ts', 'tmpf']]


def to_unit(f, unit):
    return np.round(f) if unit == 'F' else np.round((f - 32) / 1.8)


station = {}; rows = []
for city, cands in CANDIDATES.items():
    w = win[win['city'] == city]
    if w.empty: continue
    best = None
    for icao in cands:
        if icao not in obs: continue
        o = obs[icao].copy(); o['date'] = o['ts'].dt.tz_convert(TZ[city]).dt.date
        dmax = o.groupby('date')['tmpf'].max()
        x = w.assign(f=w['target_date'].map(dmax)).dropna(subset=['f'])
        x['v'] = [to_unit(f, u) for f, u in zip(x['f'], x['unit'])]
        rate = ((x['v'] >= x['lo']) & (x['v'] <= x['hi'])).mean() if len(x) else 0
        rows.append(dict(city=city, icao=icao, days=len(x), match=round(rate, 3)))
        if best is None or rate > best[1]: best = (icao, rate)
    if best: station[city] = best[0]
st = pd.DataFrame(rows); print('=== station match rates ===\n' + st.to_string(index=False))
json.dump(dict(station=station, rates=rows), open('results/a11_stations.json', 'w'), indent=1)
good_city = set(st.sort_values('match').groupby('city').tail(1).query('match >= 0.85')['city'])
print(f'cities with a station matching >= 85%: {len(good_city)}')

# --- snapshots: every bucket of every event at local hours on target day, with max-so-far and current temp
ev = m[m['city'].isin(good_city)].drop_duplicates('event_id')[['event_id', 'city', 'target_date']]
snap = []
for e in ev.itertuples():
    tz = ZoneInfo(TZ[e.city]); o = obs[station[e.city]]
    d0 = dt.datetime(e.target_date.year, e.target_date.month, e.target_date.day, tzinfo=tz)
    day = o[(o['ts'] >= d0) & (o['ts'] < d0 + dt.timedelta(days=1))]
    for hh in HOURS:
        T = d0 + dt.timedelta(hours=hh)
        seen = day[day['ts'] <= T - dt.timedelta(minutes=5)]
        if len(seen) < 3: continue
        snap.append((e.event_id, hh, int(T.timestamp()), seen['tmpf'].max(), seen['tmpf'].iloc[-1]))
S = pd.DataFrame(snap, columns=['event_id', 'hour', 'T', 'maxf', 'curf'])
S = S.merge(m[['event_id', 'market_id', 'yes', 'unit', 'lo', 'hi', 'city']], on='event_id')
S['maxu'] = [to_unit(f, u) for f, u in zip(S['maxf'], S['unit'])]
S['curu'] = [to_unit(f, u) for f, u in zip(S['curf'], S['unit'])]
S['dead'] = S['hi'] < S['maxu']
S['cur_bucket'] = (S['maxu'] >= S['lo']) & (S['maxu'] <= S['hi'])

c = duckdb.connect()
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1 from read_parquet('{EXT}/weather_trades/*.parquet', union_by_name=true) order by market_id, ts")
c.register('s_df', S[['event_id', 'hour', 'T', 'market_id']])
px = c.execute("""select s.*, sig.p1 sig, sig.ts sig_ts, b.p1 ask1, b.ts ask_ts, x.p1 bid1, x.ts bid_ts from s_df s
  asof left join f sig on s.market_id = sig.market_id and s.T > sig.ts
  asof left join (select * from f where buy1) b on s.market_id = b.market_id and s.T + 30 <= b.ts
  asof left join (select * from f where not buy1) x on s.market_id = x.market_id and s.T + 30 <= x.ts""").df()
S = S.merge(px, on=['event_id', 'hour', 'T', 'market_id'])
S['period'] = np.where(S['T'] < SPLIT, 'H1', 'H2')
S['yes_ok'] = S['ask1'].notna() & (S['ask_ts'] - S['T'] <= 900)
S['no_ok'] = S['bid1'].notna() & (S['bid_ts'] - S['T'] <= 900)
fee = lambda p, T: np.where(T >= FEE_START, .05 * (1 - p), 0.)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(f'{len(S):,} bucket snapshots, {S.event_id.nunique():,} events')

# (A) dead buckets: sanity (should never win) and NO trade
dd = S[S['dead'] & S['sig'].notna()]
print(f"\n=== (A) dead buckets: n={len(dd):,}, won anyway={dd['yes'].sum()} ({dd['yes'].mean():.4%}) ===")
x = dd[dd['no_ok']].assign(px=lambda t: 1 - t['bid1'])
x = x[x['px'] < .999]
x['r'] = (1 - x['yes']) / x['px'] - 1 - fee(x['px'], x['T'])
x['pbin'] = pd.cut(x['sig'], [0, .005, .01, .02, .05, .1, 1.01], right=False)
print(x.groupby(['hour', 'pbin'], observed=True).agg(n=('r', 'size'), no_px=('px', 'mean'), won_yes=('yes', 'sum'), roi=('r', 'mean')).round(4).to_string())
print('by period:\n' + x.groupby('period').agg(n=('r', 'size'), roi=('r', 'mean'), losses=('yes', 'sum')).round(4).to_string())

# (B) lock model on the bucket containing max-so-far
cb = S[S['cur_bucket'] & S['sig'].notna()].copy()
cb['drop'] = cb['maxu'] - cb['curu']
cb['room'] = np.where(cb['hi'] >= 999, 9, cb['hi'] - cb['maxu'])  # degrees left before leaving the bucket upward
cb['isF'] = (cb['unit'] == 'F').astype(int)
X = lambda t: np.column_stack([np.ones(len(t)), t['hour'], t['hour'] ** 2, t['drop'].clip(0, 8), t['room'].clip(0, 9), t['isF'], t['room'].clip(0, 9) * t['hour']])
tr = cb[cb['period'] == 'H1']
beta = np.zeros(X(tr).shape[1])
for _ in range(50):  # Newton-Raphson logistic regression
    z = X(tr) @ beta; p = 1 / (1 + np.exp(-z)); W = p * (1 - p)
    beta += np.linalg.solve(X(tr).T @ (X(tr) * W[:, None]) + 1e-6 * np.eye(len(beta)), X(tr).T @ (tr['yes'] - p))
cb['P'] = 1 / (1 + np.exp(-(X(cb) @ beta)))
cb['pb'] = pd.cut(cb['P'], [0, .3, .5, .7, .8, .9, .95, .98, 1.01])
print('\n=== (B) lock model calibration and market price, by period ===')
print(cb.groupby(['period', 'pb'], observed=True).agg(n=('yes', 'size'), P=('P', 'mean'), won=('yes', 'mean'), mkt=('sig', 'mean')).round(4).to_string())
res = []
for t in [.02, .05, .08, .12]:
    for side in ['YES', 'NO']:
        if side == 'YES':
            x = cb[cb['yes_ok']].assign(px=lambda d: d['ask1'], w=lambda d: d['yes'], edge=lambda d: d['P'] - d['ask1'])
        else:
            x = cb[cb['no_ok']].assign(px=lambda d: 1 - d['bid1'], w=lambda d: 1 - d['yes'], edge=lambda d: (1 - d['P']) - (1 - d['bid1']))
        x = x[(x['edge'] > t) & x['px'].between(.02, .995)]
        x['r'] = x['w'] / x['px'] - 1 - fee(x['px'], x['T'])
        for per, g in x.groupby('period'):
            res.append(dict(thr=t, side=side, period=per, n=len(g), per_day=round(len(g) / max(g['T'].floordiv(86400).nunique(), 1), 2),
                            win=round(g['w'].mean(), 4), px=round(g['px'].mean(), 4), roi=round(g['r'].mean(), 4), se=round(g['r'].sem(), 4)))
print('\n=== (B) trades where model beats the real ask (model fitted on H1 only) ===\n' + pd.DataFrame(res).to_string(index=False))
cb.drop(columns='pb').to_parquet('../hf_data/a11_cb.parquet', compression='zstd')
json.dump(dict(beta=beta.tolist(), trades=res), open('results/a11_obs_lag.json', 'w'), indent=1)
