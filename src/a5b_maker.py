"""Step 5b: rigorous test of the weather-favourite MAKER strategy and a look-ahead-free TAKER check.
Maker: at the v1 decision time post a YES bid at b = p - offset on the favourite (lo <= p < hi), keep it W hours.
Filled only if takers later traded THROUGH it (token1 price < b), and at least MIN_USD traded through (capacity).
Taker (no look-ahead): entry = p + measured half-spread, on ALL favourites (not only those with a later trade).
Reports by month, walk-forward choice of (offset, W, band) on prior months, date-clustered bootstrap.
Run: python src/a5b_maker.py <ext_dir>
"""
import sys, glob, json, itertools, numpy as np, pandas as pd
from analysis_lib import load_markets

EXT = sys.argv[1] if len(sys.argv) > 1 else '../hf_data/ext'
CAP, MIN_USD = 5., 10.
FEE_START = pd.Timestamp('2026-03-30', tz='UTC').timestamp()

m = load_markets()
m = m[m['yes'].notna() & m['p'].notna()].copy()
m['market_id'] = m['market_id'].astype(str)
m['fav'] = m.groupby('event_id')['p'].rank(ascending=False, method='first') == 1
f = m[m['fav'] & (m['p'] >= .2) & (m['p'] < .95)].copy()
f['month'] = pd.to_datetime(f['decision_ts'], unit='s').dt.to_period('M')
dec = f.set_index('market_id')['decision_ts']; pmap = f.set_index('market_id')['p']

w = pd.concat([pd.read_parquet(x, columns=['timestamp', 'market_id', 'p1', 'buy1', 'usd_amount']) for x in glob.glob(f'{EXT}/weather_trades/*.parquet')])
w = w[w['market_id'].isin(dec.index)]
w['rel'] = w['timestamp'] - w['market_id'].map(dec)
sells = w[(~w['buy1']) & (w['rel'] > 0)]
print(f'{len(f):,} favourites, {sells.market_id.nunique():,} with later taker sells')

OFFSETS, WINDOWS = [0., .01, .02, .03, .04], [1, 6, 24, 48]
for d, W in itertools.product(OFFSETS, WINDOWS):
    x = sells[sells['rel'] <= W * 3600]
    b = x['market_id'].map(pmap) - d
    thr = x[x['p1'] < b - 1e-9].groupby('market_id')['usd_amount'].sum()
    f[f'fill_{d}_{W}'] = f['market_id'].map(thr).fillna(0) >= MIN_USD


def pnl_maker(df, d):
    b = df['p'] - d
    return np.where(df['yes'] == 1, CAP / b, 0) - CAP


BANDS = [(.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .75), (.75, .95), (.3, .9), (.2, .95)]
out = {}

# 1) full grid, in-sample, by period
rows = []
for (lo, hi), d, W in itertools.product(BANDS, OFFSETS, WINDOWS):
    g = f[(f['p'] >= lo) & (f['p'] < hi)]
    fl = g[g[f'fill_{d}_{W}']]
    if len(fl) < 30: continue
    r = pnl_maker(fl, d)
    early = fl['decision_ts'] < FEE_START
    rows.append(dict(band=f'{lo}-{hi}', offset=d, hours=W, n=len(g), fills=len(fl), fill_rate=round(len(fl) / len(g), 3),
                     roi=round(r.mean() / CAP, 4), roi_early=round(r[early].mean() / CAP, 4) if early.sum() > 10 else None,
                     roi_late=round(r[~early].mean() / CAP, 4), pnl=round(r.sum(), 1),
                     win_filled=round(fl['yes'].mean(), 4), win_unfilled=round(g[~g[f'fill_{d}_{W}']]['yes'].mean(), 4)))
grid = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print('\n=== MAKER grid (through-fills with >= $10 traded through) ===\n' + grid.sort_values('pnl', ascending=False).head(40).to_string(index=False))
out['grid'] = rows

# 2) walk-forward: each month pick (band, offset, W) with best total PnL on all prior months (>= 3 months history)
months = sorted(f['month'].unique()); wf = []; trades = []
for i, mo in enumerate(months):
    if i < 3: continue
    tr = f[f['month'] < mo]; te = f[f['month'] == mo]
    best = None
    for (lo, hi), d, W in itertools.product(BANDS[:6] + [(.3, .9)], OFFSETS, WINDOWS):
        g = tr[(tr['p'] >= lo) & (tr['p'] < hi) & tr[f'fill_{d}_{W}']]
        if len(g) < 50: continue
        s = pnl_maker(g, d).sum()
        if best is None or s > best[0]: best = (s, lo, hi, d, W)
    _, lo, hi, d, W = best
    g = te[(te['p'] >= lo) & (te['p'] < hi) & te[f'fill_{d}_{W}']]
    r = pnl_maker(g, d)
    trades.append(g.assign(pnl=r))
    wf.append(dict(month=str(mo), band=f'{lo}-{hi}', offset=d, hours=W, fills=len(g), pnl=round(r.sum(), 1), roi=round(r.mean() / CAP, 4) if len(g) else None))
wf = pd.DataFrame(wf); T = pd.concat(trades)
print('\n=== MAKER walk-forward (choice on prior months only) ===\n' + wf.to_string(index=False))
rng = np.random.default_rng(0); days = T.groupby('target_date')['pnl'].agg(['sum', 'size'])
bs = [days['sum'].to_numpy()[i].sum() / (CAP * days['size'].to_numpy()[i].sum()) for i in (rng.integers(0, len(days), len(days)) for _ in range(5000))]
summ = dict(fills=len(T), pnl=round(T['pnl'].sum(), 1), roi=round(T['pnl'].mean() / CAP, 4), ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            losing_months=int((wf['pnl'] < 0).sum()), months=len(wf), fills_per_day=round(len(T) / T['target_date'].nunique(), 1))
cum = T.sort_values('decision_ts')['pnl'].cumsum(); summ['max_drawdown'] = round(float((cum.cummax() - cum).max()), 1)
print(summ); out['walkforward'] = dict(months=wf.to_dict('records'), summary=summ)

# 3) fixed simple rule out-of-sample by month: band 0.3-0.9, offset 0.02, 24h
for lo, hi, d, W in [(.3, .9, .02, 24), (.3, .9, .01, 6), (.4, .5, .02, 24)]:
    g = f[(f['p'] >= lo) & (f['p'] < hi) & f[f'fill_{d}_{W}']].assign(pnl=lambda x: pnl_maker(x, d))
    mm = g.groupby('month').agg(fills=('pnl', 'size'), pnl=('pnl', 'sum'), win=('yes', 'mean'), b=('p', lambda p: (p - d).mean()))
    mm['roi'] = mm['pnl'] / (CAP * mm['fills'])
    print(f'\n=== fixed rule band {lo}-{hi} offset {d} {W}h by month ===\n' + mm.round(3).to_string())
    out[f'fixed_{lo}_{hi}_{d}_{W}'] = mm.round(4).reset_index().astype({'month': str}).to_dict('records')

# 4) taker without look-ahead: entry = p + half-spread h on ALL favourites in band, fee after FEE_START
rows = []
for lo, hi in [(.3, .9), (.4, .5), (.6, .9)]:
    g = f[(f['p'] >= lo) & (f['p'] < hi)]
    for h in [.005, .01, .015]:
        e = g['p'] + h; fee = np.where(g['decision_ts'] >= FEE_START, .05 * e * (1 - e), 0)
        sh = CAP / e; r = np.where(g['yes'] == 1, sh, 0) - CAP - sh * fee
        late = g['decision_ts'] >= FEE_START
        rows.append(dict(band=f'{lo}-{hi}', half_spread=h, n=len(g), roi=round(r.sum() / (CAP * len(g)), 4), roi_late=round(r[late].sum() / (CAP * late.sum()), 4)))
print('\n=== TAKER, no look-ahead, all favourites ===\n' + pd.DataFrame(rows).to_string(index=False))
out['taker_nolook'] = rows
json.dump(out, open('results/a5b_maker.json', 'w'), indent=1, default=str)
