"""Step 13c: crypto up/down latency test with 1-second Binance spot (bulk 1s klines, filtered to sampled market windows).
Decision every 10 s inside each window: P(up) from z = ln(S[T-1s]/S[start]) / (sigma_1m * sqrt(min left)), logistic fit on H1.
Entry = first real taker fill after T + 1 s (<= 30 s). Cells: model prob x entry price, by period; market-level bootstrap;
capacity = taker USD on the same side at <= entry + 0.3c within 60 s.
Run: python src/a13c_crypto_1s.py <hf_data_dir>   (downloads ~5 GB once, keeps only the needed seconds)
"""
import sys, os, io, json, zipfile, urllib.request, duckdb, numpy as np, pandas as pd
from analysis_lib import load_meta

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SPLIT = pd.Timestamp('2026-06-01', tz='UTC').timestamp()
COINS = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT', 'doge': 'DOGEUSDT', 'bnb': 'BNBUSDT'}
STEP, D, WAIT, RATE = 10, 1, 30, .07

meta = load_meta(HF)
meta = meta[pd.to_numeric(meta['market_id'], errors='coerce') % 20 == 0]
x = meta['event_slug'].astype(str).str.extract(r'^(btc|eth|sol|xrp|doge|bnb)-updown-(\d+)m-(\d+)$')
mk = meta.assign(coin=x[0], w=pd.to_numeric(x[1]), s=pd.to_numeric(x[2])).dropna(subset=['coin', 'win1'])
mk = mk[(mk['end_ts'] - mk['s'] == mk['w'] * 60) & mk['w'].isin([5, 15])][['market_id', 'coin', 'w', 's', 'end_ts', 'win1']]

os.makedirs(f'{HF}/binance1s', exist_ok=True)
for coin, sym in COINS.items():
    p = f'{HF}/binance1s/{sym}.parquet'
    if os.path.exists(p): continue
    w = mk[mk.coin == coin]
    need = np.unique(np.concatenate([np.arange(a - 1, b + 1) for a, b in zip(w['s'].astype(int), w['end_ts'].astype(int))]))
    parts = []
    months = sorted({pd.Timestamp(t, unit='s').strftime('%Y-%m') for t in w['s']})
    for mth in months:
        urls = [f'https://data.binance.vision/data/spot/monthly/klines/{sym}/1s/{sym}-1s-{mth}.zip']
        if mth == '2026-10':
            urls = [f'https://data.binance.vision/data/spot/daily/klines/{sym}/1s/{sym}-1s-2026-10-{d:02d}.zip' for d in range(1, 6)]
        for u in urls:
            try:
                z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u, timeout=600).read()))
            except Exception as e:
                print('  miss', u, e); continue
            d = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 4], names=['t', 'open', 'close'])
            d['t'] = d['t'] // 1000 if d['t'].iloc[0] < 1e15 else d['t'] // 1_000_000
            parts.append(d[np.isin(d['t'].to_numpy(), need)])
        print(sym, mth, flush=True)
    pd.concat(parts).drop_duplicates('t').sort_values('t').to_parquet(p, compression='zstd', index=False)

# features
rows = []
for coin, sym in COINS.items():
    s1 = pd.read_parquet(f'{HF}/binance1s/{sym}.parquet').set_index('t')
    k = pd.read_parquet(f'{HF}/binance/{sym}-1m.parquet').set_index('t')
    sig = np.log(k['close']).diff().rolling(60, min_periods=40).std()
    for r in mk[mk.coin == coin].itertuples():
        s0, e = int(r.s), int(r.end_ts)
        if s0 not in s1.index: continue
        o = s1.at[s0, 'open']; sg = sig.get(s0 - 60, np.nan)
        T = np.arange(s0 + STEP, e - 1, STEP)
        c = s1['close'].reindex(T - 1).ffill().to_numpy()
        for t, cc in zip(T, c):
            rows.append((r.market_id, coin, r.w, t, r.win1, np.log(cc / o), sg, (e - t) / 60))
S = pd.DataFrame(rows, columns=['market_id', 'coin', 'w', 'T', 'win1', 'x', 'sig', 'left'])
S['z'] = S['x'] / (S['sig'] * np.sqrt(S['left'])); S = S[np.isfinite(S['z'])]
S['period'] = np.where(S['T'] < SPLIT, 'H1', 'H2')
print(f'{len(S):,} snapshots on {S.market_id.nunique():,} markets')

c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3')
c.register('mk_df', mk[['market_id']])
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{HF}/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from mk_df) order by market_id, ts")
c.register('s_df', S[['market_id', 'T']])
q = c.execute(f"""select s.*, b.p1 ask1, b.ts ask_ts, x.p1 bid1, x.ts bid_ts from s_df s
  asof left join (select * from f where buy1) b on s.market_id = b.market_id and s.T + {D} <= b.ts
  asof left join (select * from f where not buy1) x on s.market_id = x.market_id and s.T + {D} <= x.ts""").df()
Q = S.merge(q, on=['market_id', 'T'])


def logit_fit(X, y, it=40):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


F = lambda t: np.column_stack([np.ones(len(t)), t['z'].clip(-8, 8), t['z'].clip(-8, 8) ** 2 * np.sign(t['z'])])
tr = Q[Q.period == 'H1'].drop_duplicates('market_id', keep='last').sample(frac=1, random_state=0)  # thin to reduce dependence
tr = Q[Q.period == 'H1'].sample(n=min(300000, (Q.period == 'H1').sum()), random_state=0)
bz = logit_fit(F(tr), tr['win1'].to_numpy()); Q['P'] = 1 / (1 + np.exp(-(F(Q) @ bz)))
up = Q[Q.ask1.notna() & (Q.ask_ts - Q['T'] <= WAIT)].assign(tok=1, px=lambda d: d.ask1, win=lambda d: d.win1, P2=lambda d: d.P, ts=lambda d: d.ask_ts)
dn = Q[Q.bid1.notna() & (Q.bid_ts - Q['T'] <= WAIT)].assign(tok=2, px=lambda d: 1 - d.bid1, win=lambda d: 1 - d.win1, P2=lambda d: 1 - d.P, ts=lambda d: d.bid_ts)
t = pd.concat([up, dn]); t = t[t.px.between(.02, .995)]
t['r'] = t.win / t.px - 1 - RATE * (1 - t.px)
t['edge'] = t.P2 - t.px
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
out = {}
print('\n=== by model prob x entry price (one row per snapshot) ===')
g = t.groupby(['period', pd.cut(t.px, [0, .5, .8, .9, .95, .97, .98, .99, .995]), pd.cut(t.P2, [0, .9, .97, .99, .997, 1])], observed=True).agg(
    n=('r', 'size'), px=('px', 'mean'), win=('win', 'mean'), roi=('r', 'mean'), se=('r', 'sem'))
print(g[g.n >= 100].round(4).to_string())
print('\n=== edge threshold rules, first qualifying snapshot per market ===')
res = []
for thr in [.005, .01, .02, .03, .05]:
    for lo in [.5, .9, .95]:
        x = t[(t.edge > thr) & (t.px >= lo)].sort_values('T').drop_duplicates('market_id')
        for per, h in x.groupby('period'):
            a = h['r'].to_numpy(); rng = np.random.default_rng(0)
            bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(1000)]
            res.append(dict(thr=thr, px_min=lo, period=per, n=len(h), px=round(h.px.mean(), 4), win=round(h.win.mean(), 4),
                            roi=round(a.mean(), 4), lo95=round(np.percentile(bs, 2.5), 4), hi95=round(np.percentile(bs, 97.5), 4)))
R = pd.DataFrame(res); print(R.to_string(index=False))
t.to_parquet(f'{HF}/a13c_t.parquet', compression='zstd')
json.dump(dict(beta=bz.tolist(), rules=res), open('results/a13c_crypto_1s.json', 'w'), indent=1, default=float)
