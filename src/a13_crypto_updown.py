"""Step 13b: do Polymarket crypto up/down (5m/15m) markets lag Binance spot? 5% sample, point-in-time.
At each whole minute T inside the window: spot move so far x = ln(close[T-1m] / open[start]), sigma = std of 1m log returns
over the previous hour, z = x / (sigma * sqrt(minutes left)), model P(up) = Phi(z) (recalibrated on H1 by logistic fit).
Market: signal = last fill before T; entry = first real taker fill after T + DELAY (<= 60 s) on Up (token1) or Down.
Trade when model - ask > thr. Taker fee 0.07 * p * (1 - p) per share. Also: does z add information beyond the market price?
Run: python src/a13_crypto_updown.py <hf_data_dir>
"""
import sys, json, math, duckdb, numpy as np, pandas as pd
from analysis_lib import load_meta

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SPLIT = pd.Timestamp('2026-06-01', tz='UTC').timestamp()
COINS = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT', 'doge': 'DOGEUSDT', 'bnb': 'BNBUSDT'}
RATE = .07

meta = load_meta(HF)
meta = meta[pd.to_numeric(meta['market_id'], errors='coerce') % 20 == 0]
x = meta['event_slug'].astype(str).str.extract(r'^(btc|eth|sol|xrp|doge|bnb)-updown-(\d+)m-(\d+)$')
mk = meta.assign(coin=x[0], w=pd.to_numeric(x[1]), s=pd.to_numeric(x[2])).dropna(subset=['coin', 'win1'])
mk = mk[(mk['end_ts'] - mk['s'] == mk['w'] * 60) & mk['w'].isin([5, 15])][['market_id', 'coin', 'w', 's', 'end_ts', 'win1']]
print(f'{len(mk):,} sampled up/down markets:', mk.groupby(['coin', 'w']).size().to_dict())

# spot features
K = {}
for coin, sym in COINS.items():
    k = pd.read_parquet(f'{HF}/binance/{sym}-1m.parquet').set_index('t')
    k = k.reindex(np.arange(k.index.min(), k.index.max() + 60, 60)).ffill()
    lr = np.log(k['close']).diff()
    k['sig60'] = lr.rolling(60, min_periods=40).std()
    K[coin] = k
rows = []
for r in mk.itertuples():
    k = K[r.coin]
    if r.s not in k.index or r.end_ts - 60 not in k.index: continue
    o = k.at[r.s, 'open']
    up_bn = k.at[r.end_ts - 60, 'close'] >= o
    for j in range(1, int(r.w)):
        T = r.s + 60 * j; c = k.at[T - 60, 'close']; sg = k.at[T - 60, 'sig60']
        rows.append((r.market_id, r.coin, r.w, j, T, r.win1, up_bn, math.log(c / o), sg, (r.end_ts - T) / 60))
S = pd.DataFrame(rows, columns=['market_id', 'coin', 'w', 'j', 'T', 'win1', 'up_bn', 'x', 'sig', 'left'])
S['z'] = S['x'] / (S['sig'] * np.sqrt(S['left']))
S = S[np.isfinite(S['z'])]
v = S.drop_duplicates('market_id')
print(f"Binance close>=open agrees with resolution: {(v['up_bn'] == (v['win1'] == 1)).mean():.4f} on {len(v):,} markets")

c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3')
c.register('mk_df', mk[['market_id']])
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{HF}/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from mk_df) order by market_id, ts")
c.register('s_df', S[['market_id', 'j', 'T']])
out = []
for D in [2, 10]:
    q = c.execute(f"""select s.*, sig.p1 mkt, sig.ts mkt_ts, b.p1 ask1, b.ts ask_ts, b.usd ask_usd, x.p1 bid1, x.ts bid_ts, x.usd bid_usd from s_df s
      asof left join f sig on s.market_id = sig.market_id and s.T > sig.ts
      asof left join (select * from f where buy1) b on s.market_id = b.market_id and s.T + {D} <= b.ts
      asof left join (select * from f where not buy1) x on s.market_id = x.market_id and s.T + {D} <= x.ts""").df()
    q['delay'] = D; out.append(q)
Q = pd.concat(out).merge(S, on=['market_id', 'j', 'T'])
Q = Q[Q['mkt'].notna() & (Q['T'] - Q['mkt_ts'] <= 120)]
Q['period'] = np.where(Q['T'] < SPLIT, 'H1', 'H2')
print(f'{len(Q[Q.delay == 2]):,} snapshots with a market price <= 2 min old')


def logit_fit(X, y, it=40):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))
q0 = Q[Q.delay == 2].copy()
res = {}
for per in ['H1', 'H2']:
    t = q0[q0.period == per]
    X = np.column_stack([np.ones(len(t)), lg(t['mkt']), t['z'].clip(-6, 6)])
    b = logit_fit(X, t['win1'].to_numpy())
    res[f'info_{per}'] = dict(n=len(t), b_const=b[0], b_logit_mkt=b[1], b_z=b[2])
    print(f'{per}: win1 ~ logit(mkt) + z  ->  coef mkt={b[1]:.3f}  z={b[2]:.3f}  (z>0 means market under-reacts to spot)')

# model probability: logistic on z only, fitted on H1
t = q0[q0.period == 'H1']
bz = logit_fit(np.column_stack([np.ones(len(t)), t['z'].clip(-6, 6), t['z'].clip(-6, 6) ** 2 * np.sign(t['z'])]), t['win1'].to_numpy())
Q['P'] = 1 / (1 + np.exp(-(np.column_stack([np.ones(len(Q)), Q['z'].clip(-6, 6), Q['z'].clip(-6, 6) ** 2 * np.sign(Q['z'])]) @ bz)))
print('calibration of model P (delay 2 rows):')
q0 = Q[Q.delay == 2]
print(q0.groupby(['period', pd.cut(q0['P'], [0, .05, .2, .4, .6, .8, .95, 1])], observed=True).agg(n=('win1', 'size'), P=('P', 'mean'), won=('win1', 'mean'), mkt=('mkt', 'mean')).round(4).to_string())

trades = []
for D in [2, 10]:
    for thr in [.03, .06, .1, .15]:
        g = Q[Q.delay == D]
        up = g[g['ask1'].notna() & (g['ask_ts'] - g['T'] <= 60)].assign(px=lambda d: d['ask1'], win=lambda d: d['win1'], e=lambda d: d['P'] - d['ask1'], usd=lambda d: d['ask_usd'])
        dn = g[g['bid1'].notna() & (g['bid_ts'] - g['T'] <= 60)].assign(px=lambda d: 1 - d['bid1'], win=lambda d: 1 - d['win1'], e=lambda d: d['bid1'] - d['P'], usd=lambda d: d['bid_usd'])
        tr = pd.concat([up, dn]); tr = tr[(tr['e'] > thr) & tr['px'].between(.02, .98)]
        tr = tr.sort_values('e', ascending=False).drop_duplicates(['market_id'])  # one trade per market
        tr['r'] = tr['win'] / tr['px'] - 1 - RATE * (1 - tr['px'])
        for (per, w), h in tr.groupby(['period', 'w']):
            days = max(h['T'].floordiv(86400).nunique(), 1)
            trades.append(dict(delay=D, thr=thr, period=per, w=int(w), n=len(h), per_day_sample=round(len(h) / days, 1), px=round(h['px'].mean(), 3),
                               win=round(h['win'].mean(), 3), roi=round(h['r'].mean(), 4), se=round(h['r'].sem(), 4), fill_usd_med=round(h['usd'].median(), 1)))
T = pd.DataFrame(trades)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('\n=== model vs real ask trades (one per market) ===\n' + T.to_string(index=False))
json.dump(dict(info=res, trades=trades, beta_z=bz.tolist()), open('results/a13_crypto_updown.json', 'w'), indent=1, default=float)
Q.to_parquet(f'{HF}/a13_q.parquet', compression='zstd')
