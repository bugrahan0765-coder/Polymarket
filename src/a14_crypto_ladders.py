"""Step 14: crypto price-ladder markets (hourly/daily 'above K', multistrike, 'between A and B' ranges) vs a Binance spot model.
Resolution = Binance 1m candle close at the stated ET time (validated below). 5% fill sample, point-in-time:
T = end - horizon; spot = close of minute T-1; sigma = std of 1m log returns over the previous 24 h (and 1 h).
Raw model P = Normal probability of the condition at expiry; recalibrated by logistic on logit(P_raw) fitted on H1.
Entry = first real taker fill after T + DELAY (<= 15 min) on Yes (token1) or No. Fee 0.07 p(1-p) per share.
Run: python src/a14_crypto_ladders.py <hf_data_dir>
"""
import sys, re, json, duckdb, numpy as np, pandas as pd
from scipy.stats import norm
from analysis_lib import load_meta

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
SPLIT = pd.Timestamp('2026-06-01', tz='UTC').timestamp()
COINS = {'Bitcoin': 'BTCUSDT', 'Ethereum': 'ETHUSDT', 'Solana': 'SOLUSDT', 'XRP': 'XRPUSDT'}
HOR = {'5m': 5, '15m': 15, '1h': 60, '3h': 180, '8h': 480, '24h': 1440}
RATE, WAIT = .07, 900
num = lambda s: float(s.replace(',', '').replace('$', ''))

meta = load_meta(HF)
meta = meta[(meta.cat == 'crypto') & meta.win1.notna()]
q = meta['question'].astype(str)
coin = q.str.extract(r'\b(Bitcoin|Ethereum|Solana|XRP)\b')[0]
ab = q.str.extract(r'above \$?([\d,\.]+)')[0]
lt = q.str.extract(r'(?:less than|below) \$?([\d,\.]+)')[0]
bt = q.str.extract(r'between \$?([\d,\.]+) and \$?([\d,\.]+)')
gt = q.str.extract(r'(?:greater than|more than|over) \$?([\d,\.]+)')[0]
mk = meta.assign(coin=coin, kind=np.select([ab.notna(), gt.notna(), lt.notna(), bt[0].notna()], ['above', 'above', 'below', 'between'], ''),
                 lo=np.where(ab.notna(), ab, np.where(gt.notna(), gt, np.where(bt[0].notna(), bt[0], np.nan))),
                 hi=np.where(lt.notna(), lt, np.where(bt[1].notna(), bt[1], np.nan)))
mk = mk[mk.coin.notna() & (mk.kind != '') & ~q.str.contains('reach|hit|dip', case=False)].copy()
for col in ['lo', 'hi']:
    mk[col] = [num(v) if isinstance(v, str) else np.nan for v in mk[col]]
mk['fam'] = mk['event_slug'].astype(str).str.replace(r'\d+', '#', regex=True).str.replace(
    r'(january|february|march|april|may|june|july|august|september|october|november|december)', 'MON', regex=True).str.replace(
    r'^(bitcoin|ethereum|solana|xrp|btc|eth|sol)-', '', regex=True)
mk = mk[['market_id', 'coin', 'kind', 'lo', 'hi', 'end_ts', 'win1', 'fam', 'question']]
print(f'{len(mk):,} ladder markets; kinds', mk.kind.value_counts().to_dict())

K = {}
for c_, sym in COINS.items():
    k = pd.read_parquet(f'{HF}/binance/{sym}-1m.parquet').set_index('t')
    k = k.reindex(np.arange(k.index.min(), k.index.max() + 60, 60)).ffill()
    lr = np.log(k['close']).diff()
    k['s24'] = lr.rolling(1440, min_periods=1000).std(); k['s1'] = lr.rolling(60, min_periods=40).std()
    K[c_] = k


def cond(S, r):
    if r.kind == 'above': return S > r.lo
    if r.kind == 'below': return S < r.hi
    return (S >= r.lo) & (S < r.hi)


# resolution check: which candle close matches win1
chk = []
for r in mk.sample(min(20000, len(mk)), random_state=0).itertuples():
    k = K[r.coin]; e = int(r.end_ts)
    if e + 60 not in k.index or e - 60 not in k.index: continue
    chk.append((r.fam, cond(k.at[e - 60, 'close'], r) == (r.win1 == 1), cond(k.at[e, 'close'], r) == (r.win1 == 1)))
chk = pd.DataFrame(chk, columns=['fam', 'close_at_end', 'candle_starting_end'])
print('resolution agreement:\n' + chk.groupby('fam')[['close_at_end', 'candle_starting_end']].mean().round(4).assign(n=chk.groupby('fam').size()).query('n>=50').to_string())

samp = mk[pd.to_numeric(mk.market_id, errors='coerce') % 20 == 0]
rows = []
for r in samp.itertuples():
    k = K[r.coin]; e = int(r.end_ts)
    for h, mins in HOR.items():
        T = e - 60 * mins
        if T - 60 not in k.index: continue
        S, s24, s1 = k.at[T - 60, 'close'], k.at[T - 60, 's24'], k.at[T - 60, 's1']
        sd = s24 * np.sqrt(mins + 1)
        f = lambda B: norm.cdf(np.log(S / B) / sd)  # P(S_end > B)
        p = f(r.lo) if r.kind == 'above' else (1 - f(r.hi) if r.kind == 'below' else f(r.lo) - f(r.hi))
        rows.append((r.market_id, r.coin, r.kind, r.fam, h, T, r.win1, p, s1 / s24))
S = pd.DataFrame(rows, columns=['market_id', 'coin', 'kind', 'fam', 'horizon', 'T', 'win1', 'p_raw', 'vr'])
S = S[np.isfinite(S.p_raw)]
c = duckdb.connect(); c.execute("SET memory_limit='4GB'"); c.execute('SET threads=3')
c.register('ids', S[['market_id']].drop_duplicates())
c.execute(f"create table f as select market_id, timestamp ts, p1, buy1, usd_amount usd from read_parquet('{HF}/ext/sample_trades/*.parquet', union_by_name=true) where market_id in (select market_id from ids) order by market_id, ts")
c.register('s_df', S[['market_id', 'horizon', 'T']])
out = []
for D in [10, 60]:
    x = c.execute(f"""select s.*, sig.p1 mkt, sig.ts mkt_ts, b.p1 ask1, b.ts ask_ts, b.usd ask_usd, x.p1 bid1, x.ts bid_ts, x.usd bid_usd from s_df s
      asof left join f sig on s.market_id = sig.market_id and s.T > sig.ts
      asof left join (select * from f where buy1) b on s.market_id = b.market_id and s.T + {D} <= b.ts
      asof left join (select * from f where not buy1) x on s.market_id = x.market_id and s.T + {D} <= x.ts""").df()
    x['delay'] = D; out.append(x)
Q = pd.concat(out).merge(S, on=['market_id', 'horizon', 'T'])
Q['period'] = np.where(Q['T'] < SPLIT, 'H1', 'H2')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def logit_fit(X, y, it=40):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b))); W = p * (1 - p)
        b += np.linalg.solve(X.T @ (X * W[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b


Fx = lambda t: np.column_stack([np.ones(len(t)), lg(t.p_raw)])
tr = Q[(Q.delay == 10) & (Q.period == 'H1')]
b = logit_fit(Fx(tr), tr.win1.to_numpy()); Q['P'] = 1 / (1 + np.exp(-(Fx(Q) @ b)))
print('calibration beta', b.round(3))
q0 = Q[(Q.delay == 10) & Q.mkt.notna() & (Q['T'] - Q.mkt_ts <= 6 * 3600)]
for per in ['H1', 'H2']:
    t = q0[q0.period == per]
    bb = logit_fit(np.column_stack([np.ones(len(t)), lg(t.mkt), lg(t.P)]), t.win1.to_numpy())
    print(f'{per}: win1 ~ logit(mkt) {bb[1]:.3f} + logit(model) {bb[2]:.3f}   n={len(t)}')
print('Brier: model', ((q0.P - q0.win1) ** 2).groupby(q0.period).mean().round(4).to_dict(), ' market', ((q0.mkt - q0.win1) ** 2).groupby(q0.period).mean().round(4).to_dict())

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
res = []
for D in [10, 60]:
    g = Q[Q.delay == D]
    up = g[g.ask1.notna() & (g.ask_ts - g['T'] <= WAIT)].assign(px=lambda d: d.ask1, win=lambda d: d.win1, e=lambda d: d.P - d.ask1, usd=lambda d: d.ask_usd)
    dn = g[g.bid1.notna() & (g.bid_ts - g['T'] <= WAIT)].assign(px=lambda d: 1 - d.bid1, win=lambda d: 1 - d.win1, e=lambda d: d.bid1 - d.P, usd=lambda d: d.bid_usd)
    t = pd.concat([up, dn]); t = t[t.px.between(.02, .98)]
    t['r'] = t.win / t.px - 1 - RATE * (1 - t.px)
    for thr in [.03, .05, .08, .12]:
        for lo, hi in [(.02, .5), (.5, .9), (.9, .98)]:
            x = t[(t.e > thr) & (t.px >= lo) & (t.px < hi)].sort_values('e', ascending=False).drop_duplicates(['market_id', 'horizon'])
            for (per, h), y in x.groupby(['period', 'horizon']):
                if len(y) < 20: continue
                res.append(dict(delay=D, thr=thr, band=f'{lo}-{hi}', horizon=h, period=per, n=len(y), px=round(y.px.mean(), 3), win=round(y.win.mean(), 3),
                                roi=round(y.r.mean(), 4), se=round(y.r.sem(), 4), usd_med=round(y.usd.median(), 1)))
R = pd.DataFrame(res)
P = R.pivot_table(index=['delay', 'thr', 'band', 'horizon'], columns='period', values=['n', 'roi', 'se']).dropna()
P.columns = [f'{a}_{b}' for a, b in P.columns]
print('\n=== rules: H1 vs H2 (model fitted on H1) ===\n' + P.round(4).to_string())
R.to_csv('results/a14_rules.csv', index=False)
Q.to_parquet(f'{HF}/a14_q.parquet', compression='zstd')
json.dump(dict(beta=b.tolist()), open('results/a14_crypto_ladders.json', 'w'))
