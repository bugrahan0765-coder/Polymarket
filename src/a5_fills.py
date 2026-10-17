"""Step 5: real execution on weather markets from on-chain fills (hf_extract.py output).
For every bucket at the v1 decision time: first taker-buy (ask hit) / taker-sell (bid hit) after the decision,
then (a) taker strategy with entry at the real ask + fee, (b) maker strategy: resting bid at p - d, filled only if
takers later sold through it, with adverse-selection check (win rate of filled vs unfilled).
Run: python src/a5_fills.py <ext_dir>
"""
import sys, glob, json, numpy as np, pandas as pd
from analysis_lib import load_markets

EXT = sys.argv[1] if len(sys.argv) > 1 else '../hf_data/ext'
FEE_START = pd.Timestamp('2026-03-30', tz='UTC').timestamp()
FEE_RATE = .05
CAP = 5.


def fee(p, ts):  # taker fee per share
    return np.where(ts >= FEE_START, FEE_RATE * p * (1 - p), 0.)


m = load_markets()
m = m[m['yes'].notna() & m['p'].notna()].copy()
m['market_id'] = m['market_id'].astype(str)
m['rank'] = m.groupby('event_id')['p'].rank(ascending=False, method='first')
m['fav'] = m['rank'] == 1
dec = m.set_index('market_id')['decision_ts']

w = pd.concat([pd.read_parquet(f, columns=['timestamp', 'market_id', 'p1', 'buy1', 'usd_amount']) for f in glob.glob(f'{EXT}/weather_trades/*.parquet')])
w = w[w['market_id'].isin(dec.index)]
w['rel'] = w['timestamp'] - w['market_id'].map(dec)
w = w.sort_values(['market_id', 'timestamp'])
print(f'{len(w):,} fills on {w.market_id.nunique():,} of {len(m):,} priced buckets')

after = w[w['rel'] > 0]
H = 6 * 3600


def first(df, buy):
    x = df[(df['buy1'] == buy) & (df['rel'] <= H)]
    return x.groupby('market_id').agg(px=('p1', 'first'), dt=('rel', 'first'))


ask = first(after, True).add_prefix('ask_'); bid = first(after, False).add_prefix('bid_')
m = m.join(ask, on='market_id').join(bid, on='market_id')
m['ask_minus_p'] = m['ask_px'] - m['p']; m['bid_minus_p'] = m['bid_px'] - m['p']
BANDS = [0, .1, .2, .3, .4, .5, .6, .75, .9, 1.01]
m['band'] = pd.cut(m['p'], BANDS, right=False)
out = {}

s = m.groupby(['fav', 'band'], observed=True).agg(n=('p', 'size'), has_ask=('ask_px', lambda x: x.notna().mean()),
                                                 ask_minus_p_med=('ask_minus_p', 'median'), ask_minus_p_mean=('ask_minus_p', 'mean'),
                                                 bid_minus_p_med=('bid_minus_p', 'median'), ask_delay_min=('ask_dt', lambda x: x.median() / 60))
print('\n=== real ask/bid vs decision price (first taker trade within 6h after decision) ===\n' + s.round(4).to_string())
out['ask_bid'] = s.round(4).reset_index().astype({'band': str}).to_dict('records')


def taker_roi(df, entry, ts):
    sh = CAP / entry; cost = CAP + sh * fee(entry, ts)
    pnl = np.where(df['yes'] == 1, sh, 0) - cost
    return pnl.sum() / cost.sum(), len(df)


# taker strategy: favourites in band, entry at real ask vs p+0.01 on the same markets
rows = []
for lo, hi in [(.3, .9), (.4, .5), (.6, .9), (.6, .75)]:
    f = m[m['fav'] & (m['p'] >= lo) & (m['p'] < hi) & m['ask_px'].notna() & (m['ask_px'] < .99)]
    for half, g in [('all', f), ('<Apr', f[f.decision_ts < FEE_START]), ('>=Apr', f[f.decision_ts >= FEE_START])]:
        if len(g) < 20: continue
        r_ask, n = taker_roi(g, g['ask_px'].to_numpy(), g['decision_ts'].to_numpy())
        r_p1, _ = taker_roi(g, (g['p'] + .01).to_numpy(), g['decision_ts'].to_numpy())
        rows.append(dict(band=f'{lo}-{hi}', period=half, n=n, roi_real_ask=round(r_ask, 4), roi_p_plus_1c=round(r_p1, 4),
                         win=round(g['yes'].mean(), 4), mean_ask=round(g['ask_px'].mean(), 4), mean_p=round(g['p'].mean(), 4)))
t = pd.DataFrame(rows); print('\n=== TAKER favourite strategy (with fee from 2026-03-30) ===\n' + t.to_string(index=False))
out['taker'] = rows

# maker strategy: bid resting at b = p - d from decision for W hours; filled if taker sells trade at p1 <= b (or < b, conservative)
sells = after[~after['buy1']]
rows = []
for d in [0., .01, .02, .03]:
    for W in [1, 6, 24, 48]:
        x = sells[sells['rel'] <= W * 3600][['market_id', 'p1', 'usd_amount']]
        b = x['market_id'].map(m.set_index('market_id')['p'] - d)
        hit_le = x[x['p1'] <= b + 1e-9].groupby('market_id')['usd_amount'].sum()
        hit_lt = x[x['p1'] < b - 1e-9].groupby('market_id')['usd_amount'].sum()
        for who, sub in [('fav', m[m['fav'] & (m['p'] >= .3) & (m['p'] < .9)]), ('fav40-50', m[m['fav'] & (m['p'] >= .4) & (m['p'] < .5)]),
                         ('nonfav10-30', m[~m['fav'] & (m['p'] >= .1) & (m['p'] < .3)])]:
            sub = sub.assign(b=sub['p'] - d)
            for mode, hit in [('touch', hit_le), ('through', hit_lt)]:
                filled = sub['market_id'].isin(hit.index)
                fsub = sub[filled]
                if len(fsub) < 20: continue
                pnl = np.where(fsub['yes'] == 1, CAP / fsub['b'], 0) - CAP
                rows.append(dict(offset=d, hours=W, who=who, mode=mode, n=len(sub), fill_rate=round(filled.mean(), 3),
                                 win_filled=round(fsub['yes'].mean(), 4), win_unfilled=round(sub[~filled]['yes'].mean(), 4),
                                 mean_b=round(fsub['b'].mean(), 4), roi=round(pnl.sum() / (CAP * len(fsub)), 4), pnl=round(pnl.sum(), 1)))
t = pd.DataFrame(rows); print('\n=== MAKER: resting bid at p-offset from decision (no fee) ===\n' + t.to_string(index=False))
out['maker'] = rows
json.dump(out, open('results/a5_fills.json', 'w'), indent=1, default=str)
