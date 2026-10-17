"""UNTESTED. Cross-category calibration pipeline (workstream `calibration`).

Never run: Polymarket APIs are blocked on the machine this was written on. The code needs a
machine with legitimate access. Read-only public endpoints only. No orders, keys or wallets.

Steps (each step caches raw responses under data/calibration/raw/):
  python src/calibration/calib_pipeline.py markets   # Gamma closed binary markets, last ~15 months
  python src/calibration/calib_pipeline.py prices    # CLOB prices-history -> price at 30d/7d/3d/1d/6h/1h before end
  python src/calibration/calib_pipeline.py trades    # data-api taker trades around horizons (realistic entry, capacity)
  python src/calibration/calib_pipeline.py tables    # calibration tables + bond / longshot-NO cell stats

Design notes:
- Horizons are anchored on the SCHEDULED endDate (known in advance, so a strategy could act on it).
  An observation is dropped if the market closed (closedTime) before the horizon time.
  Capital lock-up = closedTime - horizon time.
- Price at a horizon = last prices-history point at or before T. Its age is stored, and points
  older than max(1h, 10% of the horizon) are flagged stale.
- Realistic entry = VWAP of taker BUYs of the chosen outcome in [T, T+1h] (fall back to [T, T+6h]).
  Capacity = USDC taker-bought within 2c of the entry price in that window.
  The data-api side/outcome semantics MUST be checked by hand on a few markets before trusting this.
- Fees: look up the current schedule on docs.polymarket.com and fill in FEE_RULES (not done:
  the docs are blocked here too). Placeholder = 0, so any ROI from this code is gross of fees.
"""
import json, os, sys, time, gzip, random
import datetime as dt
import numpy as np, pandas as pd, requests

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
DATA = 'https://data-api.polymarket.com'
OUT = 'data/calibration'
RAW = f'{OUT}/raw'
HORIZONS = {'30d': 30 * 86400, '7d': 7 * 86400, '3d': 3 * 86400, '1d': 86400, '6h': 6 * 3600, '1h': 3600}
START = dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc)
END = dt.datetime.now(dt.timezone.utc)
FEE_RULES = {}  # TODO: category/date -> taker fee function f(price) per share, from docs.polymarket.com

S = requests.Session()


def get(url, params=None, cache=None, tries=8):
    if cache and os.path.exists(cache):
        with gzip.open(cache, 'rt') as f:
            return json.load(f)
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=40)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** k)); continue
            r.raise_for_status(); js = r.json()
            if cache:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                with gzip.open(cache, 'wt') as f:
                    json.dump(js, f)
            return js
        except requests.RequestException:
            time.sleep(min(60, 2 ** k))
    return None


CAT_KEYS = [  # (category, keywords searched in tag labels/slugs then question); first match wins
    ('weather', ['weather', 'temperature', 'hurricane', 'rain', 'snow']),
    ('crypto', ['crypto', 'bitcoin', 'ethereum', 'solana', 'btc', 'eth ', 'xrp', 'up or down']),
    ('sports', ['sports', 'nba', 'nfl', 'mlb', 'nhl', 'soccer', 'football', 'tennis', 'ufc', 'esports', 'epl', 'champions league', ' vs. ', ' vs ']),
    ('politics', ['politics', 'election', 'elections', 'trump', 'president', 'senate', 'congress', 'geopolitics', 'world']),
    ('economics', ['economy', 'economics', 'fed', 'inflation', 'cpi', 'rates', 'gdp', 'finance', 'stocks', 'earnings', 'business']),
    ('culture', ['culture', 'pop-culture', 'movies', 'music', 'tech', 'ai', 'mentions', 'tweets', 'science']),
]


def category(m):
    tags = ' '.join((t.get('label', '') + ' ' + t.get('slug', '')).lower() for e in m.get('events', []) or [] for t in (e.get('tags') or []))
    q = (m.get('question') or '').lower()
    for src in (tags, q):
        for c, kws in CAT_KEYS:
            if any(k in src for k in kws):
                return c
    return 'other'


def cmd_markets():
    rows = []; d = START
    while d < END:
        d2 = d + dt.timedelta(days=1); off = 0
        while True:
            p = dict(closed='true', limit=500, offset=off, end_date_min=d.isoformat(), end_date_max=d2.isoformat())
            js = get(f'{GAMMA}/markets', p, cache=f'{RAW}/gamma/{d:%Y%m%d}_{off}.json.gz')
            if not js: break
            for m in js:
                try:
                    outs = json.loads(m.get('outcomes') or '[]'); px = [float(x) for x in json.loads(m.get('outcomePrices') or '[]')]
                    toks = json.loads(m.get('clobTokenIds') or '[]')
                except Exception:
                    continue
                if len(outs) != 2 or len(toks) != 2 or len(px) != 2: continue
                if sorted(px) != [0.0, 1.0]: continue  # drops 50/50 and ambiguous resolutions
                rows.append(dict(id=m['id'], condition_id=m.get('conditionId'), question=m.get('question'),
                                 yes_token=toks[0], no_token=toks[1], yes_won=int(px[0] == 1.0),
                                 end_ts=pd.Timestamp(m['endDate']).timestamp() if m.get('endDate') else np.nan,
                                 closed_ts=pd.Timestamp(m['closedTime']).timestamp() if m.get('closedTime') else np.nan,
                                 volume=float(m.get('volumeNum') or m.get('volume') or 0), liquidity=float(m.get('liquidityNum') or 0),
                                 neg_risk=bool(m.get('negRisk')), event_id=(m.get('events') or [{}])[0].get('id'), category=category(m)))
            if len(js) < 500: break
            off += 500
        d = d2
    df = pd.DataFrame(rows).drop_duplicates('id')
    os.makedirs(OUT, exist_ok=True); df.to_parquet(f'{OUT}/markets.parquet'); print(len(df), df.category.value_counts().to_dict())


def sample_markets(df, n=25000, seed=0):
    """Stratified by category x volume bucket; markets with >$10k volume are taken first."""
    df = df.copy(); df['vb'] = pd.cut(df.volume, [-1, 1e3, 1e4, 1e5, 1e12], labels=['<1k', '1-10k', '10-100k', '>100k'])
    hi = df[df.volume > 1e4]
    if len(hi) >= n:
        return hi.groupby('category', group_keys=False).apply(lambda g: g.sample(min(len(g), max(500, n * len(g) // len(hi))), random_state=seed))
    rest = df[df.volume <= 1e4]
    k = n - len(hi)
    return pd.concat([hi, rest.groupby(['category', 'vb'], group_keys=False, observed=True).apply(lambda g: g.sample(min(len(g), max(50, k * len(g) // len(rest))), random_state=seed))])


def price_at(hist, T):
    pts = [h for h in hist if h['t'] <= T]
    if not pts: return np.nan, np.nan
    h = pts[-1]; return float(h['p']), T - h['t']


def cmd_prices():
    df = sample_markets(pd.read_parquet(f'{OUT}/markets.parquet')); rows = []
    for m in df.itertuples():
        t0 = int(m.end_ts - 31 * 86400)
        js = get(f'{CLOB}/prices-history', dict(market=m.yes_token, startTs=t0, endTs=int(m.end_ts), fidelity=10), cache=f'{RAW}/px/{m.yes_token[-16:]}.json.gz')
        hist = (js or {}).get('history', [])
        for h, sec in HORIZONS.items():
            T = m.end_ts - sec
            if not np.isnan(m.closed_ts) and m.closed_ts <= T: continue  # market already closed: not tradable
            p, age = price_at(hist, T)
            rows.append(dict(id=m.id, event_id=m.event_id, category=m.category, volume=m.volume, neg_risk=m.neg_risk, horizon=h, T=T,
                             p=p, age=age, stale=age > max(3600, 0.1 * sec) if age == age else True, yes_won=m.yes_won,
                             lock_days=((m.closed_ts if m.closed_ts == m.closed_ts else m.end_ts) - T) / 86400, condition_id=m.condition_id))
    pd.DataFrame(rows).to_parquet(f'{OUT}/horizon_prices.parquet')


def cmd_trades(window=3600):
    """Taker trades for the candidate cells only (the bond and longshot rows), within [T, T+window]."""
    hp = pd.read_parquet(f'{OUT}/horizon_prices.parquet')
    cand = hp[(~hp.stale) & (((hp.p >= .90) & (hp.p < .995)) | ((hp.p > .005) & (hp.p <= .10)))]
    out = []
    for cid, g in cand.groupby('condition_id'):
        trades = []; off = 0
        while off < 10000:  # assumed data-api offset cap; check it
            js = get(f'{DATA}/trades', dict(market=cid, takerOnly='true', limit=500, offset=off), cache=f'{RAW}/tr/{cid[-16:]}_{off}.json.gz')
            if not js: break
            trades += js
            if len(js) < 500: break
            off += 500
        tr = pd.DataFrame(trades)
        if tr.empty: continue
        # Expected fields: timestamp, side (BUY/SELL by taker), outcome / outcomeIndex, price, size. VERIFY on a few markets.
        for r in g.itertuples():
            side_out = 0 if r.p >= .5 else 1  # favourite -> buy YES (index 0); longshot -> buy NO (index 1)
            w = tr[(tr.timestamp >= r.T) & (tr.timestamp < r.T + window) & (tr.side == 'BUY') & (tr.outcomeIndex == side_out)]
            if w.empty: continue
            vwap = (w.price * w['size']).sum() / w['size'].sum()
            cap = (w.price * w['size'])[w.price <= vwap + .02].sum()
            out.append(dict(id=r.id, horizon=r.horizon, entry=vwap, capacity_usdc=cap, n_fills=len(w)))
    pd.DataFrame(out).to_parquet(f'{OUT}/entries.parquet')


def cluster_boot(df, val, cl, n=2000, seed=0):
    rng = np.random.default_rng(seed); gs = [g[val].values for _, g in df.groupby(cl)]
    if not gs: return (np.nan, np.nan)
    st = [np.concatenate([gs[i] for i in rng.integers(0, len(gs), len(gs))]).mean() for _ in range(n)]
    return tuple(np.percentile(st, [2.5, 97.5]))


def cmd_tables():
    hp = pd.read_parquet(f'{OUT}/horizon_prices.parquet'); hp = hp[~hp.stale & hp.p.notna()]
    bins = [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .97, .99, 1.0001]
    hp['bin'] = pd.cut(hp.p, bins, right=False)
    t = hp.groupby(['category', 'horizon', 'bin'], observed=True).agg(n=('p', 'size'), mean_p=('p', 'mean'), win=('yes_won', 'mean'))
    t['edge_yes'] = t.win - t.mean_p
    os.makedirs('results/calibration', exist_ok=True); t.to_csv('results/calibration/calibration_table.csv')
    # Bond cell (buy YES at 0.90-0.99) and longshot-NO cell (YES at 0.01-0.10, so buy NO), at last price + 1c.
    # Time split: thresholds are chosen before the split date and tested after it.
    hp['month'] = pd.to_datetime(hp['T'], unit='s').dt.to_period('M')
    for name, sel, entry, win in [('bond', (hp.p >= .90) & (hp.p < .99), hp.p + .01, hp.yes_won == 1),
                                  ('longshot_no', (hp.p > .01) & (hp.p <= .10), 1 - hp.p + .01, hp.yes_won == 0)]:
        c = hp[sel].copy(); c['entry'] = entry[sel]; c['ret'] = np.where(win[sel], 1 / c.entry - 1, -1)
        c['ret_per_day'] = c.ret / c.lock_days.clip(lower=1 / 24)
        s = c.groupby(['horizon', 'category']).agg(n=('ret', 'size'), roi=('ret', 'mean'), loss_rate=('ret', lambda x: (x < 0).mean()),
                                                    lock_days=('lock_days', 'median'))
        s.to_csv(f'results/calibration/{name}_cells.csv')


if __name__ == '__main__':
    {'markets': cmd_markets, 'prices': cmd_prices, 'trades': cmd_trades, 'tables': cmd_tables}[sys.argv[1]]()
