"""Fill-price realism check for the favourite strategy (READ-ONLY research).

STATUS: network stages (verify/fetch) are UNTESTED. This was written on a machine where the
Polymarket APIs are blocked. `sample` has run; `analyze` was only smoke-tested on synthetic trades.

Uses only public, unauthenticated GET endpoints (Gamma + data-api). Never places
orders, never uses keys, never touches wallets.

Question: backtests fill at price_last_le_dec + 0.01 (last CLOB prices-history
point at/before the decision time). Is that realistic? We compare it with the
prices at which takers actually bought YES around the decision time.

Run from the repo root, in this order (each stage is resumable; raw API
responses are cached under data/raw/fill_cache/, which is git-ignored):

  python src/fill_check.py sample     # offline: draws the stratified sample
  python src/fill_check.py verify     # fetches 3 markets, prints field semantics
  python src/fill_check.py fetch      # Gamma metadata + taker trades for the sample
  python src/fill_check.py analyze    # writes results/fill_check*.{json,csv.gz,md}

Environment overrides (used for offline smoke tests): FILL_CACHE_DIR, FILL_RESULTS_DIR.

Trade -> YES-equivalent mapping used (data-api /trades with takerOnly=true reports
the TAKER's side, the traded token in `asset`, and the price of that token):
  taker BUY  YES @x -> YES ask lifted at x       (kind=ybuy,  yes_px=x)
  taker SELL NO  @x -> taker sold NO into a bid at x == bought YES at 1-x
                                                  (kind=ybuy,  yes_px=1-x)
  taker SELL YES @x -> YES bid hit at x          (kind=ysell, yes_px=x)
  taker BUY  NO  @x -> bought NO at x == sold YES at 1-x
                                                  (kind=ysell, yes_px=1-x)
YES vs NO is decided by `asset` == clobTokenIds[index of "Yes" in outcomes]
(falls back to outcomeIndex if asset is missing); the agreement between the two
is reported in the diagnostics.
"""
import datetime as dt
import gzip
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'decision_prices.csv.gz'
CACHE = Path(os.environ.get('FILL_CACHE_DIR', ROOT / 'data' / 'raw' / 'fill_cache'))
RES = Path(os.environ.get('FILL_RESULTS_DIR', ROOT / 'results'))
SAMPLE = CACHE / 'sample.csv'

GAMMA = 'https://gamma-api.polymarket.com'
DAPI = 'https://data-api.polymarket.com'
PAGE = 500
MAX_OFFSET = 20000          # hard stop; data-api rejects very large offsets anyway
H = 3600
WINDOWS = {'pre1h': (-H, 0), 'pre6h': (-6 * H, 0), 'post1h': (0, H), 'post6h': (0, 6 * H)}
FAV_BANDS = [0.30, 0.40, 0.50, 0.60, 0.75, 0.95]
NONFAV_BANDS = [0.05, 0.15, 0.25, 0.40]
N_FAV, N_NONFAV, SEED = 1500, 500, 0


def band_label(p, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi == edges[-1]
        if lo <= p < hi or (last and p == hi):
            return f'{lo:.2f}-{hi:.2f}'
    return None


# ----------------------------------------------------------------------------- sample
def allocate(sizes, total):
    """Spread `total` evenly over strata; small strata are taken whole and the
    remainder is redistributed (water-filling)."""
    alloc, left, rem = {}, len(sizes), total
    for k, n in sorted(sizes.items(), key=lambda kv: (kv[1], kv[0])):
        take = min(n, rem // left if left else 0)
        alloc[k] = take
        rem -= take
        left -= 1
    return alloc


def strat_sample(df, keys, total, rng):
    df = df.sort_values('market_id')
    groups = list(df.groupby(keys))
    alloc = allocate({k: len(g) for k, g in groups}, total)
    parts = []
    for k, g in groups:
        n = alloc[k]
        if n:
            parts.append(g.iloc[np.sort(rng.choice(len(g), n, replace=False))])
    return pd.concat(parts)


def make_sample():
    d = pd.read_csv(DATA)
    d = d[d.price_age_s <= 120].copy()
    d['month'] = d.target_date.str[:7]
    fav_idx = d.groupby('event_id').price_last_le_dec.idxmax()
    fav = d.loc[fav_idx]
    fav = fav[fav.price_last_le_dec.between(0.30, 0.95) & fav.outcome_yes.isin([0, 1])].copy()
    fav['band'] = [band_label(p, FAV_BANDS) for p in fav.price_last_le_dec]
    fav['group'] = 'fav'
    non = d.drop(index=fav_idx)
    non = non[non.price_last_le_dec.between(0.05, 0.40) & non.outcome_yes.isin([0, 1])].copy()
    non['band'] = [band_label(p, NONFAV_BANDS) for p in non.price_last_le_dec]
    non['group'] = 'nonfav'
    rng = np.random.default_rng(SEED)
    s_fav = strat_sample(fav, ['month', 'band'], N_FAV, rng)
    s_non = strat_sample(non, ['month'], N_NONFAV, rng)
    s = pd.concat([s_fav, s_non]).reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    s.to_csv(SAMPLE, index=False)
    print(f'sample: {len(s_fav)} favourites (of {len(fav)}), {len(s_non)} non-favourites (of {len(non)}) -> {SAMPLE}')
    print(pd.crosstab(s_fav.month, s_fav.band, margins=True).to_string())
    return s


def load_sample():
    if not SAMPLE.exists():
        return make_sample()
    return pd.read_csv(SAMPLE)


# ----------------------------------------------------------------------------- http
class Http:
    def __init__(self, min_interval=0.15):
        self.s = requests.Session()
        self.s.headers['User-Agent'] = 'readonly-research/0.1'
        self.lock = threading.Lock()
        self.last = 0.0
        self.min_interval = min_interval

    def get(self, url, params=None, tries=8):
        for k in range(tries):
            with self.lock:                       # global throttle
                wait = self.last + self.min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            try:
                r = self.s.get(url, params=params, timeout=40)
            except requests.RequestException as ex:
                err = repr(ex)
                time.sleep(min(60, 2 ** k))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                err = f'HTTP {r.status_code}'
                time.sleep(min(60, 2 ** k + float(r.headers.get('Retry-After', 0) or 0)))
                continue
            return {'_error': r.status_code, '_body': r.text[:300]}   # 4xx: do not retry
        raise RuntimeError(f'giving up on {url} {params}: {err}')


def cached(path, fn):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return out


def gamma_market(http, mid):
    def f():
        j = http.get(f'{GAMMA}/markets/{mid}')
        if isinstance(j, dict) and '_error' in j:
            j2 = http.get(f'{GAMMA}/markets', params={'id': mid, 'closed': 'true'})
            if isinstance(j2, list) and j2:
                return j2[0]
        return j
    return cached(CACHE / 'gamma' / f'{mid}.json', f)


def yes_token(m):
    toks = m.get('clobTokenIds')
    outs = m.get('outcomes')
    toks = json.loads(toks) if isinstance(toks, str) else toks
    outs = json.loads(outs) if isinstance(outs, str) else outs
    if not toks:
        return None, None, None
    yi = 0
    if outs and 'Yes' in outs:
        yi = outs.index('Yes')
    return toks[yi], yi, toks


def trades_for(http, cid, stop_ts):
    """Taker trades newest-first; paginate until a page reaches older than stop_ts."""
    meta_p = CACHE / 'trades' / cid / 'meta.json'
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
    else:
        meta = None
    if meta is None or meta.get('stop_ts', 1e18) > stop_ts:
        off, truncated, reached = 0, False, False
        while True:
            page = cached(CACHE / 'trades' / cid / f'{off}.json',
                          lambda: http.get(f'{DAPI}/trades', params=dict(market=cid, limit=PAGE, offset=off, takerOnly='true')))
            if isinstance(page, dict):          # error (e.g. offset too large)
                truncated = not reached
                break
            if page and min(int(t['timestamp']) for t in page) < stop_ts:
                reached = True
            if len(page) < PAGE or reached:
                break
            off += PAGE
            if off > MAX_OFFSET:
                truncated = True
                break
        meta = dict(stop_ts=stop_ts, last_offset=off, truncated=truncated)
        meta_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.write_text(json.dumps(meta))
    trades = []
    for p in sorted((CACHE / 'trades' / cid).glob('[0-9]*.json'), key=lambda p: int(p.stem)):
        j = json.loads(p.read_text())
        if isinstance(j, list):
            trades += j
    return trades, meta


# ----------------------------------------------------------------------------- fetch
def fetch(workers=4):
    import concurrent.futures as cf
    s = load_sample()
    http = Http()
    done = [0]

    def one(r):
        m = gamma_market(http, int(r.market_id))
        cid = m.get('conditionId') if isinstance(m, dict) else None
        if not cid:
            return r.market_id, 'no conditionId'
        _, meta = trades_for(http, cid, int(r.decision_ts) - 6 * H)
        done[0] += 1
        if done[0] % 100 == 0:
            print(done[0], time.strftime('%X'), flush=True)
        return r.market_id, 'truncated' if meta['truncated'] else 'ok'

    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, [r for r in s.itertuples()]))
    st = pd.Series([x[1] for x in res]).value_counts()
    print(st.to_string())


# ----------------------------------------------------------------------------- convert
def to_yes(trades, yes_tok, yes_idx):
    rows, seen = [], set()
    diag = dict(n=0, asset_vs_outcomeIndex_mismatch=0, combos={})
    for t in trades:
        key = (t.get('transactionHash'), t.get('asset'), t.get('side'), t.get('price'), t.get('size'), t.get('timestamp'), t.get('proxyWallet'))
        if key in seen:
            continue
        seen.add(key)
        asset = t.get('asset')
        oi = t.get('outcomeIndex')
        if asset:
            is_yes = str(asset) == str(yes_tok)
            if oi is not None and (int(oi) == yes_idx) != is_yes:
                diag['asset_vs_outcomeIndex_mismatch'] += 1
        else:
            is_yes = int(oi) == yes_idx
        side = str(t['side']).upper()
        x, sz, ts = float(t['price']), float(t['size']), int(t['timestamp'])
        kind = 'ybuy' if (side == 'BUY') == is_yes else 'ysell'
        ypx = x if is_yes else 1.0 - x
        combo = f"{side}_{'YES' if is_yes else 'NO'}"
        diag['combos'][combo] = diag['combos'].get(combo, 0) + 1
        diag['n'] += 1
        rows.append((ts, kind, ypx, sz, sz * ypx, is_yes))
    df = pd.DataFrame(rows, columns=['ts', 'kind', 'ypx', 'shares', 'usdc', 'from_yes_token']).sort_values('ts', kind='stable')
    return df, diag


def market_row(r, df, meta):
    dec, p = int(r.decision_ts), float(r.price_last_le_dec)
    out = dict(market_id=r.market_id, event_id=r.event_id, city=r.city, target_date=r.target_date,
               month=r.target_date[:7], group=r.group, band=r.band, bucket=r.bucket,
               decision_ts=dec, p_last=p, outcome_yes=int(r.outcome_yes),
               n_trades_total=len(df), truncated=bool(meta.get('truncated')) if meta else None)
    rel = df.ts - dec if len(df) else pd.Series(dtype=float)
    for w, (a, b) in WINDOWS.items():
        m = (rel > a) & (rel <= b) if len(df) else []
        for kind, nm in (('ybuy', 'buy'), ('ysell', 'sell')):
            x = df[m & (df.kind == kind)] if len(df) else df
            out[f'n_{nm}_{w}'] = len(x)
            out[f'vol_{nm}_{w}'] = round(x.usdc.sum(), 4) if len(x) else 0.0
            out[f'vwap_{nm}_{w}'] = round((x.ypx * x.shares).sum() / x.shares.sum(), 5) if len(x) and x.shares.sum() > 0 else None
    for kind, nm in (('ybuy', 'buy'), ('ysell', 'sell')):
        post = df[(df.ts > dec) & (df.kind == kind)] if len(df) else df
        pre = df[(df.ts <= dec) & (df.ts > dec - 6 * H) & (df.kind == kind)] if len(df) else df
        out[f'first_{nm}_post_px'] = float(post.ypx.iloc[0]) if len(post) else None
        out[f'first_{nm}_post_dt_s'] = int(post.ts.iloc[0] - dec) if len(post) else None
        p6 = post[post.ts <= dec + 6 * H] if len(post) else post
        out[f'first_{nm}_post6h_px'] = float(p6.ypx.iloc[0]) if len(p6) else None
        out[f'last_{nm}_pre6h_px'] = float(pre.ypx.iloc[-1]) if len(pre) else None
    # proxies: VWAP within +-1h of decision, falling back to +-6h
    for nm in ('buy', 'sell'):
        v = None
        for half in (H, 6 * H):
            if len(df):
                x = df[(df.ts > dec - half) & (df.ts <= dec + half) & (df.kind == ('ybuy' if nm == 'buy' else 'ysell'))]
                if len(x) and x.shares.sum() > 0:
                    v = float((x.ypx * x.shares).sum() / x.shares.sum())
                    break
        out['ask_proxy' if nm == 'buy' else 'bid_proxy'] = v
    out['ask_minus_plast'] = out['ask_proxy'] - p if out['ask_proxy'] is not None else None
    out['bid_minus_plast'] = out['bid_proxy'] - p if out['bid_proxy'] is not None else None
    out['spread_proxy'] = (out['ask_proxy'] - out['bid_proxy']) if out['ask_proxy'] is not None and out['bid_proxy'] is not None else None
    out['first_buy_post6h_minus_plast'] = out['first_buy_post6h_px'] - p if out['first_buy_post6h_px'] is not None else None
    if len(df):
        x = df[(df.ts > dec - H) & (df.ts <= dec + H) & (df.kind == 'ybuy') & (df.ypx <= p + 0.02 + 1e-9)]
        out['vol_buy_le_p2_pm1h'] = round(x.usdc.sum(), 4)
    else:
        out['vol_buy_le_p2_pm1h'] = 0.0
    return out


# ----------------------------------------------------------------------------- verify
def verify(n=3):
    s = load_sample()
    http = Http()
    rows = s[s.group == 'fav'].sort_values('decision_ts').iloc[np.linspace(0, (s.group == 'fav').sum() - 1, n).astype(int)]
    for r in rows.itertuples():
        m = gamma_market(http, int(r.market_id))
        print('=' * 80)
        print('market', r.market_id, r.city, r.target_date, r.bucket, 'p_last', r.price_last_le_dec, 'outcome', r.outcome_yes)
        print('gamma keys:', sorted(m.keys())[:60])
        print('question:', m.get('question'), '| conditionId:', m.get('conditionId'))
        print('outcomes:', m.get('outcomes'), '| clobTokenIds:', m.get('clobTokenIds'))
        ytok, yi, toks = yes_token(m)
        trades, meta = trades_for(http, m['conditionId'], int(r.decision_ts) - 6 * H)
        print('n trades', len(trades), meta)
        if trades:
            print('trade keys:', sorted(trades[0].keys()))
            for t in trades[:3]:
                print({k: t.get(k) for k in ('side', 'outcome', 'outcomeIndex', 'asset', 'price', 'size', 'timestamp', 'proxyWallet', 'transactionHash')})
            print('newest ts', trades[0]['timestamp'], 'oldest ts', trades[-1]['timestamp'], '(newest-first?', int(trades[0]['timestamp']) >= int(trades[-1]['timestamp']), ')')
        df, diag = to_yes(trades, ytok, yi)
        print('diag', diag)
        if len(df):
            print('assets seen subset of clobTokenIds:', set(str(t.get('asset')) for t in trades) <= set(map(str, toks)))
            near = df[(df.ts > r.decision_ts - 6 * H) & (df.ts <= r.decision_ts + 6 * H)]
            print('+-6h YES-equiv median price by kind & source token:')
            print(near.groupby(['kind', 'from_yes_token']).ypx.agg(['count', 'median']).to_string())


# ----------------------------------------------------------------------------- analyze
def roi(entry, payout):
    entry, payout = np.asarray(entry, float), np.asarray(payout, float)
    if len(entry) == 0:
        return dict(n=0)
    return dict(n=int(len(entry)), wins=int(payout.sum()), win_rate=round(float(payout.mean()), 4),
                mean_entry=round(float(entry.mean()), 4),
                roi=round(float((payout.sum() - entry.sum()) / entry.sum()), 4))


def boot_ci(entry, payout, reps=2000, seed=0):
    entry, payout = np.asarray(entry, float), np.asarray(payout, float)
    if len(entry) < 10:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(entry), (reps, len(entry)))
    e, p = entry[idx].sum(1), payout[idx].sum(1)
    r = (p - e) / e
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def agg_block(g):
    d = g.first_buy_post6h_minus_plast.dropna()
    out = dict(n=int(len(g)), n_with_any_trade_pm6h=int(((g.n_buy_pre6h + g.n_sell_pre6h + g.n_buy_post6h + g.n_sell_post6h) > 0).sum()),
               share_buy_post1h=round(float((g.n_buy_post1h > 0).mean()), 4),
               share_buy_post6h=round(float((g.n_buy_post6h > 0).mean()), 4),
               firstbuy6h_minus_plast_median=round(float(d.median()), 4) if len(d) else None,
               firstbuy6h_minus_plast_mean=round(float(d.mean()), 4) if len(d) else None,
               firstbuy6h_minus_plast_p25=round(float(d.quantile(.25)), 4) if len(d) else None,
               firstbuy6h_minus_plast_p75=round(float(d.quantile(.75)), 4) if len(d) else None,
               share_firstbuy6h_gt_plast_plus_1c=round(float((d > 0.01 + 1e-9).mean()), 4) if len(d) else None,
               ask_minus_plast_median=round(float(g.ask_minus_plast.median()), 4) if g.ask_minus_plast.notna().any() else None,
               bid_minus_plast_median=round(float(g.bid_minus_plast.median()), 4) if g.bid_minus_plast.notna().any() else None,
               spread_proxy_median=round(float(g.spread_proxy.median()), 4) if g.spread_proxy.notna().any() else None,
               spread_proxy_mean=round(float(g.spread_proxy.mean()), 4) if g.spread_proxy.notna().any() else None,
               vol_buy_le_p2_pm1h_median=round(float(g.vol_buy_le_p2_pm1h.median()), 2),
               share_vol_buy_le_p2_pm1h_ge_5usdc=round(float((g.vol_buy_le_p2_pm1h >= 5).mean()), 4))
    return out


def strategy_block(g):
    g = g[g.group == 'fav']
    has = g[g.first_buy_post6h_px.notna()]
    out = dict(
        modeled_all=roi(g.p_last + 0.01, g.outcome_yes),
        modeled_same=roi(has.p_last + 0.01, has.outcome_yes),
        firstbuy6h_same=roi(has.first_buy_post6h_px, has.outcome_yes),
        firstbuy6h_same_ci95=boot_ci(has.first_buy_post6h_px, has.outcome_yes),
        modeled_same_ci95=boot_ci(has.p_last + 0.01, has.outcome_yes))
    v = g[g.vwap_buy_post1h.notna()]
    out['vwap_post1h_same'] = roi(v.vwap_buy_post1h, v.outcome_yes)
    out['modeled_on_vwap_post1h_markets'] = roi(v.p_last + 0.01, v.outcome_yes)
    return out


def md_table(rows, cols, title):
    s = [f'\n### {title}\n', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        s.append('| ' + ' | '.join('' if v is None else (f'{v:.4f}' if isinstance(v, float) else str(v)) for v in r) + ' |')
    return '\n'.join(s)


def analyze():
    s = load_sample()
    rows, diag_tot, missing = [], dict(n=0, asset_vs_outcomeIndex_mismatch=0, combos={}), []
    for r in s.itertuples():
        gp = CACHE / 'gamma' / f'{int(r.market_id)}.json'
        if not gp.exists():
            missing.append(r.market_id)
            continue
        m = json.loads(gp.read_text())
        cid = m.get('conditionId')
        ytok, yi, _ = yes_token(m) if cid else (None, None, None)
        if not cid or ytok is None or not (CACHE / 'trades' / cid / 'meta.json').exists():
            missing.append(r.market_id)
            continue
        trades, meta = trades_for(None, cid, int(r.decision_ts) - 6 * H)
        df, diag = to_yes(trades, ytok, yi)
        diag_tot['n'] += diag['n']
        diag_tot['asset_vs_outcomeIndex_mismatch'] += diag['asset_vs_outcomeIndex_mismatch']
        for k, v in diag['combos'].items():
            diag_tot['combos'][k] = diag_tot['combos'].get(k, 0) + v
        row = market_row(r, df, meta)
        row['condition_id'] = cid
        # semantic check: are prices from NO-token trades consistent with YES-token trades?
        near = df[(df.ts > r.decision_ts - 6 * H) & (df.ts <= r.decision_ts + 6 * H)] if len(df) else df
        for kind in ('ybuy', 'ysell'):
            a = near[(near.kind == kind) & near.from_yes_token].ypx if len(near) else []
            b = near[(near.kind == kind) & ~near.from_yes_token].ypx if len(near) else []
            row[f'{kind}_yes_tok_median'] = float(np.median(a)) if len(a) else None
            row[f'{kind}_no_tok_median'] = float(np.median(b)) if len(b) else None
        rows.append(row)
    M = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    M.to_csv(RES / 'fill_check_markets.csv.gz', index=False, compression='gzip')

    fav, non = M[M.group == 'fav'], M[M.group == 'nonfav']
    both = M[M.vwap_buy_post6h.notna() & M.vwap_sell_post6h.notna()]
    yn = M[M.ybuy_yes_tok_median.notna() & M.ybuy_no_tok_median.notna()]
    semantics = dict(
        trades_total=diag_tot['n'], side_token_combos=diag_tot['combos'],
        asset_vs_outcomeIndex_mismatch=diag_tot['asset_vs_outcomeIndex_mismatch'],
        share_markets_vwap_buy_gt_sell_post6h=round(float((both.vwap_buy_post6h > both.vwap_sell_post6h).mean()), 4) if len(both) else None,
        median_vwap_buy_minus_sell_post6h=round(float((both.vwap_buy_post6h - both.vwap_sell_post6h).median()), 4) if len(both) else None,
        n_markets_buy_and_sell_post6h=int(len(both)),
        median_ybuy_yes_token_minus_no_token=round(float((yn.ybuy_yes_tok_median - yn.ybuy_no_tok_median).median()), 4) if len(yn) else None,
        n_markets_ybuy_both_tokens=int(len(yn)),
        markets_missing=len(missing), markets_truncated=int(M.truncated.fillna(False).sum()))
    J = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        sample=dict(n_fav=int(len(fav)), n_nonfav=int(len(non)), seed=SEED,
                    note='favourites stratified evenly over month x price band; ROI below is an equal-weight over that sample, not the population mix'),
        mapping={'BUY YES @x': 'ybuy @x', 'SELL NO @x': 'ybuy @1-x', 'SELL YES @x': 'ysell @x', 'BUY NO @x': 'ysell @1-x'},
        semantics_check=semantics,
        fav_by_band={b: agg_block(g) for b, g in fav.groupby('band')},
        fav_by_month={b: agg_block(g) for b, g in fav.groupby('month')},
        fav_all=agg_block(fav),
        nonfav_by_band={b: agg_block(g) for b, g in non.groupby('band')},
        nonfav_all=agg_block(non) if len(non) else None,
        strategy_by_band={b: strategy_block(g) for b, g in fav.groupby('band')},
        strategy_all=strategy_block(fav),
        strategy_v1_rule=strategy_block(fav[(fav.p_last + 0.01 >= 0.60) & (fav.p_last + 0.01 < 0.90)]),
    )
    (RES / 'fill_check.json').write_text(json.dumps(J, indent=1))
    write_md(J)
    print((RES / 'fill_check.md').read_text())


def write_md(J):
    cols5 = ['group', 'n', 'buy<=1h', 'buy<=6h', 'med(first-p)', 'mean(first-p)', 'share first>p+1c', 'med ask-p', 'med bid-p', 'med spread', 'med vol<=p+2c +-1h $', 'share vol>=5$']

    def r5(name, a):
        return [name, a['n'], a['share_buy_post1h'], a['share_buy_post6h'], a['firstbuy6h_minus_plast_median'], a['firstbuy6h_minus_plast_mean'],
                a['share_firstbuy6h_gt_plast_plus_1c'], a['ask_minus_plast_median'], a['bid_minus_plast_median'], a['spread_proxy_median'],
                a['vol_buy_le_p2_pm1h_median'], a['share_vol_buy_le_p2_pm1h_ge_5usdc']]
    t5b = md_table([r5(k, v) for k, v in J['fav_by_band'].items()] + [r5('ALL fav', J['fav_all'])], cols5, 'Step 5 - favourites by price band')
    t5m = md_table([r5(k, v) for k, v in J['fav_by_month'].items()], cols5, 'Step 5 - favourites by month')
    t5n = md_table([r5(k, v) for k, v in J['nonfav_by_band'].items()] + ([r5('ALL nonfav', J['nonfav_all'])] if J['nonfav_all'] else []), cols5, 'Step 5 - non-favourites by price band')
    cols6 = ['band', 'n(all)', 'ROI p+0.01 (all)', 'n(with buy<=6h)', 'ROI p+0.01 (same)', 'ROI first buy<=6h', 'CI95 first buy', 'mean entry p+0.01', 'mean entry first buy', 'win rate']

    def r6(name, b):
        a, m, f = b['modeled_all'], b['modeled_same'], b['firstbuy6h_same']
        return [name, a.get('n'), a.get('roi'), f.get('n'), m.get('roi'), f.get('roi'), str(b['firstbuy6h_same_ci95']), m.get('mean_entry'), f.get('mean_entry'), f.get('win_rate')]
    t6 = md_table([r6(k, v) for k, v in J['strategy_by_band'].items()] + [r6('ALL', J['strategy_all']), r6('v1 rule 0.60<=p+0.01<0.90', J['strategy_v1_rule'])], cols6, 'Step 6 - favourite strategy: modeled fill vs first real taker buy')
    sem = J['semantics_check']
    txt = f"""# Fill-price realism check

Generated {J['generated_utc']} by `src/fill_check.py`. Sample: {J['sample']['n_fav']} favourites, {J['sample']['n_nonfav']} non-favourites (seed {J['sample']['seed']}). {J['sample']['note']}.

Source: data-api `/trades?takerOnly=true`, converted to YES-equivalent prices
(BUY YES @x and SELL NO @x -> YES buy at x / 1-x; SELL YES @x and BUY NO @x -> YES sell at x / 1-x).
`p` = price_last_le_dec. "first" = first YES-equivalent taker buy within 6h after the decision time.
ask/bid proxy = VWAP of YES-equivalent taker buys/sells within +-1h of the decision (fallback +-6h).

Semantics check: {sem['trades_total']} trades, side/token combos {sem['side_token_combos']}, asset-vs-outcomeIndex mismatches {sem['asset_vs_outcomeIndex_mismatch']};
VWAP(YES buys) > VWAP(YES sells) in {sem['share_markets_vwap_buy_gt_sell_post6h']} of {sem['n_markets_buy_and_sell_post6h']} markets (median diff {sem['median_vwap_buy_minus_sell_post6h']});
YES-buy price from YES-token trades minus from NO-token trades, median {sem['median_ybuy_yes_token_minus_no_token']} over {sem['n_markets_ybuy_both_tokens']} markets.
Missing markets: {sem['markets_missing']}, truncated trade histories: {sem['markets_truncated']}.
{t5b}
{t5m}
{t5n}
{t6}
"""
    (RES / 'fill_check.md').write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'analyze'
    if cmd == 'sample':
        make_sample()
    elif cmd == 'verify':
        verify(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == 'fetch':
        fetch(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    elif cmd == 'analyze':
        analyze()
    else:
        print(__doc__)
