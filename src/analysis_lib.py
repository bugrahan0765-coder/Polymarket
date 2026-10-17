"""Shared helpers for the analysis scripts: load decision prices into a DataFrame and simulate v1.
Independent re-implementation of backtest_csv.py (same rules) so results can be cross-checked.
"""
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from common import TZ

DATA = 'data/decision_prices.csv.gz'


def load_markets(path=DATA, win=120):
    """One row per market (bucket). p = decision price if fresh (age <= win s), else NaN."""
    df = pd.read_csv(path, dtype={'outcome_yes': 'string', 'closed_time': 'string', 'market_id': 'string', 'event_id': 'string'})
    df['p'] = np.where(df['price_age_s'].fillna(1e18) <= win, df['price_last_le_dec'], np.nan)
    df['yes'] = df['outcome_yes'].map({'1': 1, '0': 0})
    df['target_date'] = pd.to_datetime(df['target_date']).dt.date
    return df


def release_ts(city, d, closed):
    tz = ZoneInfo(TZ[city])
    rel = (dt.datetime(d.year, d.month, d.day, tzinfo=tz) + dt.timedelta(days=1)).timestamp()
    if isinstance(closed, str) and closed:
        rel = max(rel, dt.datetime.fromisoformat(closed.replace('+00', '+00:00')).timestamp())
    return rel


def favorites(m):
    """One row per event: the highest-priced bucket among fresh-priced ones (first on ties, like v1)."""
    pr = m[m['p'].notna()]
    idx = pr.groupby('event_id', sort=False)['p'].idxmax()
    f = pr.loc[idx].copy()
    f = f[f['yes'].notna()]
    f['win'] = f['yes'] == 1
    f['rel'] = [release_ts(c, d, cl) for c, d, cl in zip(f['city'], f['target_date'], f['closed_time'])]
    f['month'] = pd.to_datetime(f['target_date']).dt.to_period('M')
    return f.reset_index(drop=True)


def sim(f, start=50., cap=5., slip=.01, lo=.6, hi=.9, fee=0., cash_limited=True, minshares=5):
    """Same rules as backtest_csv.sim. fee: rate in shares*rate*p*(1-p). Returns (trades DataFrame, end balance)."""
    cash = start; open_ = []; rows = []
    fs = f.sort_values('decision_ts', kind='stable')
    for r in fs.itertuples(index=False):
        still = []
        for o in open_:
            if o[0] <= r.decision_ts: cash += o[1]
            else: still.append(o)
        open_ = still
        bp = round(r.p + slip, 4)
        if not (lo <= bp < hi): continue
        budget = min(cap, cash) if cash_limited else cap
        sh = budget / bp
        if sh < minshares: continue
        cost = sh * bp; fe = sh * fee * bp * (1 - bp); cash -= cost + fe
        payout = sh if r.win else 0.
        open_.append((r.rel, payout))
        rows.append(dict(event_id=r.event_id, city=r.city, target_date=r.target_date, month=r.month, p=r.p, bp=bp,
                         sh=sh, cost=cost + fe, payout=payout, pnl=payout - cost - fe, win=r.win))
    return pd.DataFrame(rows), cash + sum(o[1] for o in open_)


def roi(t):
    return t['pnl'].sum() / t['cost'].sum() if len(t) else np.nan


# --- all-Polymarket helpers (HF on-chain data) ---
import re
# taker fee rate x p(1-p) per share (third-party summaries of the Polymarket fee docs, from 2026-03-30); applied to the whole period
FEE_RATE = {'crypto': .07, 'sports': .03, 'weather': .05, 'politics': .04, 'finance': .04, 'geopolitics': 0., 'other': .05}


def category(event_slug, question, answer1):
    s = f"{event_slug or ''} {question or ''}".lower(); a = (answer1 or '').lower()
    if 'temperature' in s or 'weather' in s: return 'weather'
    if a == 'up' or re.search(r'bitcoin|ethereum|solana|\bxrp\b|\bbtc\b|\beth\b|crypto|dogecoin|\bbnb\b|hyperliquid', s): return 'crypto'
    if a.startswith('over') or re.search(r'\bvs\.?\b|nba|nfl|mlb|nhl|epl|premier-league|la-liga|serie-a|bundesliga|ligue-1|ucl|uefa|fifa|atp|wta|'
                                         r'tennis|ufc|esports|lol-|cs2|dota|valorant|spread|o/u', s): return 'sports'
    if re.search(r'election|president|senate|house|governor|mayor|trump|biden|vance|democrat|republican|parliament|prime-minister|vote|nominee|primary', s): return 'politics'
    if re.search(r'fed|interest-rate|cpi|inflation|gdp|recession|s-p-500|nasdaq|stock|earnings|tesla|nvidia|apple|ipo|tariff', s): return 'finance'
    if re.search(r'ukraine|russia|israel|iran|china|taiwan|war|ceasefire|strike|nato|gaza|hamas|military', s): return 'geopolitics'
    return 'other'


def load_meta(hf_dir):
    m = pd.read_parquet(f'{hf_dir}/markets_meta.parquet')
    m['win1'] = m['outcome_prices'].map({"['1', '0']": 1, "['0', '1']": 0})
    m = m[m['win1'].notna() & (m['closed'] == 1)].copy()
    m['cat'] = [category(a, b, c) for a, b, c in zip(m['event_slug'], m['question'], m['answer1'])]
    m['end_ts'] = m['end_date'].astype('int64') // 10**9 if str(m['end_date'].dtype).startswith('datetime64[ns') else m['end_date'].map(lambda x: x.timestamp())
    return m.rename(columns={'id': 'market_id'})
