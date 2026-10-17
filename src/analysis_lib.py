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
