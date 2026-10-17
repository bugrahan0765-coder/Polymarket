"""Helpers for the weather-forecast workstream: bucket parsing, market table, fees, evaluation stats."""
import re
import numpy as np, pandas as pd

DATA = 'data/decision_prices.csv.gz'
FC = 'data/weather_forecast/forecasts_daily.csv.gz'
L0 = 'data/weather_forecast/lead0_daily.csv.gz'
FEE_RATE = 0.05  # Polymarket taker fee, Weather category (docs.polymarket.com/trading/fees): fee = C * 0.05 * p * (1-p)

RX_RANGE = re.compile(r'^\s*(-?\d+)\s*[-–]\s*(-?\d+)\s*°\s*([CF])\s*$')
RX_ONE = re.compile(r'^\s*(-?\d+)\s*°\s*([CF])\s*(or below|or higher)?\s*$')


def parse_bucket(s):
    """-> (unit, lo, hi, mid) in the market unit; interval of the CONTINUOUS temperature that rounds into the bucket.
    'X°C' -> [X-.5, X+.5); 'a-b°F' -> [a-.5, b+.5); 'X or below' -> (-inf, X+.5); 'X or higher' -> [X-.5, inf)."""
    m = RX_RANGE.match(s)
    if m:
        a, b, u = int(m.group(1)), int(m.group(2)), m.group(3)
        return u, a - .5, b + .5, (a + b) / 2, 'mid'
    m = RX_ONE.match(s)
    if m:
        x, u, t = int(m.group(1)), m.group(2), m.group(3)
        if t == 'or below': return u, -np.inf, x + .5, float(x), 'low'
        if t == 'or higher': return u, x - .5, np.inf, float(x), 'high'
        return u, x - .5, x + .5, float(x), 'mid'
    raise ValueError(s)


def f2c(x): return (x - 32.) * 5. / 9.


def load(win=120):
    df = pd.read_csv(DATA, dtype={'outcome_yes': 'string', 'closed_time': 'string'})
    df['p'] = np.where(df['price_age_s'].fillna(1e18) <= win, df['price_last_le_dec'], np.nan)
    df['yes'] = df['outcome_yes'].map({'1': 1, '0': 0, '1.0': 1, '0.0': 0})
    pb = df['bucket'].map(parse_bucket)
    df['unit'] = [x[0] for x in pb]; df['lo'] = [x[1] for x in pb]; df['hi'] = [x[2] for x in pb]
    df['mid'] = [x[3] for x in pb]; df['kind'] = [x[4] for x in pb]
    # interval bounds in Celsius (for F markets convert; inf stays inf)
    isF = df['unit'] == 'F'
    df['lo_c'] = np.where(isF, f2c(df['lo']), df['lo']); df['hi_c'] = np.where(isF, f2c(df['hi']), df['hi'])
    df['mid_c'] = np.where(isF, f2c(df['mid']), df['mid'])
    df['target_date'] = pd.to_datetime(df['target_date'])
    return df


def truth(df):
    """One row per event: city, date, unit, observed max (winning bucket midpoint, deg C), kind of winning bucket."""
    w = df[df['yes'] == 1].drop_duplicates('event_id')
    return w[['event_id', 'city', 'target_date', 'unit', 'mid_c', 'kind', 'bucket']].rename(columns={'mid_c': 'y'}).reset_index(drop=True)


def fee(c):
    return FEE_RATE * c * (1 - c)


def stats(t, stake_col='cost', n_boot=2000, seed=0):
    """t: trades with columns target_date, cost, pnl. Returns dict with n, ROI, date-clustered bootstrap CI,
    worst month, max drawdown ($), capital needed (max cost open over 2 consecutive target dates) and ROI on capital."""
    if len(t) == 0: return dict(n=0)
    t = t.sort_values('target_date')
    roi = t['pnl'].sum() / t['cost'].sum()
    g = t.groupby('target_date').agg(pnl=('pnl', 'sum'), cost=('cost', 'sum'))
    rng = np.random.default_rng(seed); P, C = g['pnl'].values, g['cost'].values; n = len(g)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n); bs.append(P[i].sum() / C[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    cum = g['pnl'].cumsum(); dd = (cum.cummax().clip(lower=0) - cum).max()
    mo = t.groupby(t['target_date'].dt.to_period('M'))['pnl'].sum()
    cap = g['cost'].rolling(2, min_periods=1).sum().max()  # positions live ~ D-1 00:00 .. D+1 resolution
    return dict(n=int(len(t)), win_rate=round(float((t['pnl'] > 0).mean()), 4), staked=round(float(t['cost'].sum()), 2),
                pnl=round(float(t['pnl'].sum()), 2), roi=round(float(roi), 4), roi_ci95=[round(float(lo), 4), round(float(hi), 4)],
                p_roi_le0=round(float(np.mean(np.array(bs) <= 0)), 4),
                worst_month=f'{mo.idxmin()} {mo.min():.2f}', months_pos=f'{int((mo > 0).sum())}/{len(mo)}',
                max_dd=round(float(dd), 2), capital_needed=round(float(cap), 2), roi_on_capital=round(float(t['pnl'].sum() / cap), 4))
