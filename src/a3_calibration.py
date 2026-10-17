"""Step 3: calibration of decision prices vs realised outcomes for ALL buckets.
Price bins x (favorite / non-favorite) x (first half / second half), YES and NO edge after slippage.
Run: python src/a3_calibration.py
"""
import json, numpy as np, pandas as pd
from analysis_lib import load_markets

m = load_markets()
m = m[m['p'].notna() & m['yes'].notna()].copy()
m['rank'] = m.groupby('event_id')['p'].rank(ascending=False, method='first')
m['fav'] = m['rank'] == 1
m['month'] = pd.to_datetime(m['target_date']).dt.to_period('M')
m['half'] = np.where(m['month'] < pd.Period('2026-04'), 'H1', 'H2')
m['unit'] = np.where(m['bucket'].str.contains('°C'), 'C', 'F')
BINS = [0, .01, .03, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0001]
m['bin'] = pd.cut(m['p'], BINS, right=False)


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); s = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def table(df, slip=.01):
    g = df.groupby('bin', observed=True).agg(n=('yes', 'size'), price=('p', 'mean'), won=('yes', 'mean'), k=('yes', 'sum'))
    ci = [wilson(k, n) for k, n in zip(g['k'], g['n'])]
    g['ci_lo'] = [c[0] for c in ci]; g['ci_hi'] = [c[1] for c in ci]
    g['edge'] = g['won'] - g['price']
    # ROI per $ of buying YES at p+slip, or NO at (1-p)+slip
    g['roi_yes'] = df.groupby('bin', observed=True).apply(lambda x: (x['yes'] / (x['p'] + slip)).mean() - 1, include_groups=False)
    g['roi_no'] = df.groupby('bin', observed=True).apply(lambda x: ((1 - x['yes']) / (1 - x['p'] + slip)).mean() - 1, include_groups=False)
    return g.drop(columns='k').round(4)


out = {}
pd.set_option('display.width', 200)
print(f'{len(m)} priced+resolved buckets, {m.event_id.nunique()} events')
s = m.groupby('event_id')['p'].sum()
print(f'Sum of bucket prices per event: median {s.median():.3f}, mean {s.mean():.3f} (1.0 = no overround)')
out['price_sum'] = dict(median=round(s.median(), 4), mean=round(s.mean(), 4))

for name, df in [('all', m), ('favorites', m[m.fav]), ('non_favorites', m[~m.fav]),
                 ('fav_H1', m[m.fav & (m.half == 'H1')]), ('fav_H2', m[m.fav & (m.half == 'H2')]),
                 ('nonfav_H1', m[~m.fav & (m.half == 'H1')]), ('nonfav_H2', m[~m.fav & (m.half == 'H2')]),
                 ('fav_C', m[m.fav & (m.unit == 'C')]), ('fav_F', m[m.fav & (m.unit == 'F')])]:
    t = table(df); print(f'\n=== {name} ===\n{t.to_string()}')
    out[name] = {str(k): v for k, v in t.to_dict('index').items()}

# favourite price level over time (explains why v1 trades fewer events later)
fv = m[m.fav].groupby('month').agg(events=('p', 'size'), fav_price_median=('p', 'median'),
                                   share_in_v1_band=('p', lambda x: ((x + .01 >= .6) & (x + .01 < .9)).mean()),
                                   share_C=('unit', lambda x: (x == 'C').mean()), fav_win=('yes', 'mean'))
print('\n=== favourite by month ===\n' + fv.round(3).to_string())
out['fav_by_month'] = {str(k): v for k, v in fv.round(4).to_dict('index').items()}

# per-city favourite edge (all prices >= .4), min 50 events
c = m[m.fav & (m.p >= .4)].groupby('city').agg(n=('yes', 'size'), price=('p', 'mean'), won=('yes', 'mean'))
c['edge'] = c['won'] - c['price']; c = c[c.n >= 50].sort_values('edge')
print('\n=== favourite (p>=0.40) edge by city ===\n' + c.round(3).to_string())
out['fav_city'] = c.round(4).to_dict('index')
json.dump(out, open('results/a3_calibration.json', 'w'), indent=1, default=str)
