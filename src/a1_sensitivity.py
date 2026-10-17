"""Step 1: slippage/fee sensitivity and date-clustered bootstrap of the v1 strategy.
Run from repo root: python src/a1_sensitivity.py
"""
import json, numpy as np, pandas as pd
from analysis_lib import load_markets, favorites, sim, roi

m = load_markets(); f = favorites(m)
out = {}

# slippage x fee grid (compounding, cash-limited like v1)
grid = []
for slip in [0., .01, .02, .03, .04, .05]:
    for fee in [0., .02, .05, .10]:
        t, end = sim(f, slip=slip, fee=fee)
        grid.append(dict(slip=slip, fee=fee, n=len(t), roi=round(roi(t), 4), end=round(end, 2)))
g = pd.DataFrame(grid)
print('ROI by slippage (rows) x fee rate (cols):')
print(g.pivot(index='slip', columns='fee', values='roi').to_string())
print('\nEnd balance:')
print(g.pivot(index='slip', columns='fee', values='end').to_string())
out['grid'] = grid

# bootstrap: iid trades vs clustered by target date vs clustered by ISO week
t, _ = sim(f)
rng = np.random.default_rng(0)
B = 5000


def boot(key):
    grp = t.groupby(key).agg(pnl=('pnl', 'sum'), cost=('cost', 'sum'))
    pnl, cost = grp['pnl'].to_numpy(), grp['cost'].to_numpy()
    k = len(grp); r = np.empty(B)
    for b in range(B):
        i = rng.integers(0, k, k); r[b] = pnl[i].sum() / cost[i].sum()
    return dict(clusters=k, lo=round(float(np.percentile(r, 2.5)), 4), hi=round(float(np.percentile(r, 97.5)), 4),
                p_loss=round(float((r <= 0).mean()), 4))


t['trade'] = np.arange(len(t))
t['week'] = pd.to_datetime(t['target_date']).dt.strftime('%G-%V')
bs = {k: boot(k) for k in ['trade', 'target_date', 'week', 'month']}
print('\nBootstrap of ROI (point %.4f):' % roi(t))
for k, v in bs.items(): print(f'  cluster={k:12s} {v}')
out['bootstrap'] = bs

# break-even slippage at fee 0 (fixed stake, so no compounding effects)
for s in np.arange(0, .08, .005):
    tt, _ = sim(f, slip=s, cash_limited=False)
    if roi(tt) <= 0: print(f'\nBreak-even slippage (fixed stake, fee 0): ~{s:.3f} $/share'); out['breakeven_slip'] = round(float(s), 3); break

json.dump(out, open('results/a1_sensitivity.json', 'w'), indent=1, default=str)
