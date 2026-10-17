"""Favourite edge x forecast agreement, as a trade rule fixed a priori (no parameters fitted on the test period).
Rule F0: buy YES on every favourite (highest fresh price in the event) with 0.30 <= p < 0.90.
Rule F1: F0 but only if the forecast model's most likely bucket is the favourite ("agree").
Rule F2: F0 but only if P_model >= p (model at least as bullish as the price).
Entry cost p+slip per share, Polymarket weather taker fee 0.05*c*(1-c) per share, $5 per trade.
Reported for pre-split (< 2026-05-01, the period the model hyper-parameters were selected on) and OOS (>= 2026-05-01).
Output: results/weather_forecast/fav_agree.json
Run: python src/weather_forecast/fav_agree.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wf_lib import fee, stats

SPLIT = pd.Timestamp('2026-05-01')
b = pd.read_csv('data/weather_forecast/model_probs.csv.gz', parse_dates=['target_date'])
b = b[b['P_model'].notna() & b['yes'].notna()]
f = b[b['p'].notna()]
f = f.loc[f.groupby('event_id')['p'].idxmax()].copy()
top = b.loc[b.groupby('event_id')['P_model'].idxmax(), ['event_id', 'bucket']].rename(columns={'bucket': 'top'})
f = f.merge(top, on='event_id')
f['agree'] = f['bucket'] == f['top']
f = f[(f['p'] >= .3) & (f['p'] < .9)]
R = {}
for slip in [.01, .02, .03]:
    c = (f['p'] + slip).round(4); sh = 5 / c
    t = f.assign(cost=5 + sh * fee(c), c=c)
    t['pnl'] = np.where(t['yes'] == 1, sh, 0) - t['cost']
    for rule, msk in [('F0_all_fav', np.ones(len(t), bool)), ('F1_agree', t['agree'].values), ('F2_model_ge_price', (t['P_model'] >= t['p']).values),
                      ('F1c_disagree', ~t['agree'].values)]:
        for per, pm in [('pre_split', t['target_date'] < SPLIT), ('oos', t['target_date'] >= SPLIT)]:
            x = t[msk & pm.values]
            r = stats(x, n_boot=1000)
            if per == 'oos' and len(x):
                r['by_unit'] = {u: stats(g, n_boot=500) for u, g in x.groupby('unit')}
                r['by_half'] = {h: stats(g, n_boot=500) for h, g in x.groupby(np.where(x['target_date'] < pd.Timestamp('2026-08-01'), 'May-Jul', 'Aug-Oct'))}
            R[f'{rule}|slip{int(slip * 100)}c|{per}'] = r
            print(f'{rule:20s} slip{int(slip*100)}c {per:9s}', {k: r.get(k) for k in ['n', 'win_rate', 'roi', 'roi_ci95', 'pnl', 'worst_month', 'max_dd', 'capital_needed', 'roi_on_capital']})
json.dump(R, open('results/weather_forecast/fav_agree.json', 'w'), indent=1, default=str)
