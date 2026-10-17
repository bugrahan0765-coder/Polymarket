"""Station sanity check. Resolution stations could NOT be read from Gamma (Polymarket blocked on this machine), so the
GUESS airport/observatory coordinates are used and flagged unverified. Here we check them indirectly: per city, compare
the winning-bucket temperature with (a) Open-Meteo lead-0 daily max (historical-forecast API, ~analysis) and (b) the
strict >=48h forecast. A wrong station typically shows as a large lead-0 MAE / bias or a bias that jumps over time.
Also records each city's market unit history (e.g. London switched F -> C).
Output: results/weather_forecast/stations.json
Run: python src/weather_forecast/stations_check.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wf_lib import load, truth, FC, L0
from fetch_forecasts import GUESS

d = load(); t = truth(d)
l0 = pd.read_csv(L0, parse_dates=['date']).rename(columns={'date': 'target_date'})
fc = pd.read_csv(FC, parse_dates=['date']).rename(columns={'date': 'target_date'})
fc = fc[fc['model'] == 'best_match'][['city', 'target_date', 'tmax_strict', 'hour_max']]
x = t.merge(l0, on=['city', 'target_date'], how='left').merge(fc, on=['city', 'target_date'], how='left')
x = x[x['kind'] == 'mid']  # exact (non-censored) outcomes only
x['e0'] = x['y'] - x['tmax_lead0']; x['e2'] = x['y'] - x['tmax_strict']
x['half'] = np.where(x['target_date'] < pd.Timestamp('2026-06-01'), 'early', 'late')
out = {}
for city, g in x.groupby('city'):
    units = d[d['city'] == city].groupby('unit')['target_date'].agg(['min', 'max'])
    hb = g.groupby('half')['e0'].mean()
    out[city] = dict(lat=GUESS[city][0], lon=GUESS[city][1], verified=False, source='GUESS in src/a4_fetch_forecasts.py',
                     unit_history={u: [str(r['min'].date()), str(r['max'].date())] for u, r in units.iterrows()},
                     n=int(len(g)), lead0_bias_C=round(float(g['e0'].mean()), 2), lead0_mae_C=round(float(g['e0'].abs().mean()), 2),
                     lead0_bias_early_late=[round(float(hb.get('early', np.nan)), 2), round(float(hb.get('late', np.nan)), 2)],
                     fc48_bias_C=round(float(g['e2'].mean()), 2), fc48_mae_C=round(float(g['e2'].abs().mean()), 2),
                     share_max_hour_18_23=round(float((g['hour_max'] >= 18).mean()), 3))
json.dump(out, open('results/weather_forecast/stations.json', 'w'), indent=1)
df = pd.DataFrame(out).T[['n', 'unit_history', 'lead0_bias_C', 'lead0_mae_C', 'lead0_bias_early_late', 'fc48_bias_C', 'fc48_mae_C', 'share_max_hour_18_23']]
print(df.sort_values('lead0_mae_C').to_string())
