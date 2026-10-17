"""Step 2: walk-forward test of the price-band rule.
Each month, choose (lo, hi) on all earlier months (expanding window), then trade only that month.
Fixed $5 stake per trade (no compounding) so months are comparable. Run: python src/a2_walkforward.py
"""
import json, itertools, numpy as np, pandas as pd
from analysis_lib import load_markets, favorites

CAP = 5.
m = load_markets(); f = favorites(m)
months = sorted(f['month'].unique())
LOS = np.round(np.arange(.40, .90, .05), 2)
BANDS = [(lo, hi) for lo, hi in itertools.product(LOS, np.round(np.arange(.45, 1.0, .05), 2)) if hi > lo]


def trades(df, lo, hi, slip):
    bp = (df['p'] + slip).round(4)
    t = df[(bp >= lo) & (bp < hi)].assign(bp=bp)
    t = t.assign(cost=CAP, pnl=np.where(t['win'], CAP / t['bp'], 0.) - CAP)
    return t


def pick(train, slip, crit):
    best = None
    for lo, hi in BANDS:
        t = trades(train, lo, hi, slip)
        if len(t) < 30: continue
        score = t['pnl'].sum() if crit == 'pnl' else t['pnl'].sum() / t['cost'].sum()
        if best is None or score > best[0]: best = (score, lo, hi)
    return best


out = {}
for slip in [.01, .02]:
    for crit in ['pnl', 'roi']:
        rows = []
        for i, mo in enumerate(months):
            if i < 3: continue  # need >= 3 months of history
            train, test = f[f['month'] < mo], f[f['month'] == mo]
            b = pick(train, slip, crit)
            if b is None: continue
            t = trades(test, b[1], b[2], slip)
            v1 = trades(test, .6, .9, slip)
            rows.append(dict(month=str(mo), lo=b[1], hi=b[2], n=len(t), pnl=round(t['pnl'].sum(), 2),
                             v1_n=len(v1), v1_pnl=round(v1['pnl'].sum(), 2)))
        r = pd.DataFrame(rows)
        tot_cost = CAP * r['n'].sum(); v1_cost = CAP * r['v1_n'].sum()
        summ = dict(oos_trades=int(r['n'].sum()), oos_pnl=round(r['pnl'].sum(), 2),
                    oos_roi=round(r['pnl'].sum() / tot_cost, 4) if tot_cost else None,
                    v1_trades=int(r['v1_n'].sum()), v1_pnl=round(r['v1_pnl'].sum(), 2),
                    v1_roi=round(r['v1_pnl'].sum() / v1_cost, 4), losing_months=int((r['pnl'] < 0).sum()), months=len(r))
        print(f'\n=== slip={slip} criterion={crit} ===')
        print(r.to_string(index=False)); print(summ)
        out[f'slip{slip}_{crit}'] = dict(months=rows, summary=summ)

# simple half split: choose on first 6 months, test on the rest
half = months[6]
for slip in [.01, .02]:
    b = pick(f[f['month'] < half], slip, 'pnl')
    t = trades(f[f['month'] >= half], b[1], b[2], slip)
    print(f'\nHalf split slip={slip}: band chosen on months<{half}: {b[1]}-{b[2]}; test n={len(t)} pnl={t["pnl"].sum():.2f} roi={t["pnl"].sum()/t["cost"].sum():.4f}')
    out[f'half_slip{slip}'] = dict(lo=b[1], hi=b[2], n=len(t), pnl=round(t['pnl'].sum(), 2), roi=round(t['pnl'].sum() / t['cost'].sum(), 4))

json.dump(out, open('results/a2_walkforward.json', 'w'), indent=1)
