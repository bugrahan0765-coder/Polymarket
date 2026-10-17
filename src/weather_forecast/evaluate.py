"""Evaluate the forecast model against market prices, trade rules (walk-forward), and favourite-edge decomposition.

Inputs : data/weather_forecast/model_probs.csv.gz (from model.py)
Outputs: results/weather_forecast/evaluation.json and printed tables.
Run    : python src/weather_forecast/evaluate.py

* Scores: per-event log-loss of the winning bucket and multi-class Brier, for model, market (prices normalised to sum 1
  within the event; only events where every bucket has a fresh price), and a blend q ~ Pm^a * Pmkt^b (logarithmic
  pooling = conditional-logit combination) whose (a, b) are refitted each month on all EARLIER months only.
* Trading: per event, the single best candidate among YES (cost p+slip) and NO (cost 1-p+slip) on every bucket, with
  expected value per share EV = q - cost - fee(cost) (fee = 0.05*c*(1-c), Polymarket weather taker fee). Trade if EV > t.
  t (and nothing else) chosen each month on all earlier months (max PnL, >= 30 trades). $5 per trade + fee.
  OOS = target dates >= 2026-05-01 (model hyper-parameters were chosen before that date).
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
from wf_lib import fee, stats

SPLIT = pd.Timestamp('2026-05-01')
OUT = 'results/weather_forecast'
STAKE = 5.
TGRID = np.round(np.arange(0.0, 0.305, 0.02), 3)
EPS = 1e-4


def load():
    b = pd.read_csv('data/weather_forecast/model_probs.csv.gz', parse_dates=['target_date'])
    b = b[b['P_model'].notna() & b['yes'].notna()].copy()
    full = b.groupby('event_id')['p'].apply(lambda x: x.notna().all())
    b['all_priced'] = b['event_id'].map(full)
    b['P_mkt'] = b['p'] / b.groupby('event_id')['p'].transform('sum')
    b['month'] = b['target_date'].dt.to_period('M')
    return b


def pool(b, a1, a2):
    lm = np.log(np.clip(b['P_model'].values, EPS, 1)); lk = np.log(np.clip(b['P_mkt'].values, EPS, 1))
    z = a1 * lm + a2 * lk
    ez = np.exp(z - pd.Series(z, index=b.index).groupby(b['event_id']).transform('max').values)
    return ez / pd.Series(ez, index=b.index).groupby(b['event_id']).transform('sum').values


def fit_pool(tr):
    def nll(a):
        q = pool(tr, a[0], a[1]); return -np.log(np.clip(q[tr['yes'].values == 1], 1e-9, 1)).mean()
    r = minimize(nll, [0.5, 0.5], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-6))
    return r.x


def add_blend(b):
    """Walk-forward monthly refit of the pooling weights on all earlier months (events where all buckets priced)."""
    b = b.copy(); b['P_blend'] = np.nan; coefs = {}
    pr = b[b['all_priced']]
    for mo in sorted(pr['month'].unique()):
        tr = pr[pr['month'] < mo]
        if tr['event_id'].nunique() < 300: continue
        a = fit_pool(tr); coefs[str(mo)] = [round(float(a[0]), 3), round(float(a[1]), 3)]
        te = pr['month'] == mo
        b.loc[pr.index[te.values], 'P_blend'] = pool(pr[te], a[0], a[1])
    return b, coefs


def scores(b, cols):
    rows = {}
    w = b[b['yes'] == 1]
    for c in cols:
        ll = -np.log(np.clip(w[c].values, EPS, 1))
        br = ((b[c] - b['yes']) ** 2).groupby(b['event_id']).sum()
        rows[c] = dict(logloss=round(float(ll.mean()), 4), brier=round(float(br.mean()), 4))
    rows['n_events'] = int(b['event_id'].nunique())
    return rows


def candidates(b, qcol, slip):
    """All YES/NO candidates with EV per share."""
    x = b[b['p'].notna() & b[qcol].notna()]
    q = x[qcol].values; p = x['p'].values
    cy = np.round(p + slip, 4); cn = np.round(1 - p + slip, 4)
    Y = pd.DataFrame(dict(event_id=x['event_id'].values, target_date=x['target_date'].values, month=x['month'].values,
                          unit=x['unit'].values, city=x['city'].values, bucket=x['bucket'].values, side='YES', p=p, q=q,
                          c=cy, win=x['yes'].values == 1))
    N = Y.copy(); N['side'] = 'NO'; N['c'] = cn; N['q'] = 1 - q; N['win'] = x['yes'].values == 0
    C = pd.concat([Y, N], ignore_index=True)
    C = C[(C['c'] > 0) & (C['c'] < 0.99)]
    C['ev'] = C['q'] - C['c'] - fee(C['c'])
    return C


def to_trades(C):
    sh = STAKE / C['c']
    cost = STAKE + sh * fee(C['c'])
    return C.assign(cost=cost, pnl=np.where(C['win'], sh, 0.) - cost)


def best_per_event(C, sides):
    C = C[C['side'].isin(sides)]
    return C.loc[C.groupby('event_id')['ev'].idxmax()]


def walk_forward(C1, start=SPLIT, min_tr=30):
    """C1: one best candidate per event. Each month >= start: t chosen on all earlier months."""
    months = sorted(C1['month'].unique()); out = []; chosen = {}
    for mo in months:
        if mo.to_timestamp() < start: continue
        tr = C1[C1['month'] < mo]; best = None
        for t in TGRID:
            x = tr[tr['ev'] > t]
            if len(x) < min_tr: continue
            pnl = to_trades(x)['pnl'].sum()
            if best is None or pnl > best[0]: best = (pnl, t)
        if best is None: continue
        chosen[str(mo)] = float(best[1])
        te = C1[(C1['month'] == mo) & (C1['ev'] > best[1])]
        out.append(to_trades(te))
    return (pd.concat(out) if out else pd.DataFrame()), chosen


def favourite_analysis(b):
    f = b[b['p'].notna()]
    f = f.loc[f.groupby('event_id')['p'].idxmax()].copy()
    am = b.loc[b.groupby('event_id')['P_model'].idxmax(), ['event_id', 'bucket']].rename(columns={'bucket': 'model_top'})
    f = f.merge(am, on='event_id')
    f['agree'] = f['bucket'] == f['model_top']
    f['win'] = f['yes'] == 1
    res = {}
    g = f.groupby('agree').agg(n=('win', 'size'), price=('p', 'mean'), P_model=('P_model', 'mean'), won=('win', 'mean'))
    g['edge_vs_price'] = g['won'] - g['price']; res['by_agree'] = g.round(4).to_dict('index')
    res['P_agree_given_win'] = round(float(f.loc[f['win'], 'agree'].mean()), 4)
    res['P_agree_given_loss'] = round(float(f.loc[~f['win'], 'agree'].mean()), 4)
    f['dif_bin'] = pd.cut(f['P_model'] - f['p'], [-1, -.2, -.1, 0, .1, .2, 1])
    g = f.groupby('dif_bin', observed=True).agg(n=('win', 'size'), price=('p', 'mean'), P_model=('P_model', 'mean'), won=('win', 'mean'))
    g['edge_vs_price'] = g['won'] - g['price']; res['by_model_minus_price'] = {str(k): v for k, v in g.round(4).to_dict('index').items()}
    # logistic regression win ~ logit(price) + logit(P_model)
    import statsmodels.api as sm
    lg = lambda v: np.log(np.clip(v, .01, .99) / (1 - np.clip(v, .01, .99)))
    X = sm.add_constant(np.column_stack([lg(f['p']), lg(f['P_model'])]))
    m1 = sm.Logit(f['win'].astype(float), sm.add_constant(lg(f['p']))).fit(disp=0)
    m2 = sm.Logit(f['win'].astype(float), X).fit(disp=0)
    res['logit_price_only'] = dict(params=[round(float(v), 3) for v in m1.params], se=[round(float(v), 3) for v in m1.bse], llf=round(float(m1.llf), 1))
    res['logit_price_plus_model'] = dict(params=[round(float(v), 3) for v in m2.params], se=[round(float(v), 3) for v in m2.bse], llf=round(float(m2.llf), 1))
    # favourite edge in price bins, overall vs model-agree subsets (does the edge survive conditioning on the model?)
    f['pbin'] = pd.cut(f['p'], [.3, .4, .5, .6, .7, .8, .95])
    g = f.groupby(['pbin', 'agree'], observed=True).agg(n=('win', 'size'), price=('p', 'mean'), won=('win', 'mean'))
    g['edge'] = g['won'] - g['price']
    res['by_price_and_agree'] = {f'{k[0]}|agree={k[1]}': v for k, v in g.round(4).to_dict('index').items()}
    g = f.groupby('pbin', observed=True).agg(n=('win', 'size'), price=('p', 'mean'), won=('win', 'mean'), P_model=('P_model', 'mean'))
    g['edge'] = g['won'] - g['price']; g['model_minus_price'] = g['P_model'] - g['price']
    res['fav_by_price'] = {str(k): v for k, v in g.round(4).to_dict('index').items()}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    b = load()
    b, coefs = add_blend(b)
    R = dict(blend_coefs_by_month=coefs)
    print('blend coefs (model, market) per test month:', coefs)
    sc = {}
    pr = b[b['all_priced'] & b['P_blend'].notna()]
    for name, x in [('pre_split', pr[pr['target_date'] < SPLIT]), ('oos', pr[pr['target_date'] >= SPLIT]),
                    ('oos_C', pr[(pr['target_date'] >= SPLIT) & (pr['unit'] == 'C')]),
                    ('oos_F', pr[(pr['target_date'] >= SPLIT) & (pr['unit'] == 'F')])]:
        sc[name] = scores(x, ['P_model', 'P_mkt', 'P_blend'])
    # per month
    for mo, x in pr.groupby('month'):
        sc[f'month_{mo}'] = scores(x, ['P_model', 'P_mkt', 'P_blend'])
    R['scores'] = sc
    print(pd.DataFrame({k: {f'{c}_{m}': v[c][m] for c in ['P_model', 'P_mkt', 'P_blend'] for m in ['logloss', 'brier']} | {'n': v['n_events']}
                        for k, v in sc.items()}).T.to_string())

    trade = {}; monthly = {}
    for qcol in ['P_model', 'P_blend']:
        for slip in [.01, .02, .03]:
            C = candidates(b, qcol, slip)
            for sides in [('YES', 'NO'), ('YES',), ('NO',)]:
                C1 = best_per_event(C, sides)
                T, chosen = walk_forward(C1)
                key = f'{qcol}|slip{int(slip * 100)}c|{"+".join(sides)}'
                r = stats(T) if len(T) else dict(n=0)
                r['t_chosen'] = chosen
                if len(T):
                    r['by_unit'] = {u: stats(x, n_boot=500) for u, x in T.groupby('unit')}
                    r['by_side'] = {u: stats(x, n_boot=500) for u, x in T.groupby('side')}
                    r['by_half'] = {h: stats(x, n_boot=500) for h, x in T.groupby(np.where(T['target_date'] < pd.Timestamp('2026-08-01'), 'May-Jul', 'Aug-Oct'))}
                    monthly[key] = T.groupby(T['month'].astype(str))['pnl'].sum().round(2).to_dict()
                trade[key] = r
                print(key, {k: r.get(k) for k in ['n', 'roi', 'roi_ci95', 'pnl', 'worst_month', 'max_dd', 'roi_on_capital']}, 't=', chosen)
    R['trading_oos'] = trade; R['trading_monthly_pnl'] = monthly

    # in-sample reference: same rule with t fixed, on pre-split (shows what training looked like)
    ins = {}
    for qcol in ['P_model', 'P_blend']:
        for slip in [.01, .02, .03]:
            C1 = best_per_event(candidates(b, qcol, slip), ('YES', 'NO'))
            for t in [0.0, .05, .1, .2]:
                x = C1[(C1['target_date'] < SPLIT) & (C1['ev'] > t)]
                ins[f'{qcol}|slip{int(slip * 100)}c|t{t}'] = stats(to_trades(x), n_boot=300) if len(x) else dict(n=0)
                y = C1[(C1['target_date'] >= SPLIT) & (C1['ev'] > t)]
                ins[f'{qcol}|slip{int(slip * 100)}c|t{t}|OOS_fixed'] = stats(to_trades(y), n_boot=300) if len(y) else dict(n=0)
    R['fixed_threshold_reference'] = ins
    for k, v in ins.items():
        print(k, {kk: v.get(kk) for kk in ['n', 'roi', 'roi_ci95', 'pnl']})

    R['favourite'] = favourite_analysis(b)
    R['favourite_oos'] = favourite_analysis(b[b['target_date'] >= SPLIT])
    print(json.dumps(R['favourite'], indent=1, default=str))
    json.dump(R, open(os.path.join(OUT, 'evaluation.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
