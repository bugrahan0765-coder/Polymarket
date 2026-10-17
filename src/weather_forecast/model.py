"""Walk-forward probabilistic forecast model for daily-max-temperature buckets.

For event (city, target date D) the model only uses
  * Open-Meteo forecasts for D made >= 48 h before (strict construction, see fetch_forecasts.py), and
  * realised outcomes (winning bucket) of the same city with target date in [D-W-1, D-2]  (all resolved before the
    decision at D-1 00:00 local; D-1 is excluded because it resolves after the decision).
Per model m: bias b_m = mean(y - f_m) over the window; bias-corrected forecasts are blended (inverse-MSE weights or
equal weights); residual scale s = RMSE of the blended forecast over the window. P(bucket) = integral over the bucket's
continuous interval [lo, hi) (rounding to whole degrees, F-ranges a-b -> [a-.5, b+.5)) of a Student-t(nu) with
location mu and scale k*s*sqrt((nu-2)/nu). Hyper-parameters (W, blend, nu, k) are chosen ONLY on events with
target date < SPLIT (2026-05-01) by mean log-likelihood of the winning bucket; everything later is out-of-sample.
Observed y = winning-bucket midpoint (deg C); for tail winners 'X or higher'/'X or below' y = X +/- 1 unit (censored, approximated).
Output: data/weather_forecast/model_probs.csv.gz (one row per bucket) + results/weather_forecast/model_selection.json
Run: python src/weather_forecast/model.py
"""
import json, os, sys, itertools
import numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, os.path.dirname(__file__))
from wf_lib import load, truth, FC, f2c

SPLIT = pd.Timestamp('2026-05-01')
MODELS = ['best_match', 'ecmwf_ifs025', 'gfs_seamless', 'icon_seamless', 'ukmo_seamless', 'gem_seamless', 'jma_seamless']
MIN_N = 15
OUT = 'results/weather_forecast'


def build_obs():
    d = load(); t = truth(d)
    unit_step = np.where(t['unit'] == 'F', 5 / 9, 1.0)
    t['y'] = t['y'] + np.where(t['kind'] == 'high', unit_step, 0) - np.where(t['kind'] == 'low', unit_step, 0)
    fc = pd.read_csv(FC); fc['date'] = pd.to_datetime(fc['date'])
    wide = fc.pivot_table(index=['city', 'date'], columns='model', values='tmax_strict')
    wide_d2 = fc.pivot_table(index=['city', 'date'], columns='model', values='tmax_d2')
    return d, t, wide, wide_d2


def location_scale(t, wide, W, scheme, models=MODELS):
    """Per event mu, s (deg C) using only prior resolved events of the same city."""
    out = []
    for city, g in t.groupby('city'):
        g = g.sort_values('target_date')
        try: F = wide.loc[city].reindex(index=g["target_date"], columns=models).values
        except KeyError: continue
        dates = g['target_date'].values; y = g['y'].values
        for i in range(len(g)):
            D = dates[i]
            lo, hi = D - np.timedelta64(W + 1, 'D'), D - np.timedelta64(2, 'D')
            msk = (dates >= lo) & (dates <= hi)
            f_now = F[i]
            ok_m = ~np.isnan(f_now)
            if msk.sum() < MIN_N or not ok_m.any():
                out.append((g['event_id'].iat[i], np.nan, np.nan, int(msk.sum()))); continue
            R = y[msk][:, None] - F[msk]  # residuals per model
            cnt = (~np.isnan(R)).sum(0)
            use = ok_m & (cnt >= MIN_N)
            if not use.any():
                out.append((g['event_id'].iat[i], np.nan, np.nan, int(msk.sum()))); continue
            b = np.nanmean(R[:, use], 0)
            mse = np.nanmean((R[:, use] - b) ** 2, 0)
            w = (1 / mse) if scheme == 'invmse' else np.ones(use.sum())
            w = w / w.sum()
            mu = float(np.sum(w * (f_now[use] + b)))
            # in-window error of the same blend (rows where all used models are present)
            Fc = F[msk][:, use] + b
            rows = ~np.isnan(Fc).any(1)
            e = y[msk][rows] - Fc[rows] @ w
            s = float(np.sqrt(np.mean(e ** 2))) if rows.sum() >= MIN_N else float(np.sqrt(np.sum(w * mse)))
            out.append((g['event_id'].iat[i], mu, max(s, .3), int(msk.sum())))
    return pd.DataFrame(out, columns=['event_id', 'mu', 's', 'n_hist'])


def bucket_probs(b, mu, s, nu, k):
    sc = k * s * (np.sqrt((nu - 2) / nu) if np.isfinite(nu) else 1.)
    cdf = (lambda z: st.t.cdf(z, nu)) if np.isfinite(nu) else st.norm.cdf
    P = cdf((b['hi_c'].values - mu) / sc) - cdf((b['lo_c'].values - mu) / sc)
    return np.clip(P, 1e-6, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    d, t, wide, wide_d2 = build_obs()
    # quick diagnostics: raw 48h forecast errors per model (deg C), all events
    diag = {}
    for mdl in MODELS:
        f = wide[mdl].reindex(pd.MultiIndex.from_frame(t[['city', 'target_date']])).values
        e = t['y'].values - f
        diag[mdl] = dict(n=int(np.isfinite(e).sum()), bias=round(float(np.nanmean(e)), 3), mae=round(float(np.nanmean(np.abs(e))), 3))
    f2 = wide_d2['best_match'].reindex(pd.MultiIndex.from_frame(t[['city', 'target_date']])).values
    fs = wide['best_match'].reindex(pd.MultiIndex.from_frame(t[['city', 'target_date']])).values
    diag['strict_vs_d2_best_match_mean_abs_diff_C'] = round(float(np.nanmean(np.abs(f2 - fs))), 3)
    print(json.dumps(diag, indent=1))

    cfgs = {}
    for W, scheme in itertools.product([30, 60, 90], ['invmse', 'equal']):
        ls = location_scale(t, wide, W, scheme); cfgs[(W, scheme)] = ls
        print('computed', W, scheme, ls['mu'].notna().sum(), flush=True)
    for single in ['best_match', 'ecmwf_ifs025']:
        ls = location_scale(t, wide, 60, 'equal', models=[single]); cfgs[(60, single)] = ls
        print('computed', 60, single, ls['mu'].notna().sum(), flush=True)

    b = d[['event_id', 'market_id', 'lo_c', 'hi_c', 'yes']].copy()
    win = b[b['yes'] == 1].drop_duplicates('event_id')
    tt = t.set_index('event_id')
    sel = []
    for (W, scheme), ls in cfgs.items():
        x = win.merge(ls, on='event_id').dropna(subset=['mu'])
        x['target_date'] = tt.loc[x['event_id'], 'target_date'].values
        for nu, k in itertools.product([3, 4, 6, 10, np.inf], [.8, .9, 1., 1.1, 1.2, 1.35, 1.5]):
            pw = bucket_probs(x, x['mu'].values, x['s'].values, nu, k)
            ll = np.log(pw)
            tr = (x['target_date'] < SPLIT).values
            sel.append(dict(W=W, scheme=scheme, nu=nu, k=k, n_train=int(tr.sum()), ll_train=float(ll[tr].mean()),
                            n_test=int((~tr).sum()), ll_test=float(ll[~tr].mean())))
    S = pd.DataFrame(sel)
    # compare on the common set of events (W=90 needs more history) -> restrict to events with all cfgs present
    common = set.intersection(*[set(v.dropna(subset=['mu'])['event_id']) for v in cfgs.values()])
    sel2 = []
    for (W, scheme), ls in cfgs.items():
        x = win.merge(ls[ls['event_id'].isin(common)], on='event_id').dropna(subset=['mu'])
        x['target_date'] = tt.loc[x['event_id'], 'target_date'].values
        tr = (x['target_date'] < SPLIT).values
        for nu, k in itertools.product([3, 4, 6, 10, np.inf], [.8, .9, 1., 1.1, 1.2, 1.35, 1.5]):
            ll = np.log(bucket_probs(x, x['mu'].values, x['s'].values, nu, k))
            sel2.append(dict(W=W, scheme=scheme, nu=nu, k=k, n_train=int(tr.sum()), ll_train=float(ll[tr].mean()), ll_test=float(ll[~tr].mean())))
    S2 = pd.DataFrame(sel2).sort_values('ll_train', ascending=False)
    best = S2.iloc[0]
    print(S2.head(15).to_string())
    print('per-config best (train, common events):')
    print(S2.groupby(['W', 'scheme']).apply(lambda g: g.loc[g['ll_train'].idxmax()], include_groups=False)[['nu', 'k', 'll_train', 'll_test']].to_string())
    W, scheme, nu, k = int(best['W']), best['scheme'], float(best['nu']), float(best['k'])
    ls = cfgs[(W, scheme)]
    allb = d.merge(ls, on='event_id', how='left')
    allb['P_model'] = np.nan
    okm = allb['mu'].notna()
    allb.loc[okm, 'P_model'] = bucket_probs(allb[okm], allb.loc[okm, 'mu'].values, allb.loc[okm, 's'].values, nu, k)
    allb['P_model'] = allb['P_model'] / allb.groupby('event_id')['P_model'].transform('sum')
    cols = ['event_id', 'city', 'target_date', 'decision_ts', 'market_id', 'bucket', 'unit', 'lo_c', 'hi_c', 'kind', 'p', 'yes',
            'closed_time', 'mu', 's', 'n_hist', 'P_model']
    allb[cols].to_csv('data/weather_forecast/model_probs.csv.gz', index=False)
    json.dump(dict(chosen=dict(W=W, scheme=scheme, nu=nu, k=k), split=str(SPLIT.date()), raw_error_diag=diag,
                   grid_common_events=S2.head(40).to_dict('records')),
              open(os.path.join(OUT, 'model_selection.json'), 'w'), indent=1, default=str)
    print('chosen', W, scheme, nu, k, 'events with model', allb.loc[okm, 'event_id'].nunique())


if __name__ == '__main__':
    main()
