"""Step 4: match Polymarket game markets to odds, measure edge, walk-forward backtest (UNTESTED).

Pipeline
 1. Build per-market pre-game snapshots at T-24h, T-6h, T-1h, T-5m: last price <= t (prices-history),
    and realistic entry = VWAP of taker BUY trades on that token in [t, t+30min] (data-api trades).
 2. Match: soccer by (league, date of kickoff UTC +/-1 day, normalized home/away names); Polymarket soccer
    uses 3 binary markets per game ("Will <home> win?", "Will <away> win?", "Draw?") -> map to H/D/A.
    US sports: 2-outcome moneyline market (outcomes = team names) -> map to home/away.
 3. Fair prob = power-method de-vig of Pinnacle closing (PSC*) [benchmark]; signal uses opening/pre-game
    odds available at decision time (PS* for soccer, ESPN open for US).
 4. Signal: edge = fair_open - entry_price. Buy when edge > thr. thr chosen on TRAIN (first 60% by date),
    tested on TEST (last 40%). Report CLV (fair_close - entry) and realized PnL vs outcomes.
 5. Calibration: bucket Polymarket price at T-1h vs realized win frequency, per sport.

Run: python -I src/sports/s4_analyze.py   -> results/sports/*.json, *.csv
"""
import os, json, re, unicodedata, datetime as dt
import numpy as np, pandas as pd
from common_sports import CACHE, ROOT, devig_power, taker_fee

RES = os.path.join(ROOT, 'results', 'sports')
os.makedirs(RES, exist_ok=True)
HORIZONS = {'24h': 24 * 3600, '6h': 6 * 3600, '1h': 3600, '5m': 300}
# Fee rate per (league, date). FILL FROM docs.polymarket.com (fees page) before trusting results.
FEE_RATE = {}  # e.g. {'NBA': [(start_date, rate), ...]}


def fee_rate(league, t):
    r = 0.0
    for start, rate in FEE_RATE.get(league, []):
        if t >= start:
            r = rate
    return r


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'\b(fc|cf|afc|ac|sc|ssc|as|us|rc|cd|ud|sd|1\.|vfb|vfl|tsg|bsc|sv|fsv|rcd|ca|ogc|stade|olympique)\b', ' ', s)
    return re.sub(r'[^a-z]', '', s)


def ts(s):
    return int(dt.datetime.fromisoformat(str(s).replace('Z', '+00:00').replace(' ', 'T')).timestamp())


def snapshots(m):
    cid = m['conditionId']
    px = json.load(open(os.path.join(CACHE, 'px', cid + '.json')))
    tr = json.load(open(os.path.join(CACHE, 'trades', cid + '.json')))
    ko = ts(m['gameStartTime'])
    toks = json.loads(m['tokens']) if isinstance(m['tokens'], str) else m['tokens']
    outs = json.loads(m['outcomes']) if isinstance(m['outcomes'], str) else m['outcomes']
    fin = [float(x) for x in (json.loads(m['outcomePrices']) if isinstance(m['outcomePrices'], str) else m['outcomePrices'])]
    rows = []
    for tok, out, f in zip(toks, outs, fin):
        h = sorted(px.get(tok, []), key=lambda x: x['t'])
        buys = [x for x in tr if str(x.get('asset')) == str(tok) and x.get('side') == 'BUY']
        for hn, sec in HORIZONS.items():
            t0 = ko - sec
            last = [x['p'] for x in h if x['t'] <= t0]
            w = [x for x in buys if t0 <= int(x['timestamp']) < t0 + 1800 and int(x['timestamp']) < ko]
            sz = sum(float(x['size']) for x in w)
            vwap = sum(float(x['price']) * float(x['size']) for x in w) / sz if sz else np.nan
            near = sum(float(x['size']) * float(x['price']) for x in w if last and float(x['price']) <= last[-1] + 0.02)
            rows.append(dict(conditionId=cid, league=m['league'], question=m['question'], outcome=out, token=tok,
                             kickoff=ko, horizon=hn, last=last[-1] if last else np.nan, taker_vwap=vwap,
                             taker_usdc_within2c=near, won=1 if f > 0.5 else 0))
    return rows


def main():
    inv = [json.loads(l) for l in open(os.path.join(CACHE, 'inventory.jsonl'), encoding='utf8')]
    snaps = []
    for m in inv:
        try:
            snaps += snapshots(m)
        except Exception:
            pass
    S = pd.DataFrame(snaps)
    S.to_csv(os.path.join(RES, 'snapshots.csv.gz'), index=False)

    # --- calibration (Polymarket alone)
    c = S[(S.horizon == '1h') & S['last'].between(0.01, 0.99)].copy()
    c['bucket'] = (c['last'] * 10).astype(int) / 10
    cal = c.groupby(['league', 'bucket']).agg(n=('won', 'size'), price=('last', 'mean'), win=('won', 'mean')).reset_index()
    cal.to_csv(os.path.join(RES, 'calibration_1h.csv'), index=False)

    # --- odds matching (soccer)
    fd = pd.read_csv(os.path.join(CACHE, 'odds', 'football_data_all.csv'))
    fd['date'] = pd.to_datetime(fd['Date'], dayfirst=True).dt.date
    fd['h'] = fd['HomeTeam'].map(norm)
    fd['a'] = fd['AwayTeam'].map(norm)
    close_cols = ['PSCH', 'PSCD', 'PSCA'] if fd.get('PSCH') is not None and fd['PSCH'].notna().mean() > 0.5 else ['AvgCH', 'AvgCD', 'AvgCA']
    open_cols = ['PSH', 'PSD', 'PSA'] if fd.get('PSH') is not None and fd['PSH'].notna().mean() > 0.5 else ['AvgH', 'AvgD', 'AvgA']
    for pre, cols in (('close', close_cols), ('open', open_cols)):
        P = fd[cols].apply(lambda r: pd.Series(devig_power(list(r)) if r.notna().all() else [np.nan] * 3), axis=1)
        fd[[f'{pre}_H', f'{pre}_D', f'{pre}_A']] = P.values

    # map each Polymarket soccer market to a (fd row, side)
    soc = S[S.league.isin(['EPL', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1'])].copy()
    soc = soc[soc.outcome.str.lower() == 'yes']  # binary per-outcome markets; YES token = the named result
    soc['date'] = pd.to_datetime(soc.kickoff, unit='s', utc=True).dt.date
    out = []
    for (lg, d), g in soc.groupby(['league', 'date']):
        cand = fd[(fd.league == lg) & (fd.date == d)]
        for _, r in g.iterrows():
            q = norm(r.question)
            side = None
            if 'draw' in r.question.lower():
                hits = cand[[x in q and y in q for x, y in zip(cand.h, cand.a)]] if len(cand) else cand
                side = 'D'
            else:
                hh = cand[[x in q for x in cand.h]]
                aa = cand[[x in q for x in cand.a]]
                hits, side = (hh, 'H') if len(hh) == 1 else (aa, 'A') if len(aa) == 1 else (cand.iloc[0:0], None)
            if side and len(hits) == 1:
                fr = hits.iloc[0]
                out.append({**r.to_dict(), 'side': side, 'fair_open': fr[f'open_{side}'], 'fair_close': fr[f'close_{side}']})
    M = pd.DataFrame(out)
    M.to_csv(os.path.join(RES, 'matched_soccer.csv.gz'), index=False)
    # TODO: US sports matching against espn_us_odds.csv (2-way moneyline, American odds -> decimal).

    # --- walk-forward threshold test
    rep = {'odds_columns': {'open': open_cols, 'close': close_cols}, 'n_matched': int(M.conditionId.nunique()) if len(M) else 0}
    for hn in HORIZONS:
        H = M[(M.horizon == hn)].dropna(subset=['fair_open', 'fair_close']).sort_values('kickoff')
        if H.empty:
            continue
        for price_col, adj in (('last', 0.01), ('taker_vwap', 0.0)):
            X = H.dropna(subset=[price_col]).copy()
            X['entry'] = X[price_col] + adj
            X['fee'] = [taker_fee(p, 1, fee_rate(l, k)) for p, l, k in zip(X.entry, X.league, X.kickoff)]
            X['edge'] = X.fair_open - X.entry
            X['pnl'] = X.won - X.entry - X.fee  # per share
            X['clv'] = X.fair_close - X.entry - X.fee
            cut = X.kickoff.quantile(0.6)
            tr, te = X[X.kickoff <= cut], X[X.kickoff > cut]
            best = max(np.arange(0.0, 0.10, 0.01), key=lambda t: tr[tr.edge > t].pnl.sum() / max(tr[tr.edge > t].entry.sum(), 1e-9) if (tr.edge > t).sum() >= 30 else -9)
            sel = te[te.edge > best]
            roi = sel.pnl.sum() / sel.entry.sum() if len(sel) else np.nan
            # bootstrap clustered by kickoff date
            sel = sel.assign(day=pd.to_datetime(sel.kickoff, unit='s').dt.date)
            days = sel.day.unique()
            bs = []
            rng = np.random.default_rng(0)
            for _ in range(2000):
                pick = rng.choice(days, len(days)) if len(days) else []
                b = pd.concat([sel[sel.day == d] for d in pick]) if len(pick) else sel
                bs.append(b.pnl.sum() / b.entry.sum() if b.entry.sum() else np.nan)
            mon = sel.assign(m=pd.to_datetime(sel.kickoff, unit='s').dt.to_period('M')).groupby('m').pnl.sum()
            eq = sel.sort_values('kickoff').pnl.cumsum()
            rep[f'{hn}_{price_col}'] = dict(thr=float(best), n_test=int(len(sel)), roi=float(roi),
                                           ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                                           mean_clv=float(sel.clv.mean()) if len(sel) else None,
                                           worst_month=float(mon.min()) if len(mon) else None,
                                           max_dd=float((eq.cummax() - eq).max()) if len(eq) else None,
                                           median_capacity_usdc=float(sel.taker_usdc_within2c.median()) if len(sel) else None)
            # descriptive: mean (price - fair_close) by horizon
            rep[f'{hn}_{price_col}_gap_vs_close'] = float((X.entry - X.fair_close).mean())
    json.dump(rep, open(os.path.join(RES, 'walkforward.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
