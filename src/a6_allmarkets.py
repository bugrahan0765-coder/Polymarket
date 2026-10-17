"""Step 6: all-Polymarket calibration on REAL taker prices (hf_extract.py hourly output + markets_meta.parquet).
Snapshot each resolved binary market at horizons before its last trade: token1 ask = VWAP of taker buys of token1,
token1 bid = VWAP of taker sells (so token2 ask ~ 1 - bid) in the latest hour within 6h before the snapshot.
Reports ROI (after taker fees) of buying token1 / token2 at those asks by category x horizon x price bin,
split by period, plus the 'market family age' effect. Run: python src/a6_allmarkets.py <hf_data_dir>
"""
import sys, re, json, duckdb, numpy as np, pandas as pd

HF = sys.argv[1] if len(sys.argv) > 1 else '../hf_data'
HORIZONS = {'1h': 1, '6h': 6, '1d': 24, '3d': 72, '7d': 168, '30d': 720}
SPLIT = pd.Timestamp('2026-05-01', tz='UTC').timestamp()
# taker fee rate x p(1-p) per share (third-party summaries of Polymarket docs; applied to the whole period, conservative)
FEE = {'crypto': .07, 'sports': .03, 'weather': .05, 'politics': .04, 'finance': .04, 'geopolitics': 0., 'other': .05}

c = duckdb.connect()
c.execute(f"create view h as select * from read_parquet('{HF}/ext/hourly/*.parquet')")
c.execute(f"""create table meta as select id market_id, question, event_slug, answer1, neg_risk, volume, created_at, end_date,
              case when outcome_prices = '[''1'', ''0'']' then 1 when outcome_prices = '[''0'', ''1'']' then 0 end win1
              from read_parquet('{HF}/markets_meta.parquet') where closed = 1""")
c.execute("""create table span as select market_id, min(hour) t0, max(hour) t1, sum(usd) usd_total from h group by market_id""")

snaps = []
for name, hh in HORIZONS.items():
    q = f"""
    with s as (select market_id, t1 - {hh * 3600} tgt from span where t1 - t0 >= {hh * 3600}),
    b as (select h.market_id, arg_max(h.vwap, h.hour) ask1, sum(h.usd) ask_usd from h join s using (market_id)
          where h.buy1 and h.hour <= s.tgt and h.hour > s.tgt - 6*3600 group by h.market_id),
    x as (select h.market_id, arg_max(h.vwap, h.hour) bid1 from h join s using (market_id)
          where not h.buy1 and h.hour <= s.tgt and h.hour > s.tgt - 6*3600 group by h.market_id)
    select s.market_id, s.tgt, '{name}' horizon, b.ask1, b.ask_usd, x.bid1 from s left join b using (market_id) left join x using (market_id)
    where b.ask1 is not null or x.bid1 is not null"""
    snaps.append(c.execute(q).df())
snap = pd.concat(snaps)
meta = c.execute("select * from meta where win1 is not null").df()
d = snap.merge(meta, on='market_id')
print(f'{len(d):,} snapshots on {d.market_id.nunique():,} resolved markets')


def category(r):
    s = f"{r.event_slug or ''} {r.question or ''}".lower(); a = (r.answer1 or '').lower()
    if 'temperature' in s or 'weather' in s or 'rain' in s and 'precip' in s: return 'weather'
    if a in ('up',) or re.search(r'bitcoin|ethereum|solana|\bxrp\b|\bbtc\b|\beth\b|crypto|dogecoin|\bbnb\b|hyperliquid', s): return 'crypto'
    if a.startswith('over') or re.search(r'\bvs\.?\b|nba|nfl|mlb|nhl|epl|premier-league|la-liga|serie-a|bundesliga|ligue-1|ucl|uefa|fifa|atp|wta|tennis|ufc|' +
                                         r'esports|lol-|cs2|dota|valorant|spread|o/u|win on 20', s): return 'sports'
    if re.search(r'election|president|senate|house|governor|mayor|trump|biden|vance|democrat|republican|parliament|prime-minister|vote|nominee|primary', s): return 'politics'
    if re.search(r'fed|interest-rate|cpi|inflation|gdp|recession|s-p-500|nasdaq|stock|earnings|tesla|nvidia|apple|ipo|tariff', s): return 'finance'
    if re.search(r'ukraine|russia|israel|iran|china|taiwan|war|ceasefire|strike|nato|gaza|hamas|military', s): return 'geopolitics'
    return 'other'


cats = d[['market_id', 'event_slug', 'question', 'answer1']].drop_duplicates('market_id')
cats['cat'] = [category(r) for r in cats.itertuples()]
d = d.merge(cats[['market_id', 'cat']], on='market_id')
d['period'] = np.where(d['tgt'] < SPLIT, 'H1', 'H2')
d['rate'] = d['cat'].map(FEE)


def roi(price, win, rate):
    fee = rate * price * (1 - price)
    return win / price - 1 - fee / price  # per $ staked, holding to resolution


BINS = [0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, 1.0001]
rows = []
for side in ['tok1', 'tok2']:
    if side == 'tok1':
        x = d[d['ask1'].between(.005, .995)].assign(px=lambda t: t['ask1'], win=lambda t: t['win1'])
    else:
        x = d[d['bid1'].between(.005, .995)].assign(px=lambda t: 1 - t['bid1'], win=lambda t: 1 - t['win1'])
    x = x.assign(r=roi(x['px'], x['win'], x['rate']), r0=roi(x['px'], x['win'], 0), bin=pd.cut(x['px'], BINS, right=False))
    g = x.groupby(['cat', 'horizon', 'bin', 'period'], observed=True).agg(n=('r', 'size'), px=('px', 'mean'), win=('win', 'mean'),
                                                                            roi=('r', 'mean'), roi_nofee=('r0', 'mean'), se=('r', 'sem')).reset_index()
    g['side'] = side; rows.append(g)
res = pd.concat(rows)
res['bin'] = res['bin'].astype(str)
res.to_csv('results/a6_calibration_cells.csv', index=False)

# cells that are positive and reasonably sized in BOTH periods (stability filter)
p = res.pivot_table(index=['side', 'cat', 'horizon', 'bin'], columns='period', values=['n', 'roi', 'se']).dropna()
p.columns = [f'{a}_{b}' for a, b in p.columns]
good = p[(p.roi_H1 > 0) & (p.roi_H2 > 0) & (p.n_H1 >= 100) & (p.n_H2 >= 100)].copy()
good['t_H2'] = good.roi_H2 / good.se_H2
good = good.sort_values('t_H2', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('\n=== cells with ROI>0 after fees in both H1 (<May 2026) and H2, n>=100 each ===\n' + good.round(4).head(60).to_string())
print('\n=== category totals (n snapshots by side/period) ===\n' + res.groupby(['side', 'cat', 'period'])['n'].sum().unstack().to_string())

# market family age: family = event_slug with dates/numbers stripped
MONTHS = 'january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec'
fam = meta[['market_id', 'event_slug', 'created_at']].copy()
fam['family'] = fam['event_slug'].fillna('').str.replace(rf'\b({MONTHS})\b', '', regex=True).str.replace(r'\d+', '', regex=True).str.replace(r'-+', '-', regex=True).str.strip('-')
fam['fam_start'] = fam.groupby('family')['created_at'].transform('min')
fam['fam_size'] = fam.groupby('family')['market_id'].transform('size')
d = d.merge(fam[['market_id', 'family', 'fam_start', 'fam_size']], on='market_id')
d['age_days'] = (pd.to_datetime(d['tgt'], unit='s', utc=True) - d['fam_start']).dt.days
d['age'] = pd.cut(d['age_days'], [-1, 30, 90, 180, 10000], labels=['<1m', '1-3m', '3-6m', '>6m'])
y = d[(d['fam_size'] >= 50) & (d['horizon'] == '1d') & d['ask1'].between(.05, .95)].copy()
y['abs_err'] = (y['ask1'] - y['win1']).abs()
y['fav'] = y['ask1'] >= .5
y['r_fav'] = np.where(y['fav'], roi(y['ask1'], y['win1'], y['rate']), roi(1 - y['bid1'].fillna(1 - y['ask1']), 1 - y['win1'], y['rate']))
age = y.groupby(['cat', 'age'], observed=True).agg(n=('win1', 'size'), families=('family', 'nunique'), brier=('abs_err', lambda e: (e ** 2).mean()),
                                                   win_minus_px=('win1', lambda w: (w - y.loc[w.index, 'ask1']).mean())).round(4)
print('\n=== family age effect (1d horizon, families with >=50 markets) ===\n' + age.to_string())
json.dump(dict(good=good.round(4).reset_index().to_dict('records'), age=age.reset_index().astype(str).to_dict('records')),
          open('results/a6_allmarkets.json', 'w'), indent=1, default=str)
