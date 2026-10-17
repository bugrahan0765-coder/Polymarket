import json,csv,gzip,collections
from common import meta
ev={e['id']:e for e in json.load(open('events_all.json'))}
px={}
for l in open('px.jsonl'):
    r=json.loads(l);px[r['mid']]=r
n=0
with gzip.open('decision_prices.csv.gz','wt',newline='') as f:
    w=csv.writer(f);w.writerow(['event_id','city','target_date','decision_ts','market_id','bucket','outcome_yes','closed_time','price_last_le_dec','price_age_s','n_samples_1h'])
    for eid,e in ev.items():
        mt=meta(e)
        if not mt: continue
        city,d,dec=mt
        for m in e['markets']:
            r=px.get(m['id'])
            h=[p for p in (r['h'] if r else []) if p['t']<=dec]
            last=max(h,key=lambda p:p['t']) if h else None
            op=m.get('outcomePrices')
            res={'["1", "0"]':1,'["0", "1"]':0}.get(op,'')
            w.writerow([eid,city,d,dec,m['id'],m.get('groupItemTitle'),res,m.get('closedTime') or '',last['p'] if last else '',dec-last['t'] if last else '',len(h)]);n+=1
print(n)
