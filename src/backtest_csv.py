"""Reproduces the 12-month backtest from data/decision_prices.csv.gz (no API needed).
Run: python3 src/backtest_csv.py
"""
import csv,gzip,collections,datetime as dt
from zoneinfo import ZoneInfo
from common import TZ

def load(path='data/decision_prices.csv.gz',win=120):
    ev=collections.OrderedDict()
    for r in csv.DictReader(gzip.open(path,'rt')):
        e=ev.setdefault(r['event_id'],dict(city=r['city'],d=dt.date.fromisoformat(r['target_date']),dec=int(r['decision_ts']),ms=[]))
        age=r['price_age_s']
        p=float(r['price_last_le_dec']) if r['price_last_le_dec']!='' and int(age)<=win else None
        e['ms'].append(dict(id=r['market_id'],bucket=r['bucket'],p=p,yes=r['outcome_yes'],closed=r['closed_time']))
    out=[]
    for eid,e in ev.items():
        priced=[m for m in e['ms'] if m['p'] is not None]
        if not priced: out.append(dict(eid=eid,city=e['city'],d=e['d'],dec=e['dec'],fav=None));continue
        f=max(priced,key=lambda m:m['p'])
        if f['yes']=='': continue  # unresolved
        tz=ZoneInfo(TZ[e['city']]);nxt=dt.datetime(e['d'].year,e['d'].month,e['d'].day,tzinfo=tz)+dt.timedelta(days=1)
        rel=nxt.timestamp()
        if f['closed']:
            rel=max(rel,dt.datetime.fromisoformat(f['closed'].replace('+00','+00:00')).timestamp())
        out.append(dict(eid=eid,city=e['city'],d=e['d'],dec=e['dec'],fav=f['id'],bucket=f['bucket'],p=f['p'],win=f['yes']=='1',rel=rel,n=len(e['ms']),n_priced=len(priced)))
    return out

def sim(cands,start=50.,cap=5.,slip=.01,lo=.6,hi=.9,fee=0.,cash_limited=True,minshares=5):
    cash=start;open_=[];trades=[]
    for c in sorted([c for c in cands if c['fav']],key=lambda c:c['dec']):
        for o in list(open_):
            if o['rel']<=c['dec']: cash+=o['payout'];open_.remove(o)
        bp=round(c['p']+slip,4)
        if not(lo<=bp<hi): continue
        budget=min(cap,cash) if cash_limited else cap
        sh=budget/bp
        if sh<minshares: continue
        cost=sh*bp;f=sh*fee*bp*(1-bp);cash-=cost+f
        payout=sh if c['win'] else 0.
        t=dict(c,bp=bp,sh=sh,cost=cost+f,payout=payout,pnl=payout-cost-f)
        open_.append(t);trades.append(t)
    return trades,cash+sum(o['payout'] for o in open_)

if __name__=='__main__':
    c=load();t,end=sim(c)
    print(len(c),'events;',len(t),'trades;',sum(x['win'] for x in t),'wins; end balance',round(end,2))
