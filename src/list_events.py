import json,requests,datetime as dt,concurrent.futures as cf
S='https://gamma-api.polymarket.com/events'
def fetch(day):
    out=[];off=0
    a=day.isoformat()+'T00:00:00Z';b=(day+dt.timedelta(days=1)).isoformat()+'T00:00:00Z'
    while True:
        for _ in range(4):
            try:
                r=requests.get(S,params=dict(tag_slug='highest-temperature',closed='true',limit=100,offset=off,end_date_min=a,end_date_max=b),timeout=40);r.raise_for_status();break
            except Exception as ex: err=ex
        else: raise err
        d=r.json();out+=d
        if len(d)<100:break
        off+=100
    return out
days=[dt.date(2025,10,15)+dt.timedelta(days=i) for i in range(367)]
res={}
with cf.ThreadPoolExecutor(12) as ex:
    for day,o in zip(days,ex.map(fetch,days)): res[day]=o
ev={}
for o in res.values():
    for e in o: ev[e['id']]=e
json.dump(list(ev.values()),open('events.json','w'))
print(len(ev))
import collections
c=collections.Counter(e.get('seriesSlug') for e in ev.values());print(len(c),c.most_common(60))
