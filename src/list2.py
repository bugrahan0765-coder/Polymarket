import json,requests,datetime as dt,concurrent.futures as cf
def fetch(day):
    out=[];off=0
    a=day.isoformat()+'T00:00:00Z';b=(day+dt.timedelta(days=1)).isoformat()+'T00:00:00Z'
    while True:
        for _ in range(4):
            try:
                r=requests.get('https://gamma-api.polymarket.com/events',params=dict(tag_slug='weather',closed='true',limit=100,offset=off,end_date_min=a,end_date_max=b),timeout=40);r.raise_for_status();break
            except Exception as ex: err=ex
        else: raise err
        d=r.json();out+=d
        if len(d)<100:break
        off+=100
    return out
days=[dt.date(2025,10,15)+dt.timedelta(days=i) for i in range(125)]
old={e['id'] for e in json.load(open('events.json'))}
new={}
with cf.ThreadPoolExecutor(12) as ex:
    for o in ex.map(fetch,days):
        for e in o:
            if e['id'] not in old and e['title'].startswith('Highest temperature in'): new[e['id']]=e
print(len(new))
import collections
print(collections.Counter((e.get('seriesSlug'),e['endDate'][:7]) for e in new.values()).most_common(30))
json.dump(list(new.values()),open('events2.json','w'))
