import json,requests,sys,time,os,concurrent.futures as cf,threading
from common import meta
ev=json.load(open('events_all.json'))
done=set()
if os.path.exists('px.jsonl'):
    for l in open('px.jsonl'): done.add(json.loads(l)['tok'])
jobs=[]
for e in ev:
    mt=meta(e)
    if not mt: continue
    city,d,dec=mt
    for m in e['markets']:
        tok=json.loads(m['clobTokenIds'])[0]
        if tok in done: continue
        jobs.append((e['id'],m['id'],tok,dec))
print(len(jobs),flush=True)
lock=threading.Lock();f=open('px.jsonl','a');n=0
def go(j):
    eid,mid,tok,dec=j
    for k in range(6):
        try:
            r=requests.get('https://clob.polymarket.com/prices-history',params=dict(market=tok,startTs=dec-3600,endTs=dec,fidelity=1),timeout=40)
            if r.status_code==429: time.sleep(2+k*2);continue
            r.raise_for_status();h=r.json().get('history',[]);break
        except Exception as ex:
            time.sleep(1+k);h=None
    return dict(eid=eid,mid=mid,tok=tok,dec=dec,h=h)
with cf.ThreadPoolExecutor(24) as ex:
    for r in ex.map(go,jobs):
        if r['h'] is None: continue
        with lock:
            f.write(json.dumps(r)+'\n');n+=1
            if n%2000==0: f.flush();print(n,time.strftime('%X'),flush=True)
f.close();print('DONE',n)
