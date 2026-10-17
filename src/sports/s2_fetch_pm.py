"""Step 2: per market, fetch price history around kickoff and taker trades (UNTESTED).

prices-history: CLOB /prices-history?market=<token>&startTs&endTs&fidelity=1  (last/mid, NOT ask)
trades:         data-api /trades?market=<conditionId>&takerOnly=true&limit=500&offset=N
                -> fields used: side (BUY/SELL by taker), outcome/outcomeIndex, asset, price, size, timestamp.
                Verify on a few markets that BUY + asset==token means taker paid `price` for that token.
Outputs: data/raw/sports/px/<conditionId>.json , data/raw/sports/trades/<conditionId>.json
Run: python -I src/sports/s2_fetch_pm.py
"""
import json, os, datetime as dt, concurrent.futures as cf
from common_sports import CLOB, DATA, CACHE, get_json

inv = [json.loads(l) for l in open(os.path.join(CACHE, 'inventory.jsonl'), encoding='utf8')]
os.makedirs(os.path.join(CACHE, 'px'), exist_ok=True)
os.makedirs(os.path.join(CACHE, 'trades'), exist_ok=True)


def ts(s):
    return int(dt.datetime.fromisoformat(s.replace('Z', '+00:00').replace(' ', 'T')).timestamp())


def one(m):
    cid = m['conditionId']
    if not m.get('gameStartTime'):
        return
    ko = ts(m['gameStartTime'])
    fpx = os.path.join(CACHE, 'px', cid + '.json')
    if not os.path.exists(fpx):
        toks = json.loads(m['tokens']) if isinstance(m['tokens'], str) else m['tokens']
        res = {}
        for t in toks:
            res[t] = get_json(f'{CLOB}/prices-history',
                              dict(market=t, startTs=ko - 48 * 3600, endTs=ko + 4 * 3600, fidelity=1)).get('history', [])
        json.dump(res, open(fpx, 'w'))
    ftr = os.path.join(CACHE, 'trades', cid + '.json')
    if not os.path.exists(ftr):
        rows, off = [], 0
        while off <= 10000:  # data-api offset cap; enough for pre-game window on most markets
            d = get_json(f'{DATA}/trades', dict(market=cid, takerOnly='true', limit=500, offset=off))
            if not d:
                break
            rows += d
            if len(d) < 500:
                break
            off += 500
        json.dump(rows, open(ftr, 'w'))


with cf.ThreadPoolExecutor(4) as ex:
    for i, _ in enumerate(ex.map(one, inv)):
        if i % 500 == 0:
            print(i, len(inv), flush=True)
