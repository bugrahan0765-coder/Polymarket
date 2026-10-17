"""Step 1: inventory closed Polymarket sports game events (UNTESTED; needs Polymarket API access).

Output: data/raw/sports/inventory.jsonl  (one row per market: league, event slug, question,
conditionId, clobTokenIds, outcomes, gameStartTime, closedTime, outcomePrices(final)).
Run: python -I src/sports/s1_inventory.py 2025-07-01 2026-10-15
"""
import sys, json, os, datetime as dt
from common_sports import GAMMA, CACHE, LEAGUE_TAGS, get_json

a = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else '2025-07-01')
b = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else '2026-10-15')
out = open(os.path.join(CACHE, 'inventory.jsonl'), 'w', encoding='utf8')
n = 0
for league, tag in LEAGUE_TAGS.items():
    d = a
    while d < b:
        e = min(d + dt.timedelta(days=7), b)
        off = 0
        while True:
            evs = get_json(f'{GAMMA}/events', dict(tag_slug=tag, closed='true', limit=100, offset=off,
                                                   end_date_min=d.isoformat(), end_date_max=e.isoformat()))
            if not evs:
                break
            for ev in evs:
                for m in ev.get('markets', []):
                    # keep game-level markets only (moneyline / "Will X win on DATE?" / draw); skip props, spreads, totals
                    smt = (m.get('sportsMarketType') or '').lower()
                    q = m.get('question', '')
                    if smt and smt not in ('moneyline', 'h2h', 'winner'):
                        continue
                    if any(w in q.lower() for w in ('spread', 'o/u', 'over/under', 'total', 'points', '1h ', 'half')):
                        continue
                    out.write(json.dumps(dict(
                        league=league, event_slug=ev.get('slug'), event_title=ev.get('title'),
                        question=q, conditionId=m.get('conditionId'), tokens=m.get('clobTokenIds'),
                        outcomes=m.get('outcomes'), outcomePrices=m.get('outcomePrices'),
                        gameStartTime=m.get('gameStartTime') or ev.get('startTime'), endDate=m.get('endDate'),
                        closedTime=m.get('closedTime'), volume=m.get('volumeNum'),
                        feesEnabled=m.get('feesEnabled'), sportsMarketType=m.get('sportsMarketType'))) + '\n')
                    n += 1
            if len(evs) < 100:
                break
            off += 100
        d = e
    print(league, n, flush=True)
out.close()
