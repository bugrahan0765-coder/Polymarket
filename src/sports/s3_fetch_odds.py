"""Step 3: free historical odds (UNTESTED).

Soccer: football-data.co.uk season CSVs (mmz4281/<season>/<div>.csv). Columns: PSH/PSD/PSA (Pinnacle,
  collected ~Fri/Tue before match), PSCH/PSCD/PSCA (Pinnacle closing), plus AvgC*/MaxC* market closing.
  NOTE: Pinnacle closed its public API in July 2025; check whether PS*/PSC* are populated for 2025-26
  onward. If empty, fall back to AvgCH/AvgCD/AvgCA (market-average closing) and say so in the summary.
  football-data.co.uk is blocked from the user's Turkish connection -> run in the cloud.
US sports: ESPN core API odds per event (provider open/close moneylines, usually DraftKings/ESPN BET):
  https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/events/{id}/competitions/{id}/odds
  Not a sharp book; treat de-vigged close as a consensus benchmark, not Pinnacle. Coverage to be measured.
"""
import os, io, sys, json, datetime as dt
import pandas as pd
from common_sports import CACHE, FD_DIV, get_json
import requests

out = os.path.join(CACHE, 'odds')
os.makedirs(out, exist_ok=True)

# --- soccer
frames = []
for season in ('2425', '2526', '2627'):
    for lg, div in FD_DIV.items():
        fn = os.path.join(out, f'fd_{div}_{season}.csv')
        if not os.path.exists(fn):
            r = requests.get(f'https://www.football-data.co.uk/mmz4281/{season}/{div}.csv', timeout=60)
            r.raise_for_status()
            open(fn, 'wb').write(r.content)
        df = pd.read_csv(fn, encoding='latin1')
        df['league'] = lg
        frames.append(df)
fd = pd.concat(frames, ignore_index=True)
fd.to_csv(os.path.join(out, 'football_data_all.csv'), index=False)
print('soccer rows', len(fd), 'PSCH non-null', fd.get('PSCH', pd.Series(dtype=float)).notna().sum())

# --- US sports via ESPN scoreboard (event ids per date) + core odds endpoint
ESPN = {'NBA': ('basketball', 'nba'), 'NFL': ('football', 'nfl'), 'MLB': ('baseball', 'mlb'), 'NHL': ('hockey', 'nhl')}
a = dt.date.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else '2025-07-01')
b = dt.date.fromisoformat(sys.argv[2] if len(sys.argv) > 2 else '2026-10-15')
rows = []
for lg, (sport, league) in ESPN.items():
    d = a
    while d <= b:
        sb = get_json(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard',
                      dict(dates=d.strftime('%Y%m%d')))
        for ev in sb.get('events', []):
            comp = ev['competitions'][0]
            teams = {c['homeAway']: c['team']['displayName'] for c in comp['competitors']}
            win = {c['homeAway']: c.get('winner') for c in comp['competitors']}
            try:
                od = get_json(f'https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}/events/{ev["id"]}/competitions/{comp["id"]}/odds')
            except Exception:
                od = {}
            for it in od.get('items', []):
                h, w = it.get('homeTeamOdds', {}), it.get('awayTeamOdds', {})
                rows.append(dict(league=lg, espn_id=ev['id'], date=ev['date'], home=teams.get('home'), away=teams.get('away'),
                                 home_win=win.get('home'), provider=it.get('provider', {}).get('name'),
                                 h_open=h.get('open', {}).get('moneyLine', {}).get('american') if isinstance(h.get('open'), dict) else None,
                                 a_open=w.get('open', {}).get('moneyLine', {}).get('american') if isinstance(w.get('open'), dict) else None,
                                 h_close=h.get('close', {}).get('moneyLine', {}).get('american') if isinstance(h.get('close'), dict) else h.get('moneyLine'),
                                 a_close=w.get('close', {}).get('moneyLine', {}).get('american') if isinstance(w.get('close'), dict) else w.get('moneyLine')))
        d += dt.timedelta(days=1)
    print(lg, len(rows), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, 'espn_us_odds.csv'), index=False)
