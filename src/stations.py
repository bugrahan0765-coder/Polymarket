"""Resolution stations per city (READ-ONLY, public Gamma API only). UNTESTED: written on a
machine where Polymarket APIs are blocked; run it in a cloud environment.

For every city in common.TZ it picks the most recent event and one Oct-Dec 2025 event from
data/decision_prices.csv.gz, fetches https://gamma-api.polymarket.com/events/<id>, and extracts
station name / ICAO / source URL / unit / rounding rule from the description and
resolutionSource. Writes results/stations.json.

lat/lon come from ICAO_COORDS below: model knowledge (approx., +-0.01 deg), NOT an
authoritative source; verify against e.g. ourairports.com before relying on them.
Run: python src/stations.py
"""
import json, re, sys, time
from pathlib import Path
import pandas as pd
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TZ

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'raw' / 'stations_cache'

ICAO_COORDS = {  # source: model knowledge, approximate
    'KLGA': (40.7769, -73.8740), 'KNYC': (40.7789, -73.9692), 'KJFK': (40.6413, -73.7781), 'EGLC': (51.5053, 0.0553),
    'EGLL': (51.4700, -0.4543), 'RKSI': (37.4602, 126.4407), 'RKSS': (37.5583, 126.7906), 'KDAL': (32.8471, -96.8518),
    'KDFW': (32.8998, -97.0403), 'CYYZ': (43.6777, -79.6248), 'KATL': (33.6407, -84.4277), 'LFPG': (49.0097, 2.5479),
    'LFPB': (48.9694, 2.4414), 'LFPO': (48.7233, 2.3794), 'LTAC': (40.1281, 32.9951), 'NZWN': (-41.3272, 174.8053),
    'SAEZ': (-34.8222, -58.5358), 'SABE': (-34.5592, -58.4156), 'KSEA': (47.4502, -122.3088), 'SBGR': (-23.4356, -46.4731),
    'SBSP': (-23.6261, -46.6564), 'KMIA': (25.7959, -80.2870), 'KORD': (41.9742, -87.9073), 'KMDW': (41.7868, -87.7522),
    'VILK': (26.7606, 80.8893), 'EDDM': (48.3538, 11.7861), 'LLBG': (32.0114, 34.8867), 'RJTT': (35.5494, 139.7798),
    'ZSPD': (31.1443, 121.8083), 'ZSSS': (31.1979, 121.3363), 'WSSS': (1.3644, 103.9915), 'VHHH': (22.3080, 113.9185),
    'HKO': (22.3022, 114.1742), 'LIMC': (45.6306, 8.7281), 'LIML': (45.4451, 9.2767), 'LEMD': (40.4719, -3.5626),
    'EPWA': (52.1657, 20.9671), 'RCTP': (25.0797, 121.2342), 'RCSS': (25.0694, 121.5525), 'ZUCK': (29.7192, 106.6417),
    'ZBAA': (40.0801, 116.5846), 'ZBAD': (39.5098, 116.4105), 'ZHHH': (30.7838, 114.2081), 'ZGSZ': (22.6393, 113.8107),
    'ZUUU': (30.5785, 103.9471), 'ZUTF': (30.3125, 104.4442), 'KDEN': (39.8561, -104.6737), 'KBKF': (39.7017, -104.7517),
    'KIAH': (29.9902, -95.3368), 'KHOU': (29.6454, -95.2789), 'KLAX': (33.9416, -118.4085), 'KSFO': (37.6213, -122.3790),
    'KAUS': (30.1975, -97.6664), 'LTFM': (41.2753, 28.7519), 'LTBA': (40.9769, 28.8146), 'UUEE': (55.9726, 37.4146),
    'UUWW': (55.5915, 37.2615), 'UUDD': (55.4088, 37.9063), 'MMMX': (19.4363, -99.0721), 'RKPK': (35.1795, 128.9382),
    'EHAM': (52.3105, 4.7683), 'EFHK': (60.3172, 24.9633), 'WMKK': (2.7456, 101.7072), 'MPTO': (9.0714, -79.3835),
    'OEJN': (21.6796, 39.1565), 'FACT': (-33.9715, 18.6021), 'ZGGG': (23.3924, 113.2988), 'OPKC': (24.9065, 67.1608),
    'RPLL': (14.5086, 121.0198), 'ZSQD': (36.3619, 120.0883), 'ZHCC': (34.5197, 113.8409), 'ZSJN': (36.8572, 117.2161),
    'WIII': (-6.1256, 106.6559), 'DNMM': (6.5774, 3.3212)}


def get(eid):
    p = CACHE / f'{eid}.json'
    if p.exists():
        return json.loads(p.read_text())
    for k in range(6):
        try:
            r = requests.get(f'https://gamma-api.polymarket.com/events/{eid}', timeout=30)
            if r.status_code == 200:
                CACHE.mkdir(parents=True, exist_ok=True); p.write_text(r.text); return r.json()
            if r.status_code != 429 and r.status_code < 500:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    return None


def extract(e):
    if not e:
        return None
    ms = e.get('markets') or [{}]
    text = ' '.join(filter(None, [e.get('description'), e.get('resolutionSource'), ms[0].get('description'), ms[0].get('resolutionSource')]))
    urls = sorted(set(u.rstrip('.,;') for u in re.findall(r'https?://[^\s\)\]"<>]+', text)))
    icao = None
    for u in urls:
        m = re.search(r'/([A-Z]{4})(?:[/?#]|$)', u)
        if m:
            icao = m.group(1); break
    if not icao:
        cands = [c for c in re.findall(r'\b([A-Z]{4})\b', text) if c in ICAO_COORDS]
        icao = cands[0] if cands else ('HKO' if 'Hong Kong Observatory' in text else None)
    name = re.search(r'(?:recorded|measured|reported)\s+(?:at|by)\s+(?:the\s+)?(.{3,90}?)(?:\s+(?:in|on|for|as)\b|[,.(])', text)
    titles = ' '.join(m.get('groupItemTitle') or m.get('question') or '' for m in ms)
    unit = '°F' if ('°F' in titles + text or 'Fahrenheit' in text) else ('°C' if ('°C' in titles + text or 'Celsius' in text) else None)
    rnd = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if re.search(r'whole|round|decimal|precision', s, re.I)]
    st = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if re.search(r'station|airport|observatory', s, re.I)]
    src = [u for u in urls if 'wunderground' in u] or urls
    return dict(event_id=e.get('id'), slug=e.get('slug'), end=e.get('endDate'), icao=icao,
                name=name.group(1).strip() if name else None, unit=unit, source_url=src[0] if src else None,
                rounding=' '.join(rnd) or None, station_text=' '.join(st)[:600] or None)


def main():
    d = pd.read_csv(ROOT / 'data' / 'decision_prices.csv.gz', usecols=['event_id', 'city', 'target_date']).drop_duplicates('event_id')
    out = {}
    for city in TZ:
        g = d[d.city == city].sort_values('target_date')
        if g.empty:
            continue
        old = g[(g.target_date >= '2025-10-01') & (g.target_date <= '2025-12-31')]
        rec = extract(get(int(g.event_id.iloc[-1])))
        o = extract(get(int(old.event_id.iloc[len(old) // 2]))) if len(old) else None
        if not rec:
            out[city] = dict(error='fetch failed'); continue
        ll = ICAO_COORDS.get(rec['icao'])
        if o is None:
            note = 'no Oct-Dec 2025 event in data'
        elif (o['icao'], o['source_url']) == (rec['icao'], rec['source_url']):
            note = f"same as {o['end'][:10] if o['end'] else '?'} event"
        else:
            note = f"CHANGED: old {o['icao']} {o['source_url']} -> new {rec['icao']} {rec['source_url']}"
        out[city] = dict(icao=rec['icao'], name=rec['name'], lat=ll[0] if ll else None, lon=ll[1] if ll else None,
                         latlon_source='model knowledge (approx)' if ll else None, unit=rec['unit'], source_url=rec['source_url'],
                         rounding=rec['rounding'], changed_note=note, recent_event=rec, old_event=o)
        print(city, rec['icao'], rec['unit'], rec['source_url'], '|', note, flush=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'stations.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
