"""Step 4a: fetch historical forecasts (made >= 48h before each hour, so available before the decision time)
from the Open-Meteo Previous Runs API for every city. Output: data/forecasts.csv.gz (city, local date, model, tmax_c).
Coordinates: results/stations.json if present (resolution stations), else airport guesses below.
Run: python src/a4_fetch_forecasts.py
"""
import json, os, time, urllib.request, urllib.parse
import pandas as pd

GUESS = {'nyc': (40.7769, -73.8740), 'london': (51.5053, 0.0553), 'seoul': (37.4602, 126.4407), 'dallas': (32.8471, -96.8518),
         'toronto': (43.6777, -79.6248), 'atlanta': (33.6407, -84.4277), 'paris': (49.0097, 2.5479), 'ankara': (40.1281, 32.9951),
         'wellington': (-41.3272, 174.8053), 'buenos-aires': (-34.8222, -58.5358), 'seattle': (47.4502, -122.3088),
         'sao-paulo': (-23.4356, -46.4731), 'miami': (25.7959, -80.2870), 'chicago': (41.9742, -87.9073), 'lucknow': (26.7606, 80.8893),
         'munich': (48.3538, 11.7861), 'tel-aviv': (32.0055, 34.8854), 'tokyo': (35.5494, 139.7798), 'shanghai': (31.1443, 121.8083),
         'singapore': (1.3644, 103.9915), 'hong-kong': (22.3022, 114.1742), 'milan': (45.6306, 8.7281), 'madrid': (40.4983, -3.5676),
         'warsaw': (52.1657, 20.9671), 'taipei': (25.0694, 121.5525), 'chongqing': (29.7192, 106.6417), 'beijing': (40.0799, 116.6031),
         'wuhan': (30.7838, 114.2081), 'shenzhen': (22.6393, 113.8107), 'chengdu': (30.5785, 103.9471), 'denver': (39.8561, -104.6737),
         'houston': (29.6454, -95.2789), 'los-angeles': (33.9416, -118.4085), 'san-francisco': (37.6213, -122.3790),
         'austin': (30.1975, -97.6664), 'istanbul': (41.2753, 28.7519), 'moscow': (55.9726, 37.4146), 'mexico-city': (19.4361, -99.0719),
         'busan': (35.1795, 128.9382), 'amsterdam': (52.3105, 4.7683), 'helsinki': (60.3172, 24.9633), 'kuala-lumpur': (2.7456, 101.7072),
         'panama': (9.0714, -79.3835), 'jeddah': (21.6796, 39.1565), 'cape-town': (-33.9715, 18.6021), 'guangzhou': (23.3924, 113.2988),
         'karachi': (24.9065, 67.1608), 'manila': (14.5086, 121.0198), 'qingdao': (36.2661, 120.3744), 'zhengzhou': (34.5197, 113.8409),
         'jinan': (36.8572, 117.2160), 'jakarta': (-6.1256, 106.6559), 'lagos': (6.5774, 3.3212)}
MODELS = ['best_match', 'ecmwf_ifs025', 'gfs_seamless', 'icon_seamless']
VAR = 'temperature_2m_previous_day2'
OUT = 'data/forecasts.csv.gz'


def coords():
    c = dict(GUESS)
    if os.path.exists('results/stations.json'):
        for city, s in json.load(open('results/stations.json')).items():
            if s.get('lat') is not None and s.get('lon') is not None: c[city] = (float(s['lat']), float(s['lon']))
    return c


def fetch(lat, lon):
    q = dict(latitude=lat, longitude=lon, hourly=VAR, models=','.join(MODELS), timezone='auto',
             start_date='2025-10-01', end_date='2026-10-17')
    url = 'https://previous-runs-api.open-meteo.com/v1/forecast?' + urllib.parse.urlencode(q)
    for a in range(5):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception as e:
            print('  retry', a, e); time.sleep(5 * (a + 1))
    raise RuntimeError(url)


rows = []
for city, (lat, lon) in coords().items():
    j = fetch(lat, lon)
    h = pd.DataFrame(j['hourly']); h['date'] = h['time'].str[:10]
    for mdl in MODELS:
        col = f'{VAR}_{mdl}' if f'{VAR}_{mdl}' in h else (VAR if mdl == MODELS[0] and VAR in h else None)
        if col is None: continue
        g = h.groupby('date')[col].agg(['max', 'count'])
        g = g[g['count'] >= 20]  # need most hours of the local day
        rows += [dict(city=city, date=d, model=mdl, tmax_c=v) for d, v in g['max'].dropna().items()]
    print(city, lat, lon, 'ok'); time.sleep(.5)
pd.DataFrame(rows).to_csv(OUT, index=False)
print('wrote', OUT, len(rows))
