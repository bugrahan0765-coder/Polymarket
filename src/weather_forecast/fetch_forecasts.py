"""Workstream weather-forecast, step 1-2: fetch historical day-ahead forecasts from Open-Meteo (Previous Runs API)
for every city, plus a lead-0 'realised' proxy (Historical Forecast API daily max).

No-look-ahead construction (decision time = local midnight starting the day BEFORE the target day D):
  * `temperature_2m_previous_dayN` = value predicted N*24 h before the valid time (Open-Meteo docs, Previous Runs API).
  * For valid hours 00-17 local on D we use previous_day2: run initialised <= D-2 17:00 local; global runs are on
    Open-Meteo within ~4-7 h, i.e. before D-1 00:00 (the decision).
  * For valid hours 18-23 local on D, previous_day2 would come from runs initialised D-2 18:00-23:00, which may only be
    published after the decision, so we use previous_day3 for those hours (init <= D-3 23:00).
  * tmax_strict = max(day2[00-17], day3[18-23]).  tmax_d2 = max(day2[00-23]) kept for comparison.
Coordinates: GUESS airport/observatory coordinates from src/a4_fetch_forecasts.py (UNVERIFIED resolution stations;
Polymarket Gamma is not reachable from this machine).
Output: data/weather_forecast/forecasts_daily.csv.gz  (city, date, model, tmax_strict, tmax_d2, hour_max)
        data/weather_forecast/lead0_daily.csv.gz      (city, date, tmax_lead0)  -- historical-forecast best_match
Run: python src/weather_forecast/fetch_forecasts.py
"""
import json, os, sys, time, urllib.request, urllib.parse
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
MODEL_GROUPS = [['best_match', 'ecmwf_ifs025', 'gfs_seamless', 'icon_seamless'],
                ['ukmo_seamless', 'gem_seamless', 'jma_seamless']]
START, END = '2025-10-01', '2026-10-17'
OUTDIR = 'data/weather_forecast'
CACHE = os.path.join(OUTDIR, 'cache')  # raw JSON per request (gitignored size-wise: deleted after build)


def get(url, path):
    if os.path.exists(path):
        return json.load(open(path))
    for a in range(6):
        try:
            j = json.load(urllib.request.urlopen(url, timeout=120))
            json.dump(j, open(path, 'w'))
            return j
        except Exception as e:
            print('  retry', a, e, flush=True); time.sleep(10 * (a + 1))
    raise RuntimeError(url)


def prev_runs(city, lat, lon, models, gi):
    q = dict(latitude=lat, longitude=lon, hourly='temperature_2m_previous_day2,temperature_2m_previous_day3',
             models=','.join(models), timezone='auto', start_date=START, end_date=END)
    return get('https://previous-runs-api.open-meteo.com/v1/forecast?' + urllib.parse.urlencode(q),
               os.path.join(CACHE, f'{city}_prev{gi}.json'))


def lead0(city, lat, lon):
    q = dict(latitude=lat, longitude=lon, daily='temperature_2m_max', timezone='auto', start_date=START, end_date='2026-10-16')
    return get('https://historical-forecast-api.open-meteo.com/v1/forecast?' + urllib.parse.urlencode(q),
               os.path.join(CACHE, f'{city}_lead0.json'))


def main():
    os.makedirs(CACHE, exist_ok=True)
    rows, l0 = [], []
    for city, (lat, lon) in GUESS.items():
        for gi, models in enumerate(MODEL_GROUPS):
            j = prev_runs(city, lat, lon, models, gi)
            h = pd.DataFrame(j['hourly'])
            h['date'] = h['time'].str[:10]; h['hour'] = h['time'].str[11:13].astype(int)
            for mdl in models:
                c2, c3 = f'temperature_2m_previous_day2_{mdl}', f'temperature_2m_previous_day3_{mdl}'
                if c2 not in h:  # single-model responses have no suffix
                    c2, c3 = 'temperature_2m_previous_day2', 'temperature_2m_previous_day3'
                if c2 not in h: continue
                x = h[['date', 'hour', c2, c3]].copy()
                x['strict'] = x[c2].where(x['hour'] <= 17, x[c3])
                for d, g in x.groupby('date'):
                    s, d2 = g['strict'], g[c2]
                    if s.notna().sum() < 20: continue
                    rows.append(dict(city=city, date=d, model=mdl, tmax_strict=s.max(), tmax_d2=d2.max() if d2.notna().sum() >= 20 else None,
                                     hour_max=int(g.loc[s.idxmax(), 'hour'])))
        j = lead0(city, lat, lon)
        l0 += [dict(city=city, date=d, tmax_lead0=v) for d, v in zip(j['daily']['time'], j['daily']['temperature_2m_max'])]
        print(city, 'ok', len(rows), flush=True); time.sleep(1)
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, 'forecasts_daily.csv.gz'), index=False)
    pd.DataFrame(l0).to_csv(os.path.join(OUTDIR, 'lead0_daily.csv.gz'), index=False)
    print('done', len(rows), len(l0))


if __name__ == '__main__':
    main()
