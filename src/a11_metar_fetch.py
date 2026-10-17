"""Step 11a: fetch METAR temperature observations (Iowa Environmental Mesonet ASOS archive, free) for candidate
resolution stations of every city, Oct 2025 - Oct 2026. Output data/metar/<ICAO>.csv.gz (valid UTC, tmpf).
Run: python src/a11_metar_fetch.py
"""
import os, time, urllib.request, urllib.parse

CANDIDATES = {
    'nyc': ['KLGA', 'KJFK', 'KNYC'], 'london': ['EGLC', 'EGLL'], 'seoul': ['RKSI', 'RKSS'], 'dallas': ['KDAL', 'KDFW'],
    'toronto': ['CYYZ', 'CYTZ'], 'atlanta': ['KATL'], 'paris': ['LFPG', 'LFPO', 'LFPB'], 'ankara': ['LTAC'], 'wellington': ['NZWN'],
    'buenos-aires': ['SAEZ', 'SABE'], 'seattle': ['KSEA'], 'sao-paulo': ['SBGR', 'SBSP'], 'miami': ['KMIA'], 'chicago': ['KORD', 'KMDW'],
    'lucknow': ['VILK'], 'munich': ['EDDM'], 'tel-aviv': ['LLBG'], 'tokyo': ['RJTT'], 'shanghai': ['ZSPD', 'ZSSS'], 'singapore': ['WSSS'],
    'hong-kong': ['VHHH'], 'milan': ['LIMC', 'LIML'], 'madrid': ['LEMD'], 'warsaw': ['EPWA'], 'taipei': ['RCTP', 'RCSS'],
    'chongqing': ['ZUCK'], 'beijing': ['ZBAA', 'ZBAD'], 'wuhan': ['ZHHH'], 'shenzhen': ['ZGSZ'], 'chengdu': ['ZUUU', 'ZUTF'],
    'denver': ['KDEN', 'KBKF'], 'houston': ['KHOU', 'KIAH'], 'los-angeles': ['KLAX'], 'san-francisco': ['KSFO'], 'austin': ['KAUS'],
    'istanbul': ['LTFM', 'LTBA'], 'moscow': ['UUEE', 'UUWW', 'UUDD'], 'mexico-city': ['MMMX'], 'busan': ['RKPK'], 'amsterdam': ['EHAM'],
    'helsinki': ['EFHK'], 'kuala-lumpur': ['WMKK'], 'panama': ['MPTO'], 'jeddah': ['OEJN'], 'cape-town': ['FACT'], 'guangzhou': ['ZGGG'],
    'karachi': ['OPKC'], 'manila': ['RPLL'], 'qingdao': ['ZSQD'], 'zhengzhou': ['ZHCC'], 'jinan': ['ZSJN'], 'jakarta': ['WIII'], 'lagos': ['DNMM']}
OUT = 'data/metar'


def fetch(icao):
    q = [('station', icao[1:] if icao.startswith('K') and len(icao) == 4 else icao), ('data', 'tmpf'), ('year1', 2025), ('month1', 9), ('day1', 25),
         ('year2', 2026), ('month2', 10), ('day2', 18), ('tz', 'Etc/UTC'), ('format', 'onlycomma'), ('latlon', 'no'), ('missing', 'M'),
         ('direct', 'no'), ('report_type', 3), ('report_type', 4)]
    url = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?' + urllib.parse.urlencode(q)
    for a in range(5):
        try:
            return urllib.request.urlopen(url, timeout=300).read()
        except Exception as e:
            print('  retry', icao, e); time.sleep(20 * (a + 1))
    raise RuntimeError(icao)


if __name__ == '__main__':
    import gzip
    os.makedirs(OUT, exist_ok=True)
    for city, icaos in CANDIDATES.items():
        for icao in icaos:
            p = f'{OUT}/{icao}.csv.gz'
            if os.path.exists(p): continue
            b = fetch(icao)
            with gzip.open(p, 'wb') as fh: fh.write(b)
            print(city, icao, b.count(b'\n'), 'rows', flush=True)
            time.sleep(2)
