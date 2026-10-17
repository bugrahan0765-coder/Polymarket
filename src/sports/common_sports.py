"""Shared helpers for the sports workstream. UNTESTED (written without Polymarket API access).

Cache dir: data/raw/sports (gitignored via data/raw/).
"""
import json, os, time, hashlib, requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CACHE = os.path.join(ROOT, 'data', 'raw', 'sports')
os.makedirs(CACHE, exist_ok=True)

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
DATA = 'https://data-api.polymarket.com'

# Gamma tag slugs to inventory (verify on first run: GET /sports lists series/tags per league)
LEAGUE_TAGS = {
    'EPL': 'epl', 'LaLiga': 'la-liga', 'SerieA': 'serie-a', 'Bundesliga': 'bundesliga',
    'Ligue1': 'ligue-1', 'UCL': 'ucl', 'NBA': 'nba', 'NFL': 'nfl', 'MLB': 'mlb',
    'NHL': 'nhl', 'ATP': 'atp', 'WTA': 'wta',
}
# football-data.co.uk division codes
FD_DIV = {'EPL': 'E0', 'LaLiga': 'SP1', 'SerieA': 'I1', 'Bundesliga': 'D1', 'Ligue1': 'F1'}


def get_json(url, params=None, tries=6, sleep=0.25):
    """GET with on-disk cache and exponential backoff."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    fn = os.path.join(CACHE, 'http', key[:2], key + '.json')
    if os.path.exists(fn):
        with open(fn, encoding='utf8') as f:
            return json.load(f)
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=40)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            d = r.json()
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            with open(fn, 'w', encoding='utf8') as f:
                json.dump(d, f)
            time.sleep(sleep)
            return d
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def devig_power(odds):
    """Power method: find k with sum((1/o)^k)=1. Returns fair probabilities."""
    q = [1.0 / o for o in odds]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        s = sum(x ** k for x in q)
        if s > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    p = [x ** k for x in q]
    t = sum(p)
    return [x / t for x in p]


def devig_shin(odds):
    """Shin method (iterative). Returns fair probabilities."""
    q = [1.0 / o for o in odds]
    B = sum(q)
    z = 0.0
    for _ in range(100):
        p = [((z * z + 4 * (1 - z) * x * x / B) ** 0.5 - z) / (2 * (1 - z)) for x in q]
        z_new = (sum(((z * z + 4 * (1 - z) * x * x / B) ** 0.5) for x in q) - 2) / (len(q) - 2) if len(q) > 2 else z
        if abs(z_new - z) < 1e-10:
            break
        z = min(max(z_new, 0.0), 0.2)
    t = sum(p)
    return [x / t for x in p]


def taker_fee(price, shares, rate):
    """Polymarket taker fee model (VERIFY on docs.polymarket.com before use):
    fee = shares * rate * min(p, 1-p)  (older CLOB formula)  -- or --
    fee = shares * rate * p * (1-p)      (2025-26 fee-enabled markets).
    Set rate per market type/date from the docs; default 0 for fee-free markets."""
    return shares * rate * price * (1 - price)
