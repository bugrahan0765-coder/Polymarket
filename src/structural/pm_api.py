"""Read-only, cached, rate-limited client for Polymarket's PUBLIC endpoints.

Only GET/POST on public market-data endpoints (gamma events, CLOB books,
prices-history, data-api trades). No auth, no order endpoints, no wallets.

Raw responses are cached gzip-compressed under data/raw/structural/ (gitignored).
A global download budget (default 3 GB) stops runaway downloads.

UNTESTED against the live API from the author's machine (Polymarket is blocked
there); written from the documented response shapes.
"""
import gzip
import hashlib
import json
import os
import random
import threading
import time

import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(REPO, "data", "raw", "structural")
RESULTS = os.path.join(REPO, "results", "structural")


class BudgetExceeded(RuntimeError):
    pass


class Client:
    def __init__(self, cache_dir=RAW, max_bytes=3e9, min_interval=0.06, timeout=30):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.max_bytes = max_bytes
        self.min_interval = min_interval
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "structural-research/0.1 (read-only)"
        self.bytes = 0
        self.lock = threading.Lock()
        self._last = 0.0

    # ---- low level -------------------------------------------------------
    def _throttle(self):
        with self.lock:
            wait = self._last + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    def _key(self, method, url, params, body):
        raw = json.dumps([method, url, sorted((params or {}).items()), body], sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def _path(self, key, sub):
        d = os.path.join(self.cache_dir, sub, key[:2])
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, key + ".json.gz")

    def request(self, method, url, params=None, body=None, cache=True, sub="http", tries=7):
        key = self._key(method, url, params, body)
        path = self._path(key, sub)
        if cache and os.path.exists(path):
            with gzip.open(path, "rt") as f:
                return json.load(f)
        err = None
        for k in range(tries):
            if self.bytes > self.max_bytes:
                raise BudgetExceeded(f"download budget {self.max_bytes/1e9:.1f} GB exceeded")
            self._throttle()
            try:
                if method == "GET":
                    r = self.s.get(url, params=params, timeout=self.timeout)
                else:
                    r = self.s.post(url, params=params, json=body, timeout=self.timeout)
                self.bytes += len(r.content)
                if r.status_code == 429 or r.status_code >= 500:
                    err = RuntimeError(f"HTTP {r.status_code}")
                    time.sleep(min(60, (2 ** k) * 0.5 + random.random()))
                    continue
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                data = r.json()
                if cache:
                    with gzip.open(path, "wt") as f:
                        json.dump(data, f)
                return data
            except (requests.RequestException, ValueError) as ex:
                err = ex
                time.sleep(min(60, (2 ** k) * 0.5 + random.random()))
        raise RuntimeError(f"{method} {url} failed: {err}")

    def get(self, url, params=None, cache=True, sub="http"):
        return self.request("GET", url, params=params, cache=cache, sub=sub)

    # ---- gamma -----------------------------------------------------------
    def open_events(self, page=500, max_pages=400):
        """All currently open (active, not closed) events with nested markets."""
        out, seen, off = [], set(), 0
        for _ in range(max_pages):
            d = self.get(GAMMA + "/events", dict(closed="false", active="true", archived="false",
                                                  limit=page, offset=off), cache=False)
            if not d:
                break
            new = [e for e in d if e.get("id") not in seen]
            for e in new:
                seen.add(e.get("id"))
            out += new
            off += len(d)
            if not new:
                break
        return out

    # ---- clob ------------------------------------------------------------
    def books(self, token_ids, chunk=50, workers=None):
        """Order books for token ids. Tries batch POST /books, falls back to GET /book.
        Returns {token_id: raw_book}. Not cached per request (live data); callers
        archive whole snapshots."""
        out = {}
        ids = list(dict.fromkeys(t for t in token_ids if t))
        batch_ok = True
        for i in range(0, len(ids), chunk):
            part = ids[i:i + chunk]
            if batch_ok:
                try:
                    res = self.request("POST", CLOB + "/books", body=[{"token_id": t} for t in part],
                                       cache=False, tries=4)
                    if isinstance(res, list):
                        for b in res:
                            if isinstance(b, dict) and b.get("asset_id"):
                                out[str(b["asset_id"])] = b
                        continue
                except RuntimeError:
                    batch_ok = False
            for t in part:
                try:
                    b = self.get(CLOB + "/book", dict(token_id=t), cache=False)
                except RuntimeError:
                    b = None
                if b:
                    out[t] = b
        return out

    def prices_history(self, token, start_ts=None, end_ts=None, fidelity=60, interval=None):
        p = dict(market=token, fidelity=fidelity)
        if interval:
            p["interval"] = interval
        else:
            p["startTs"], p["endTs"] = int(start_ts), int(end_ts)
        d = self.get(CLOB + "/prices-history", p, sub="px")
        return (d or {}).get("history", [])

    # ---- data-api --------------------------------------------------------
    def taker_trades(self, condition_id, max_pages=20, page=500):
        out = []
        for k in range(max_pages):
            d = self.get(DATA + "/trades", dict(market=condition_id, takerOnly="true", limit=page,
                                                 offset=k * page), sub="trades")
            if not d:
                break
            out += d
            if len(d) < page:
                break
        return out


# ---- helpers on raw gamma objects --------------------------------------------
def jl(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except (TypeError, ValueError):
        return default


def tokens(m):
    """(token_yes_or_first, token_no_or_second, outcomes) for a gamma market."""
    t = jl(m.get("clobTokenIds"), []) or []
    o = jl(m.get("outcomes"), []) or []
    if len(t) != 2:
        return None, None, o
    return str(t[0]), str(t[1]), o


def norm_book(b):
    """raw CLOB book -> (bids desc [(p,s)], asks asc [(p,s)]) as floats."""
    if not b:
        return [], []
    bids = sorted(((float(x["price"]), float(x["size"])) for x in b.get("bids", []) if float(x["size"]) > 0),
                  key=lambda z: -z[0])
    asks = sorted(((float(x["price"]), float(x["size"])) for x in b.get("asks", []) if float(x["size"]) > 0),
                  key=lambda z: z[0])
    return bids, asks


def tag_slugs(e):
    return {str(t.get("slug", "")).lower() for t in (e.get("tags") or []) if isinstance(t, dict)}


def is_weather(e):
    s = (e.get("seriesSlug") or "") + " " + (e.get("slug") or "") + " " + (e.get("title") or "")
    s = s.lower()
    return ("highest-temperature" in s or "highest temperature" in s or "daily-weather" in s
            or bool(tag_slugs(e) & {"weather", "highest-temperature", "climate"}))
