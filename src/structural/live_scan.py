"""LIVE structural-arbitrage scanner (read-only). UNTESTED against the real API
from the author's machine (Polymarket blocked in Turkey); run it from a cloud box.

Every --every seconds (default 600) for --scans iterations:
  1. list all open events (gamma), 2. pull YES-token order books (CLOB; NO book is
  the mirror image), 3. evaluate
     * negRisk baskets: buy-all-YES (payout 1), buy-all-NO (payout n-1), partial NO
       (subset of legs with books, payout k-1)
     * weather city-day events (subset of the above, tagged)
     * ladders (date / threshold / O-U), duplicates, nested nominee->winner
  4. append per-event stats + opportunities to results/structural/live/*.jsonl,
     archive the trimmed snapshot (top 15 levels) under data/raw/structural/books/.

Usage:
  python src/structural/live_scan.py --scans 18 --every 600
  python src/structural/live_scan.py --scans 1 --max-events 300   # quick smoke test
Then: python src/structural/summarise_live.py
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book_math import basket_depth, category, fee_rate, no_asks_from_yes_bids, pair_depth  # noqa: E402
from ladders import duplicate_pairs, ladder_pairs, nested_pairs  # noqa: E402
from pm_api import RAW, RESULTS, Client, is_weather, jl, norm_book, tag_slugs, tokens  # noqa: E402

OUT = os.path.join(RESULTS, "live")


def now():
    return dt.datetime.now(dt.timezone.utc)


def days_to(end):
    try:
        e = dt.datetime.fromisoformat((end or "").replace("Z", "+00:00"))
        return max((e - now()).total_seconds() / 86400, 0.0)
    except ValueError:
        return None


def resolved_yes(m):
    p = jl(m.get("outcomePrices"), []) or []
    try:
        return m.get("closed") and float(p[0]) > 0.99
    except (IndexError, TypeError, ValueError):
        return False


def min_size(m, book):
    for v in ((book or {}).get("min_order_size"), m.get("orderMinSize")):
        try:
            if v is not None:
                return float(v)
        except (TypeError, ValueError):
            pass
    return 5.0


def eval_negrisk(e, books, rate, ts):
    mk = [m for m in e.get("markets") or [] if tokens(m)[0]]
    if any(resolved_yes(m) for m in mk):
        return None, []
    open_ = [m for m in mk if not m.get("closed") and m.get("active", True)
             and m.get("enableOrderBook", True) is not False]
    if len(open_) < 2:
        return None, []
    nb = [norm_book(books.get(tokens(m)[0])) for m in open_]
    bids = [b for b, _ in nb]
    asks = [a for _, a in nb]
    n = len(open_)
    mins = max(min_size(m, books.get(tokens(m)[0])) for m in open_)
    sum_ask = sum(a[0][0] for a in asks if a) if all(asks) else None
    sum_bid = sum(b[0][0] for b in bids if b)
    days = days_to(e.get("endDate"))
    aug = bool(e.get("negRiskAugmented"))
    has_other = any((m.get("groupItemTitle") or "").strip().lower() in ("other", "another person", "none")
                    for m in open_)
    stat = dict(ts=ts, event_id=e.get("id"), slug=e.get("slug"), title=e.get("title"), n=n,
                weather=is_weather(e), augmented=aug, has_other=has_other, rate=rate,
                sum_yes_ask=sum_ask, sum_yes_bid=sum_bid, n_missing_ask=sum(1 for a in asks if not a),
                n_missing_bid=sum(1 for b in bids if not b), days=days, min_size=mins,
                volume24=e.get("volume24hr"), liquidity=e.get("liquidity"))
    opps = []
    variants = []
    variants.append(("yes_all", asks, 1.0))
    if all(bids):
        variants.append(("no_all", [no_asks_from_yes_bids(b) for b in bids], n - 1.0))
    else:
        sub = [no_asks_from_yes_bids(b) for b in bids if b]
        if len(sub) >= 2:
            variants.append(("no_partial", sub, len(sub) - 1.0))
    for kind, legs, payout in variants:
        r = basket_depth(legs, payout, rate)
        rc = basket_depth(legs, payout, rate, conservative=True)
        stat[kind + "_edge"] = r["top_edge"]
        if r["top_edge"] is not None and r["top_edge"] > 0:
            opps.append(dict(ts=ts, kind="negrisk_" + kind, key=f"{e.get('id')}:{kind}", event_id=e.get("id"),
                             slug=e.get("slug"), title=e.get("title"), weather=stat["weather"], n=n,
                             payout=payout, top_edge=r["top_edge"], sets=r["sets"], cost=r["cost"],
                             profit=r["profit"], profit_conservative_fee=rc["profit"],
                             sets_conservative_fee=rc["sets"], executable=r["sets"] >= mins,
                             min_size=mins, days=days, augmented=aug, has_other=has_other,
                             # yes_all on an augmented event without an 'Other' leg is NOT riskless
                             riskless=not (kind == "yes_all" and aug and not has_other)))
    return stat, opps


def eval_pair(kind, hi, lo, books, rate, ts, extra):
    th, tl = tokens(hi)[0], tokens(lo)[0]
    bh, ah = norm_book(books.get(th))
    bl, al = norm_book(books.get(tl))
    if not ah or not bl:
        return None
    r = pair_depth(ah, bl, rate)
    if r["top_edge"] is None or r["top_edge"] <= 0:
        return None
    rc = pair_depth(ah, bl, rate, conservative=True)
    mins = max(min_size(hi, books.get(th)), min_size(lo, books.get(tl)))
    return dict(ts=ts, kind=kind, key=f"{kind}:{hi.get('conditionId')}:{lo.get('conditionId')}",
                hi_q=hi.get("question"), lo_q=lo.get("question"), hi_ask=ah[0][0], lo_bid=bl[0][0],
                top_edge=r["top_edge"], sets=r["sets"], cost=r["cost"], profit=r["profit"],
                profit_conservative_fee=rc["profit"], executable=r["sets"] >= mins, min_size=mins,
                days=max(filter(None, [days_to(hi.get("endDate")), days_to(lo.get("endDate"))]), default=None),
                riskless=(kind != "nested"), **extra)


def scan_once(cl, max_events=None):
    ts = now().isoformat()
    events = cl.open_events()
    if max_events:
        events = events[:max_events]
    neg = [e for e in events if (e.get("negRisk") or e.get("enableNegRisk")) and len(e.get("markets") or []) >= 2]
    lad, toks = [], set()
    neg_ids = {e.get("id") for e in neg}
    for e in neg:
        for m in e.get("markets") or []:
            if not m.get("closed") and tokens(m)[0]:
                toks.add(tokens(m)[0])
    for e in events:
        if e.get("id") in neg_ids:
            continue
        ps = list(ladder_pairs(e))
        if ps:
            lad.append((e, ps))
            for _, hi, lo, _, _ in ps:
                toks.add(tokens(hi)[0])
                toks.add(tokens(lo)[0])
    dups = list(duplicate_pairs(events))
    nests = list(nested_pairs(events))
    for _, a, b, _, _ in dups + nests:
        toks.add(tokens(a)[0])
        toks.add(tokens(b)[0])
    t0 = time.time()
    books = cl.books(sorted(toks))
    fetch_s = time.time() - t0
    # archive trimmed snapshot
    snap_dir = os.path.join(RAW, "books")
    os.makedirs(snap_dir, exist_ok=True)
    with gzip.open(os.path.join(snap_dir, ts.replace(":", "")[:15] + ".json.gz"), "wt") as f:
        json.dump({t: dict(bids=sorted(b.get("bids", []), key=lambda x: -float(x["price"]))[:15],
                           asks=sorted(b.get("asks", []), key=lambda x: float(x["price"]))[:15],
                           min_order_size=b.get("min_order_size"), tick_size=b.get("tick_size"),
                           timestamp=b.get("timestamp")) for t, b in books.items()}, f)
    stats, opps = [], []
    for e in neg:
        rate = fee_rate(category(tag_slugs(e) | ({"weather"} if is_weather(e) else set())))
        s, o = eval_negrisk(e, books, rate, ts)
        if s:
            stats.append(s)
            opps += o
    evrate = {}
    for e, ps in lad:
        rate = fee_rate(category(tag_slugs(e)))
        for kind, hi, lo, khi, klo in ps:
            r = eval_pair(kind, hi, lo, books, rate, ts, dict(event_id=e.get("id"), slug=e.get("slug"),
                                                               key_hi=khi, key_lo=klo))
            if r:
                opps.append(r)
    for e in events:
        for m in e.get("markets") or []:
            evrate[m.get("conditionId")] = fee_rate(category(tag_slugs(e)))
    for kind, a, b, ta, tb in dups + nests:
        rate = max(evrate.get(a.get("conditionId"), 0.05), evrate.get(b.get("conditionId"), 0.05))
        r = eval_pair(kind, a, b, books, rate, ts, dict(event_hi=ta, event_lo=tb))
        if r:
            opps.append(r)
    meta = dict(ts=ts, n_events=len(events), n_negrisk=len(neg), n_ladder_events=len(lad),
                n_ladder_pairs=sum(len(p) for _, p in lad), n_dup_pairs=len(dups), n_nested_pairs=len(nests),
                n_tokens=len(toks), n_books=len(books), book_fetch_s=round(fetch_s, 1),
                n_opps=len(opps), mb_downloaded=round(cl.bytes / 1e6, 1))
    return meta, stats, opps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", type=int, default=18)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--budget-gb", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cl = Client(max_bytes=a.budget_gb * 1e9)
    for i in range(a.scans):
        t = time.time()
        try:
            meta, stats, opps = scan_once(cl, a.max_events)
        except Exception as ex:  # keep looping on transient failures
            meta, stats, opps = dict(ts=now().isoformat(), error=repr(ex)), [], []
        for name, rows in (("scans", [meta]), ("negrisk_stats", stats), ("opportunities", opps)):
            with open(os.path.join(OUT, name + ".jsonl"), "a") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
        print(json.dumps(meta), flush=True)
        if i + 1 < a.scans:
            time.sleep(max(0, a.every - (time.time() - t)))


if __name__ == "__main__":
    main()
