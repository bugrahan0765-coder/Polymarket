"""Historical ladder / nested / duplicate violations. UNTESTED against the real
API (Polymarket blocked on the author's machine); run from a cloud box.

Step 1 (indicative): hourly prices-history for both legs of every constraint pair
  in open events + events closed in the last --days days. Count hours where
  p(lo) - p(hi) > threshold. prices-history is last-trade/mid, NOT executable.
Step 2 (execution check): for pairs with violating hours, pull data-api TAKER
  trades for both markets and look for 5-minute windows where a taker BOUGHT
  YES(hi) at x and a taker SOLD YES(lo) at y with y - x > fees. Those trades
  prove the quotes existed at that moment; their sizes bound executable depth.

  python src/structural/history_ladders.py --days 60 --max-pairs 400
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book_math import category, fee_per_share, fee_rate  # noqa: E402
from ladders import duplicate_pairs, ladder_pairs, nested_pairs  # noqa: E402
from pm_api import GAMMA, RESULTS, Client, tag_slugs, tokens  # noqa: E402


def closed_events(cl, days):
    out, end = [], dt.datetime.now(dt.timezone.utc)
    for d in range(days):
        a = (end - dt.timedelta(days=d + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        b = (end - dt.timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ")
        off = 0
        while True:
            r = cl.get(GAMMA + "/events", dict(closed="true", limit=500, offset=off, end_date_min=a, end_date_max=b))
            if not r:
                break
            out += r
            off += len(r)
            if len(r) < 500 or off > 20000:
                break
    return out


def series(cl, tok, t0, t1):
    h = cl.prices_history(tok, t0, t1, fidelity=60)
    return {int(x["t"]) // 3600: float(x["p"]) for x in h}


def trade_check(cl, hi, lo, rate, window=300):
    th, tl = tokens(hi)[0], tokens(lo)[0]
    buys = [t for t in cl.taker_trades(hi["conditionId"]) if str(t.get("asset")) == th and t.get("side") == "BUY"]
    sells = [t for t in cl.taker_trades(lo["conditionId"]) if str(t.get("asset")) == tl and t.get("side") == "SELL"]
    sells.sort(key=lambda t: t["timestamp"])
    hits = []
    j0 = 0
    for b in sorted(buys, key=lambda t: t["timestamp"]):
        x = float(b["price"])
        while j0 < len(sells) and sells[j0]["timestamp"] < b["timestamp"] - window:
            j0 += 1
        for s in sells[j0:]:
            if s["timestamp"] > b["timestamp"] + window:
                break
            y = float(s["price"])
            edge = y - x - fee_per_share(x, rate) - fee_per_share(1 - y, rate)
            if edge > 0:
                hits.append(dict(ts=b["timestamp"], buy_hi=x, sell_lo=y, edge=edge,
                                 size=min(float(b["size"]), float(s["size"]))))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--max-pairs", type=int, default=400)
    ap.add_argument("--thresholds", default="0.01,0.02,0.05")
    a = ap.parse_args()
    cl = Client()
    events = cl.open_events() + closed_events(cl, a.days)
    pairs = []
    for e in events:
        if e.get("negRisk") or e.get("enableNegRisk"):
            continue
        rate = fee_rate(category(tag_slugs(e)))
        for kind, hi, lo, khi, klo in ladder_pairs(e):
            pairs.append((kind, hi, lo, rate, e.get("slug")))
    for kind, x, y, tx, ty in list(duplicate_pairs(events)) + list(nested_pairs(events)):
        pairs.append((kind, x, y, 0.05, f"{tx} | {ty}"))
    # prioritise adjacent-looking pairs: keep at most max-pairs, spread over kinds
    pairs = pairs[: a.max_pairs]
    ths = [float(t) for t in a.thresholds.split(",")]
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    rows = []
    for kind, hi, lo, rate, src in pairs:
        t0 = now - a.days * 86400
        sh, sl = series(cl, tokens(hi)[0], t0, now), series(cl, tokens(lo)[0], t0, now)
        common = sorted(set(sh) & set(sl))
        if not common:
            continue
        gaps = [sl[h] - sh[h] for h in common]
        row = dict(kind=kind, src=src, hi=hi.get("question"), lo=lo.get("question"), hours=len(common),
                   max_gap=max(gaps), **{f"hours_gap_gt_{t}": sum(1 for g in gaps if g > t) for t in ths})
        if row[f"hours_gap_gt_{ths[0]}"] > 0:
            hits = trade_check(cl, hi, lo, rate)
            row.update(trade_hits=len(hits), trade_hit_size=sum(h["size"] for h in hits),
                       trade_hit_profit=sum(h["size"] * h["edge"] for h in hits), trade_examples=hits[:5])
        rows.append(row)
    os.makedirs(RESULTS, exist_ok=True)
    tot = dict(n_pairs=len(rows), by_kind={})
    for r in rows:
        k = tot["by_kind"].setdefault(r["kind"], dict(pairs=0, pair_hours=0, trade_hits=0, trade_hit_profit=0.0,
                                                       **{f"pairs_gap_gt_{t}": 0 for t in ths}))
        k["pairs"] += 1
        k["pair_hours"] += r["hours"]
        k["trade_hits"] += r.get("trade_hits", 0)
        k["trade_hit_profit"] += r.get("trade_hit_profit", 0.0)
        for t in ths:
            k[f"pairs_gap_gt_{t}"] += r[f"hours_gap_gt_{t}"] > 0
    json.dump(dict(summary=tot, rows=rows), open(os.path.join(RESULTS, "history_ladders.json"), "w"),
              indent=1, default=str)
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()
