"""Summarise results/structural/live/*.jsonl written by live_scan.py.
UNTESTED on real scan output (no live data available on the author's machine);
tested on synthetic rows via selftest.py.

Outputs results/structural/live_summary.json and live_summary.md:
  * per kind: scans seen, distinct opportunities, median/max top edge, executable
    profit (sum over distinct opportunities of best observed depth), persistence
    (consecutive scans an opportunity stayed open; resolution = scan interval)
  * negRisk / weather distributions of sum(best YES ask) and sum(best YES bid)
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pm_api import RESULTS  # noqa: E402

LIVE = os.path.join(RESULTS, "live")


def load(name):
    p = os.path.join(LIVE, name + ".jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return dict(n=len(xs), p05=q(xs, .05), p25=q(xs, .25), median=q(xs, .5), p75=q(xs, .75), p95=q(xs, .95),
                min=min(xs), max=max(xs))


def persistence(opps, scan_ts):
    order = {t: i for i, t in enumerate(sorted(scan_ts))}
    runs = {}
    for o in opps:
        runs.setdefault(o["key"], set()).add(order.get(o["ts"], -1))
    lens = []
    for k, idx in runs.items():
        idx = sorted(i for i in idx if i >= 0)
        cur = best = 1
        for a, b in zip(idx, idx[1:]):
            cur = cur + 1 if b == a + 1 else 1
            best = max(best, cur)
        lens.append(best if idx else 0)
    return lens


def summarise(scans, stats, opps):
    scan_ts = [s["ts"] for s in scans if "error" not in s]
    out = dict(n_scans=len(scan_ts), n_scan_errors=sum(1 for s in scans if "error" in s),
               first=min(scan_ts) if scan_ts else None, last=max(scan_ts) if scan_ts else None)
    for label, rows in (("negrisk_all", stats), ("negrisk_weather", [s for s in stats if s.get("weather")]),
                        ("negrisk_nonweather", [s for s in stats if not s.get("weather")])):
        out[label] = dict(
            event_scans=len(rows), distinct_events=len({r["event_id"] for r in rows}),
            sum_yes_ask=dist([r.get("sum_yes_ask") for r in rows]),
            sum_yes_bid=dist([r.get("sum_yes_bid") for r in rows]),
            yes_all_edge=dist([r.get("yes_all_edge") for r in rows]),
            no_all_edge=dist([r.get("no_all_edge") for r in rows]),
            frac_sum_ask_lt_1=(sum(1 for r in rows if r.get("sum_yes_ask") is not None and r["sum_yes_ask"] < 1)
                               / max(1, sum(1 for r in rows if r.get("sum_yes_ask") is not None))),
            frac_sum_bid_gt_1=(sum(1 for r in rows if r.get("sum_yes_bid") is not None and r["sum_yes_bid"] > 1)
                               / max(1, len(rows))))
    kinds = {}
    for o in opps:
        k = o["kind"] + ("_weather" if o.get("weather") else "")
        kinds.setdefault(k, []).append(o)
    out["opportunities"] = {}
    for k, lst in sorted(kinds.items()):
        best = {}
        for o in lst:
            if o["key"] not in best or o["profit"] > best[o["key"]]["profit"]:
                best[o["key"]] = o
        ex = [o for o in best.values() if o.get("executable") and o.get("riskless", True)]
        per = persistence(lst, scan_ts)
        out["opportunities"][k] = dict(
            observations=len(lst), distinct=len(best), scans_with_any=len({o["ts"] for o in lst}),
            top_edge=dist([o["top_edge"] for o in lst]),
            distinct_executable_riskless=len(ex),
            sum_best_profit_usd=round(sum(o["profit"] for o in ex), 2),
            sum_best_profit_conservative_fee_usd=round(sum(o.get("profit_conservative_fee", 0) for o in ex), 2),
            sum_capital_usd=round(sum(o["cost"] for o in ex), 2),
            median_days_locked=q([o.get("days") for o in ex], .5),
            persistence_scans=dist(per),
            frac_seen_in_one_scan_only=(sum(1 for x in per if x <= 1) / len(per)) if per else None,
            examples=sorted(best.values(), key=lambda o: -o["profit"])[:5])
    return out


def md(out):
    L = [f"Scans: {out['n_scans']} ({out['first']} to {out['last']}), errors {out['n_scan_errors']}", ""]
    L.append("| set | event-scans | median sum YES ask | frac sum ask<1 | median sum YES bid | frac sum bid>1 |")
    L.append("|---|---|---|---|---|---|")
    for k in ("negrisk_all", "negrisk_weather", "negrisk_nonweather"):
        r = out[k]
        L.append(f"| {k} | {r['event_scans']} | {r['sum_yes_ask'].get('median')} | {r['frac_sum_ask_lt_1']:.3f} | "
                 f"{r['sum_yes_bid'].get('median')} | {r['frac_sum_bid_gt_1']:.3f} |")
    L += ["", "| kind | obs | distinct | exec+riskless | profit $ (base fee) | profit $ (cons. fee) | capital $ | "
              "median days | median persistence (scans) | one-scan-only |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["opportunities"].items():
        L.append(f"| {k} | {r['observations']} | {r['distinct']} | {r['distinct_executable_riskless']} | "
                 f"{r['sum_best_profit_usd']} | {r['sum_best_profit_conservative_fee_usd']} | {r['sum_capital_usd']} | "
                 f"{r['median_days_locked']} | {r['persistence_scans'].get('median')} | {r['frac_seen_in_one_scan_only']} |")
    return "\n".join(L) + "\n"


def main():
    out = summarise(load("scans"), load("negrisk_stats"), load("opportunities"))
    json.dump(out, open(os.path.join(RESULTS, "live_summary.json"), "w"), indent=1, default=str)
    open(os.path.join(RESULTS, "live_summary.md"), "w").write(md(out))
    print(md(out))


if __name__ == "__main__":
    main()
