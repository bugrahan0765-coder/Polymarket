"""Realistic yearly return for a ~$100 account from structural arbs (scenario model).

Annual profit = min(demand-limited, capital-limited) where
  demand-limited  = opps_per_day * capture_prob * sets_per_opp * edge_per_set * 365
  capital-limited = (bankroll / capital_per_set) / max(lockup_days, 1/24) * edge_per_set * 365
  sets_per_opp    = min(depth_sets, bankroll / capital_per_set), and >= min order size (5 sh) or 0
Inputs come from results/structural/weather_offline.json (offline, mid-price based),
and, when present, results/structural/live_summary.json (live books; UNTESTED here).

  python src/structural/returns_model.py
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pm_api import RESULTS  # noqa: E402

BANK = 100.0
MIN_SH = 5


def annual(opps_day, capture, depth, edge, cap_set, lockup_days, bank=BANK):
    sets = min(depth, bank / cap_set)
    if sets < MIN_SH or edge <= 0:
        return 0.0
    demand = opps_day * capture * sets * edge * 365
    capital = (bank / cap_set) / max(lockup_days, 1 / 24) * edge * 365
    return min(demand, capital)


def main():
    w = json.load(open(os.path.join(RESULTS, "weather_offline.json")))
    days_span = 367
    ev_day = w["usable_events"] / days_span
    rows = []
    # weather buy-all-NO basket: n~11 legs, capital per set ~ n - sum(bids) ~ 10 USDC
    for r in w["basket_edges_at_mid_plus_half_spread"]:
        if not r["no_basket_frac_pos"]:
            continue
        edge = r["no_basket_median_edge_if_pos"]
        for capture, depth, lock in itertools.product((1.0, 0.25, 0.05), (5, 20, 100), (1.5, 0.0)):
            rows.append(dict(strategy="weather NO basket", half_spread=r["half_spread_per_leg"],
                             opps_per_day=round(ev_day * r["no_basket_frac_pos"], 2), capture=capture,
                             depth_sets=depth, edge_per_set=edge, capital_per_set=10.0,
                             lockup_days=lock if lock else "0 (convert NO set -> USDC on-chain)",
                             annual_usd=round(annual(ev_day * r["no_basket_frac_pos"], capture, depth, edge, 10.0,
                                                     lock), 1)))
    # generic ladder / duplicate pair: capital ~ $1 per pair, locked until resolution
    for edge, lock, opps in itertools.product((0.01, 0.02, 0.05), (7, 30, 120), (0.1, 1.0)):
        rows.append(dict(strategy="ladder/duplicate pair", half_spread=None, opps_per_day=opps, capture=0.25,
                         depth_sets=50, edge_per_set=edge, capital_per_set=1.0, lockup_days=lock,
                         annual_usd=round(annual(opps, 0.25, 50, edge, 1.0, lock), 1)))
    out = dict(bankroll=BANK, assumptions=__doc__.strip().splitlines()[2:6], weather_events_per_day=round(ev_day, 1),
               rows=rows)
    lp = os.path.join(RESULTS, "live_summary.json")
    if os.path.exists(lp):  # UNTESTED: plug in measured live numbers
        live = json.load(open(lp))
        hours = max(1e-9, live.get("n_scans", 0) / 6.0)
        meas = {}
        for k, v in live.get("opportunities", {}).items():
            if v["distinct_executable_riskless"] and v["sum_capital_usd"] > 0:
                per_day = v["distinct_executable_riskless"] / hours * 24
                ret = v["sum_best_profit_usd"] / v["sum_capital_usd"]
                lock = v["median_days_locked"] or 1.0
                meas[k] = dict(distinct_per_day=round(per_day, 2), return_on_capital=round(ret, 5),
                               median_lockup_days=lock,
                               annual_usd_capture_25pct=round(min(per_day * 0.25 * v["sum_best_profit_usd"]
                                                                  / v["distinct_executable_riskless"] * 365,
                                                                  BANK * ret * 365 / max(lock, 1 / 24)), 1))
        out["live_measured"] = meas
    json.dump(out, open(os.path.join(RESULTS, "returns_model.json"), "w"), indent=1, default=str)
    hdr = "| strategy | half-spread | opps/day | capture | depth | edge/set $ | cap/set $ | lockup d | $/yr on $100 |"
    print(hdr + "\n|" + "---|" * 9)
    for r in rows:
        print("| " + " | ".join(str(r[k]) for k in ("strategy", "half_spread", "opps_per_day", "capture", "depth_sets",
                                                     "edge_per_set", "capital_per_set", "lockup_days",
                                                     "annual_usd")) + " |")


if __name__ == "__main__":
    main()
