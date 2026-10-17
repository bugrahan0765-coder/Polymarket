"""OFFLINE weather check (runs on local data, no Polymarket access needed).

Uses data/decision_prices.csv.gz: for every "Highest temperature in <city> on
<date>" event, the last prices-history price of each bucket at the decision time
(24 h before local midnight of the target day). Prices-history is last-trade/mid,
NOT the ask, so this gives an upper bound on how often a sum-of-asks < 1 basket
could exist, and how big the overround is that a buy-all-NO basket would need
to beat.

For each event with every bucket priced (fresh price, age <= --max-age s):
  S = sum of bucket prices
  buy-all-YES edge at mid  = 1 - S - sum(fees)           (needs S < 1)
  buy-all-NO  edge at mid  = S - 1 - sum(fees on NO legs) (payout n-1, needs S > 1)
with an extra per-leg half-spread h in {0, 0.005, 0.01} to move from mid to ask
(YES basket) or bid (NO basket). Fee rate 0.05 (weather) from 2026-03-30, 0 before.

  python src/structural/weather_offline.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book_math import fee_per_share  # noqa: E402
from pm_api import REPO, RESULTS  # noqa: E402

FEE_START = pd.Timestamp("2026-03-30")


def unit(bucket):
    b = str(bucket)
    i = max(b.rfind("F"), b.rfind("C"))
    return "F" if b.rfind("F") > b.rfind("C") else ("C" if i >= 0 else "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age", type=float, default=3600)
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(REPO, "data", "decision_prices.csv.gz"))
    d["target_date"] = pd.to_datetime(d["target_date"])
    g = d.groupby("event_id")
    ev = pd.DataFrame(dict(
        city=g["city"].first(), date=g["target_date"].first(), n=g.size(),
        n_priced=g["price_last_le_dec"].count(),
        max_age=g["price_age_s"].max(), S=g["price_last_le_dec"].sum(min_count=1),
        n_yes=g["outcome_yes"].sum(), unit=g["bucket"].first().map(unit)))
    total_events = len(ev)
    ok = ev[(ev.n == ev.n_priced) & (ev.max_age <= a.max_age) & (ev.n_yes == 1)].copy()
    rate = np.where(ok.date >= FEE_START, 0.05, 0.0)
    ok["rate"] = rate
    # per-event fee sums at mid prices
    dd = d[d.event_id.isin(ok.index)].copy()
    dd["rate"] = np.where(dd.target_date >= FEE_START, 0.05, 0.0)
    dd["fee_yes"] = [fee_per_share(p, r) for p, r in zip(dd.price_last_le_dec, dd.rate)]
    dd["fee_yes_c"] = [fee_per_share(p, r, True) for p, r in zip(dd.price_last_le_dec, dd.rate)]
    ok["fees"] = dd.groupby("event_id")["fee_yes"].sum()  # symmetric: NO leg fee identical at 1-p
    ok["fees_c"] = dd.groupby("event_id")["fee_yes_c"].sum()
    res = dict(source="data/decision_prices.csv.gz", total_events=total_events, usable_events=len(ok),
               filter=f"all buckets priced, max price age <= {a.max_age}s, exactly one winning bucket",
               buckets_per_event=ok.n.describe().round(2).to_dict(),
               S_distribution={k: round(float(v), 4) for k, v in
                               ok.S.quantile([.01, .05, .1, .25, .5, .75, .9, .95, .99]).items()},
               frac_S_lt={str(t): round(float((ok.S < t).mean()), 4) for t in (0.95, 0.97, 0.99, 1.0)},
               frac_S_gt={str(t): round(float((ok.S > t).mean()), 4) for t in (1.0, 1.02, 1.05, 1.1)})
    rows = []
    for h in (0.0, 0.005, 0.01):
        yes_edge = 1 - ok.S - ok.n * h - ok.fees
        no_edge = ok.S - 1 - ok.n * h - ok.fees
        no_edge_c = ok.S - 1 - ok.n * h - ok.fees_c
        rows.append(dict(half_spread_per_leg=h,
                         yes_basket_frac_pos=round(float((yes_edge > 0).mean()), 4),
                         yes_basket_mean_edge_if_pos=round(float(yes_edge[yes_edge > 0].mean()), 4) if (yes_edge > 0).any() else None,
                         no_basket_frac_pos=round(float((no_edge > 0).mean()), 4),
                         no_basket_median_edge_if_pos=round(float(no_edge[no_edge > 0].median()), 4) if (no_edge > 0).any() else None,
                         no_basket_frac_pos_conservative_fee=round(float((no_edge_c > 0).mean()), 4),
                         no_basket_return_on_capital_median=(round(float((no_edge[no_edge > 0] / (ok.n[no_edge > 0] - 1)).median()), 5)
                                                             if (no_edge > 0).any() else None)))
    res["basket_edges_at_mid_plus_half_spread"] = rows
    ok["period"] = np.where(ok.date >= FEE_START, "fees (>=2026-03-30)", "no fees (<2026-03-30)")
    res["by_period"] = {p: dict(events=int(len(x)), S_median=round(float(x.S.median()), 4),
                                frac_S_lt_1=round(float((x.S < 1).mean()), 4),
                                frac_S_gt_1_05=round(float((x.S > 1.05).mean()), 4),
                                median_n=float(x.n.median()), median_fee_sum=round(float(x.fees.median()), 4))
                        for p, x in ok.groupby("period")}
    ok["month"] = ok.date.dt.to_period("M").astype(str)
    res["by_month"] = {m: dict(events=int(len(x)), S_median=round(float(x.S.median()), 4),
                               frac_S_lt_1=round(float((x.S < 1).mean()), 4))
                       for m, x in ok.groupby("month")}
    res["by_unit"] = {u: dict(events=int(len(x)), S_median=round(float(x.S.median()), 4),
                              median_n=float(x.n.median())) for u, x in ok.groupby("unit")}
    # tails: how much of S sits in buckets priced <= 2c (these are the legs whose ask is
    # typically several ticks above the mid -> overround is mostly tail spread)
    dd["is_tail"] = dd.price_last_le_dec <= 0.02
    tail = dd[dd.is_tail].groupby("event_id").agg(n_tail=("is_tail", "size"), S_tail=("price_last_le_dec", "sum"))
    res["tail_buckets_le_2c"] = dict(median_n_tail=float(tail.n_tail.median()), median_S_tail=round(float(tail.S_tail.median()), 4))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "weather_offline.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
