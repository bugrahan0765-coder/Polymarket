"""Offline self-test of the scanner logic on synthetic events/books (no network).
  python src/structural/selftest.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book_math import basket_depth, fee_per_share, no_asks_from_yes_bids, pair_depth  # noqa: E402
from ladders import duplicate_pairs, ladder_pairs, nested_pairs, parse_date, parse_num  # noqa: E402
import live_scan  # noqa: E402
import summarise_live  # noqa: E402


def mk(q, git, cid, tok, outcomes=("Yes", "No"), **kw):
    m = dict(question=q, groupItemTitle=git, conditionId=cid, clobTokenIds=json.dumps([tok, tok + "n"]),
             outcomes=json.dumps(list(outcomes)), active=True, closed=False, endDate="2030-01-01T00:00:00Z")
    m.update(kw)
    return m


def book(bids, asks):
    return dict(bids=[dict(price=str(p), size=str(s)) for p, s in bids],
                asks=[dict(price=str(p), size=str(s)) for p, s in asks], min_order_size="5")


def approx(a, b, eps=1e-9):
    assert abs(a - b) < eps, (a, b)


def main():
    # fees
    approx(fee_per_share(0.5, 0.05), 0.0125)
    assert fee_per_share(0.01, 0.05, True) > fee_per_share(0.01, 0.05)
    # basket: 3 legs, asks 0.30/0.30/0.30 (10 sh) then 0.40 -> sum 0.90 then 1.0+
    legs = [[(0.30, 10), (0.40, 100)]] * 3
    r = basket_depth(legs, 1.0, 0.0)
    approx(r["top_edge"], 0.10)
    approx(r["sets"], 10)
    approx(r["profit"], 1.0)
    r = basket_depth(legs, 1.0, 0.05)
    approx(r["top_edge"], 0.10 - 3 * 0.05 * 0.3 * 0.7)
    assert basket_depth([[(0.5, 1)], []], 1, 0)["top_edge"] is None
    # NO mirror and pair
    assert no_asks_from_yes_bids([(0.6, 5)]) == [(0.4, 5)]
    r = pair_depth([(0.40, 20), (0.55, 50)], [(0.50, 30), (0.45, 10)], 0.0)
    approx(r["top_edge"], 0.10)
    approx(r["sets"], 20)  # second ask level 0.55 vs bid 0.50 -> stop
    # parsers
    assert parse_num("$100,000") == 100000 and parse_num("120k") == 120000 and parse_num("2.5%") == 2.5
    assert parse_num("↑ 1.2b") == 1.2e9
    assert str(parse_date("March 31", 2026)) == "2026-03-31"
    assert str(parse_date("June", 2026)) == "2026-06-30"
    assert str(parse_date("December 31, 2027", 2026)) == "2027-12-31"
    # ladders
    ev_date = dict(id="d", endDate="2026-12-31T00:00:00Z", markets=[
        mk("Will X happen by March 31?", "March 31", "c1", "t1"),
        mk("Will X happen by June 30?", "June 30", "c2", "t2"),
        mk("Will X happen by December 31?", "December 31", "c3", "t3")])
    p = list(ladder_pairs(ev_date))
    assert len(p) == 3 and all(k == "date" for k, *_ in p)
    hi_lo = {(h["conditionId"], l["conditionId"]) for _, h, l, _, _ in p}
    assert hi_lo == {("c2", "c1"), ("c3", "c1"), ("c3", "c2")}, hi_lo
    ev_th = dict(id="t", endDate="2026-12-31T00:00:00Z", markets=[
        mk("Will Bitcoin reach $120,000 in October?", "↑ 120,000", "u1", "a1"),
        mk("Will Bitcoin reach $130,000 in October?", "↑ 130,000", "u2", "a2"),
        mk("Will Bitcoin dip to $90,000 in October?", "↓ 90,000", "d1", "b1"),
        mk("Will Bitcoin dip to $80,000 in October?", "↓ 80,000", "d2", "b2"),
        mk("Will the high be between 90-95k?", "90-95k", "r1", "r1")])
    hi_lo = {(h["conditionId"], l["conditionId"]) for _, h, l, _, _ in ladder_pairs(ev_th)}
    assert hi_lo == {("u1", "u2"), ("d1", "d2")}, hi_lo
    ev_ou = dict(id="o", endDate="2026-12-31T00:00:00Z", markets=[
        mk("A vs B: O/U 2.5", "O/U 2.5", "o1", "q1", outcomes=("Over", "Under")),
        mk("A vs B: O/U 3.5", "O/U 3.5", "o2", "q2", outcomes=("Over", "Under"))])
    hi_lo = {(h["conditionId"], l["conditionId"]) for _, h, l, _, _ in ladder_pairs(ev_ou)}
    assert hi_lo == {("o1", "o2")}, hi_lo
    # duplicates + nested
    e1 = dict(id="e1", title="Fed", markets=[mk("Will the Fed cut rates in December?", "", "x1", "f1")])
    e2 = dict(id="e2", title="Fed2", markets=[mk("Will the Fed cut rates in December?", "", "x2", "f2")])
    assert len(list(duplicate_pairs([e1, e2]))) == 2
    nom = dict(id="n", title="Republican Presidential Nominee 2028", negRisk=True,
               markets=[mk("Will JD Vance be the nominee?", "JD Vance", "n1", "nv")])
    win = dict(id="w", title="Presidential Election Winner 2028", negRisk=True,
               markets=[mk("Will JD Vance win?", "JD Vance", "w1", "wv")])
    np_ = list(nested_pairs([nom, win]))
    assert len(np_) == 1 and np_[0][1]["conditionId"] == "n1"
    # negRisk evaluation (3 outcomes, sum asks 0.93, sum bids 0.85)
    ev = dict(id="nr", slug="nr", title="Highest temperature in NYC on X?", negRisk=True,
              endDate="2030-01-01T00:00:00Z", markets=[
                  mk("a", "a", "m1", "k1"), mk("b", "b", "m2", "k2"), mk("c", "c", "m3", "k3")])
    books = {"k1": book([(0.28, 50)], [(0.31, 20)]), "k2": book([(0.28, 50)], [(0.31, 20)]),
             "k3": book([(0.29, 50)], [(0.31, 20)])}
    stat, opps = live_scan.eval_negrisk(ev, books, 0.05, "T0")
    approx(stat["sum_yes_ask"], 0.93)
    assert stat["weather"] and len(opps) == 1 and opps[0]["kind"] == "negrisk_yes_all"
    approx(opps[0]["sets"], 20)
    # NO basket: bids sum 1.06 > 1
    books2 = {"k1": book([(0.36, 10)], [(0.40, 20)]), "k2": book([(0.35, 10)], [(0.40, 20)]),
              "k3": book([(0.35, 10)], [(0.40, 20)])}
    stat, opps = live_scan.eval_negrisk(ev, books2, 0.0, "T1")
    o = [x for x in opps if x["kind"] == "negrisk_no_all"][0]
    approx(o["top_edge"], 0.06)
    approx(o["payout"], 2)
    # summariser
    out = summarise_live.summarise([dict(ts="T0"), dict(ts="T1")], [stat], opps)
    assert out["opportunities"]["negrisk_no_all_weather"]["distinct"] == 1
    print("selftest OK")


if __name__ == "__main__":
    main()
