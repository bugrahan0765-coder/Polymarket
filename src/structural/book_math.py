"""Pure functions: fees, walking order books, multi-leg basket depth.

Fee model (Polymarket taker fees, Fee Structure V2 effective 2026-03-30; makers pay 0):
    fee per share = rate * p * (1 - p)        (peak at p = 0.5)
Category rates (third-party summaries of docs.polymarket.com; the docs themselves
could not be fetched from the author's machine; check them before trading):
    crypto 0.07, sports 0.03 (0.05 from Jul 2026), politics/finance/tech/mentions 0.04,
    economics/culture/weather/other 0.05, geopolitics/world events 0.
Some sources write the formula as C*p*rate*(p(1-p))^exponent with category exponents.
`fee_per_share(..., conservative=True)` takes the max of the base formula and a
tail-heavy variant (rate*0.25*sqrt(4p(1-p))), which charges more at extreme prices.
That variant matters most for baskets that buy many 1-5c legs.
"""
import datetime as dt
import math

RATES = {
    "crypto": 0.07, "sports": 0.05, "politics": 0.04, "finance": 0.04, "tech": 0.04,
    "mentions": 0.04, "economics": 0.05, "culture": 0.05, "weather": 0.05, "other": 0.05,
    "geopolitics": 0.0,
}
FEE_START = dt.datetime(2026, 3, 30, tzinfo=dt.timezone.utc)
SPORTS_HIKE = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)

_KEYS = [  # (category, tag-slug substrings), checked in order
    ("geopolitics", ("geopolitic", "world", "ukraine", "israel", "middle-east", "iran", "china-taiwan")),
    ("crypto", ("crypto", "bitcoin", "ethereum", "solana", "xrp")),
    ("weather", ("weather", "temperature", "climate", "hurricane")),
    ("sports", ("sports", "nba", "nfl", "mlb", "nhl", "soccer", "football", "tennis", "esports",
                "ufc", "golf", "cricket", "f1", "epl", "champions-league")),
    ("finance", ("finance", "stocks", "fed", "earnings", "commodities", "ipo")),
    ("economics", ("economy", "economics", "inflation", "cpi", "gdp", "jobs")),
    ("politics", ("politics", "elections", "election", "trump", "congress", "senate", "us-politics")),
    ("tech", ("tech", "ai", "openai", "science")),
    ("mentions", ("mention",)),
    ("culture", ("culture", "pop-culture", "movies", "music", "awards", "celebrities", "tweets")),
]


def category(tag_slugs):
    s = " ".join(sorted(tag_slugs))
    for cat, keys in _KEYS:
        if any(k in s for k in keys):
            return cat
    return "other"


def fee_rate(cat, when=None):
    when = when or dt.datetime.now(dt.timezone.utc)
    if when < FEE_START and cat not in ("crypto", "sports"):
        return 0.0
    if cat == "sports" and when < SPORTS_HIKE:
        return 0.03
    return RATES.get(cat, 0.05)


def fee_per_share(p, rate, conservative=False):
    base = rate * p * (1 - p)
    if not conservative:
        return base
    return max(base, rate * 0.25 * math.sqrt(max(0.0, 4 * p * (1 - p))))


def no_asks_from_yes_bids(bids):
    """On Polymarket's CLOB the YES and NO books mirror each other:
    a YES bid at p for s shares is a NO ask at 1-p for s shares."""
    return [(round(1 - p, 6), s) for p, s in bids]


def basket_depth(legs, payout, rate, conservative=False, max_sets=1e9):
    """Buy one share of every leg per 'set'; each set pays `payout` for sure.
    legs: list of ask ladders [(price, size)] ascending.
    Walks all ladders together while the marginal cost of one more set
    (sum of leg prices + taker fees) is below payout.
    Returns dict: top_edge (payout - marginal cost of first set), sets, cost,
    profit, max_leg_levels, or top_edge=None if a leg has no asks."""
    if not legs or any(not l for l in legs):
        return dict(top_edge=None, sets=0.0, cost=0.0, profit=0.0)
    idx = [0] * len(legs)
    rem = [l[0][1] for l in legs]
    sets = cost = 0.0
    top = None
    while sets < max_sets:
        if any(i >= len(l) for i, l in zip(idx, legs)):
            break
        prices = [l[i][0] for i, l in zip(idx, legs)]
        marg = sum(p + fee_per_share(p, rate, conservative) for p in prices)
        if top is None:
            top = payout - marg
        if marg >= payout:
            break
        step = min(min(rem), max_sets - sets)
        sets += step
        cost += step * marg
        for k in range(len(legs)):
            rem[k] -= step
            if rem[k] <= 1e-9:
                idx[k] += 1
                if idx[k] < len(legs[k]):
                    rem[k] = legs[k][idx[k]][1]
    return dict(top_edge=top, sets=sets, cost=cost, profit=sets * payout - cost,
                max_leg_levels=max(idx) + 1)


def pair_depth(ask_hi, bids_lo, rate, conservative=False):
    """Ladder violation: event H must have P(H) >= P(L) (L implies H).
    Buy YES(H) on its asks and NO(L) (i.e. hit YES(L) bids); each pair pays >= 1.
    Returns the same dict as basket_depth."""
    return basket_depth([ask_hi, no_asks_from_yes_bids(bids_lo)], 1.0, rate, conservative)


def annualised(profit, cost, days):
    if cost <= 0 or days is None or days <= 0:
        return None
    # simple (non-compounded) annualisation; compounding tiny-lockup arbs is meaningless
    # because the same opportunity cannot be repeated back-to-back.
    return (profit / cost) * 365.0 / max(days, 1.0 / 24)
