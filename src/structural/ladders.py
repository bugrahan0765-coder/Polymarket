"""Detect logical ordering constraints between binary markets.

Kinds:
  date      "X by March 31" implies "X by June 30"      -> P rises with date
  threshold "BTC above 100k" implies "BTC above 90k"     -> P falls with strike
            "dip to 80k" implies "dip to 90k"            -> P rises with strike
            "O/U 3.5 Over" implies "O/U 2.5 Over"        -> P falls with line
  duplicate same normalised question, different conditionId
  nested    "<cand> wins presidency" implies "<cand> wins <party> nomination"
            (near-riskless: fails if the candidate wins without that nomination)

Each constraint is a pair (hi, lo) of markets with P(hi) >= P(lo) by logic.
Violation when ask_yes(hi) < bid_yes(lo): buy YES hi + buy NO lo pays >= 1.

Only markets inside the same event (same template text) are paired for
date/threshold ladders, so resolution source and wording match.
"""
import calendar
import datetime as dt
import re

from pm_api import jl, tokens

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9

NUM = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:[kmbt]\b|%|bn\b|million\b|billion\b|trillion\b)?", re.I)
UP = ("above", "over ", "greater than", "more than", "at least", "exceed", "reach", "hit ", "higher than",
      "or more", "or higher", "+", "↑", "surpass", "top ")
DOWN = ("below", "under ", "less than", "lower than", "dip", "drop", "fall", "or less", "or lower", "↓", "sink")


def binary_kind(m):
    o = [str(x).lower() for x in (jl(m.get("outcomes"), []) or [])]
    if o == ["yes", "no"]:
        return "yesno"
    if o == ["over", "under"]:
        return "overunder"
    return None


def parse_num(s):
    m = NUM.search(s or "")
    if not m:
        return None
    t = m.group(0).lower().replace("$", "").replace(",", "").replace(" ", "")
    mult = 1.0
    for suf, k in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("bn", 1e9),
                   ("t", 1e12), ("b", 1e9), ("m", 1e6), ("k", 1e3), ("%", 1.0)):
        if t.endswith(suf):
            t, mult = t[: -len(suf)], k
            break
    try:
        return float(t) * mult
    except ValueError:
        return None


def parse_date(s, default_year):
    s = (s or "").lower().replace(",", " ")
    words = re.findall(r"[a-z]+|\d+", s)
    for i, w in enumerate(words):
        if w in MONTHS:
            mo = MONTHS[w]
            day = year = None
            for x in words[i + 1:i + 3]:
                if x.isdigit():
                    v = int(x)
                    if v > 31 and year is None:
                        year = v
                    elif day is None and v <= 31:
                        day = v
            year = year or default_year
            day = day or calendar.monthrange(year, mo)[1]
            try:
                return dt.date(year, mo, day)
            except ValueError:
                return None
    m = re.search(r"\b(20\d\d)\b", s)
    if m and ("end of" in s or s.strip() == m.group(1)):
        return dt.date(int(m.group(1)), 12, 31)
    return None


def _template(q, git):
    q = (q or "").lower()
    git = (git or "").lower().strip()
    if git and git in q:
        return q.replace(git, "{}")
    return NUM.sub("#", q)


def ladder_groups(event):
    """Yield (kind, template, [(key, market)], increasing) for one event.
    increasing=True means P(YES) must rise with key."""
    mk = [m for m in event.get("markets") or [] if binary_kind(m) and not m.get("closed")]
    if len(mk) < 2:
        return
    try:
        ey = dt.datetime.fromisoformat((event.get("endDate") or "").replace("Z", "+00:00")).year
    except ValueError:
        ey = dt.date.today().year
    groups = {}
    for m in mk:
        q = m.get("question") or ""
        git = m.get("groupItemTitle") or ""
        ql = q.lower()
        if re.search(r"\d\s*[-–]\s*\$?\d|between", git.lower()):
            continue  # range bucket, not a ladder
        if (" by " in ql or "before " in ql) and binary_kind(m) == "yesno":
            d = parse_date(git, ey) or parse_date(q.split(" by ")[-1], ey)
            if d:
                tpl = _template(q, git) if git else re.sub(r"\bby\b.*", "by {}", ql)
                groups.setdefault(("date", tpl, True), []).append((d.toordinal(), m))
                continue
        k = parse_num(git) if git else None
        if k is None:
            continue
        tpl = _template(q, git)
        txt = (git + " " + tpl).lower()
        if binary_kind(m) == "overunder" or "o/u" in tpl:
            if binary_kind(m) != "overunder":
                continue
            inc = False
        else:
            up = any(w in txt for w in UP)
            dn = any(w in txt for w in DOWN)
            if "↑" in git:
                up, dn = True, False
            if "↓" in git:
                up, dn = False, True
            if up == dn:
                continue
            inc = dn  # "dip to K": higher K is easier -> P rises with K
        groups.setdefault(("threshold", NUM.sub("#", tpl), inc), []).append((k, m))
    for (kind, tpl, inc), lst in groups.items():
        keys = [k for k, _ in lst]
        if len(lst) >= 2 and len(set(keys)) == len(keys):
            yield kind, tpl, sorted(lst, key=lambda z: z[0]), inc


def ladder_pairs(event):
    """All (kind, hi_market, lo_market, key_hi, key_lo) implied by ladders in an event."""
    for kind, tpl, lst, inc in ladder_groups(event):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (ka, a), (kb, b) = lst[i], lst[j]  # ka < kb
                hi, lo, khi, klo = (b, a, kb, ka) if inc else (a, b, ka, kb)
                yield kind, hi, lo, khi, klo


def norm_q(q):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$%.<>= ]", " ", (q or "").lower())).strip()


def duplicate_pairs(events):
    by = {}
    for e in events:
        for m in e.get("markets") or []:
            if binary_kind(m) == "yesno" and not m.get("closed") and tokens(m)[0]:
                by.setdefault(norm_q(m.get("question")), []).append((e, m))
    for q, lst in by.items():
        cids = {m.get("conditionId") for _, m in lst}
        if len(cids) < 2:
            continue
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i != j and lst[i][1].get("conditionId") != lst[j][1].get("conditionId"):
                    # equivalence: both directions are constraints hi>=lo
                    yield "duplicate", lst[i][1], lst[j][1], q, q


NOM = re.compile(r"nominee|nomination")
WIN = re.compile(r"presidential election winner|election winner|win the .*presidential election|next president")


def nested_pairs(events):
    neg = [e for e in events if (e.get("negRisk") or e.get("enableNegRisk"))]
    noms = [e for e in neg if NOM.search((e.get("title") or "").lower())]
    wins = [e for e in neg if WIN.search((e.get("title") or "").lower())]
    for en in noms:
        yn = set(re.findall(r"20\d\d", en.get("title") or ""))
        cand = {(m.get("groupItemTitle") or "").strip().lower(): m for m in en.get("markets") or []
                if not m.get("closed")}
        for ew in wins:
            if yn and not (yn & set(re.findall(r"20\d\d", ew.get("title") or ""))):
                continue
            for mw in ew.get("markets") or []:
                name = (mw.get("groupItemTitle") or "").strip().lower()
                if name and name in cand and not mw.get("closed") and name not in ("other", "another person"):
                    yield "nested", cand[name], mw, en.get("title"), ew.get("title")
