"""Deterministic question understanding: profile extraction, routing and tool planning.

Used (a) as the full planner in offline mode (LLM_PROVIDER=none) and (b) as a fallback when the LLM's
routing output can't be parsed. Keeping a no-LLM path also makes the eval cheap and reproducible."""
from __future__ import annotations

import re

from placement_agent.ingest.build import norm_company
from placement_agent.tools import company_names, stats_company_names

ROUTES = ("company", "compare", "structured", "eligibility", "prep_plan", "stats", "semantic")

BRANCH_WORDS = {
    "computer science": "cse", "cse": "cse", "cs": "cse", "ece": "ece", "electronics": "ece", "eee": "eee",
    "electrical": "eee", "mechanical": "mec", "mech": "mec", "civil": "civ", "chemical": "che",
    "ceramic": "cer", "metallurgy": "met", "metallurgical": "met", "mining": "min", "pharma": "phe",
    "pharmaceutical": "phe", "biochemical": "bce", "biomedical": "bme", "materials": "mst", "physics": "phy",
    "chemistry": "chy", "mathematics": "mat", "mnc": "mat", "m&c": "mat", "architecture": "apd",
}
CATEGORY_WORDS = [
    (r"\bsde\b|software|\bswe\b|developer|coding role", "sde"),
    (r"data scien|\bml\b|machine learning|\bai\b|\bds\b", "data"),
    (r"\bquant", "quant"),
    (r"\banalyst|consult", "analyst"),
    (r"\bproduct\b|\bapm\b|\bpm\b", "product"),
    (r"hardware|vlsi|\bcore\b|embedded|asic", "hardware"),
]
CITY_RE = (r"bangalore|bengaluru|hyderabad|pune|gurgaon|gurugram|noida|mumbai|delhi|chennai|kolkata|dubai|"
           r"ahmedabad|jaipur|remote")
# words that are also company names in the data but usually mean something else in a question
AMBIGUOUS = {"target", "orbit", "spectrum", "switch", "nation", "applied", "house", "edge", "visa"}


def extract_profile(text: str) -> dict:
    t = text.lower()
    prof = {}
    m = re.search(r"(?:cgpa|cpi|gpa|pointer)\s*(?:of|is|=|:|-)?\s*(\d{1,2}(?:\.\d+)?)", t) or \
        re.search(r"(\d{1,2}\.\d+)\s*(?:cgpa|cpi|gpa|pointer)", t)
    if m and 0 < float(m.group(1)) <= 10:
        prof["cgpa"] = float(m.group(1))
    for word, code in sorted(BRANCH_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", t):
            prof["branch"] = code
            break
    m = re.search(r"\b(b\.?tech|idd|imd|dual degree|m\.?tech|phd)\b", t)
    if m:
        prof["course"] = {"btech": "btech", "b.tech": "btech", "idd": "idd", "imd": "idd", "dual degree": "idd",
                          "mtech": "mtech", "m.tech": "mtech", "phd": "phd"}[m.group(1)]
    interests = sorted({cat for pat, cat in CATEGORY_WORDS if re.search(pat, t)})
    if interests and re.search(r"\b(prefer|interested|want|looking for|target|focus)\b", t):
        prof["preferred_roles"] = interests
    m = re.search(rf"\b({CITY_RE})\b", t)
    if m and re.search(r"\b(prefer|interested|want|looking for|location|based|relocat|city)\b", t):
        prof["preferred_location"] = {"bengaluru": "bangalore", "gurugram": "gurgaon"}.get(m.group(1), m.group(1))
    m = re.search(r"(?:strong|good|comfortable) (?:in|at|with) ([a-z+#/ ,]+?)\s*(?:[.?!;—–]|$| and (?:i|my|which|what)| - )", t)
    if m:
        prof["strengths"] = m.group(1).strip()
    return prof


def find_companies(text: str) -> list[str]:
    t = " " + re.sub(r"[^a-z0-9& ]", " ", text.lower()) + " "
    t_compact = t.replace(" ", "")
    found = []
    for name in company_names():
        key = norm_company(name)
        words = [w for w in re.sub(r"[^a-z0-9& ]", " ", name.lower()).split()
                 if w not in ("pvt", "ltd", "limited", "private", "india", "technologies", "solutions", "group")]
        if not key:
            continue
        full = " ".join(words)
        hit = (f" {full} " in t) or (len(key) >= 6 and key in t_compact and len(words) > 1)
        if not hit and words and len(words[0]) >= 5 and words[0] not in AMBIGUOUS:
            hit = f" {words[0]} " in t
        full_hit = f" {full} " in t
        if hit and words and words[0] in AMBIGUOUS and not re.search(rf"\b{re.escape(full)}\b", text.lower()):
            hit = False
        if hit and not any(norm_company(f[0]) == key for f in found):
            found.append((name, full_hit, words[0] if words else ""))
    # a first-word-only hit ("capital") that is just part of another full-name hit ("squarepoint capital") is dropped
    fulls = [" ".join(re.sub(r"[^a-z0-9& ]", " ", n.lower()).split()) for n, fh, _ in found if fh]
    found = [n for n, fh, w in found if fh or not any(f" {w} " in f" {f} " for f in fulls)]
    # drop names that are substrings of another found name (e.g. "Acme" vs "Acme Labs")
    return [f for f in found if not any(f != g and norm_company(f) in norm_company(g) for g in found)]


def find_stats_only_companies(text: str, portal_found: list[str]) -> list[str]:
    """Companies that appear only in the statistics sheet (e.g. internship/PPO-only recruiters)."""
    t = " " + re.sub(r"[^a-z0-9& ]", " ", text.lower()) + " "
    out = []
    for name in stats_company_names():
        base = re.sub(r"\(.*?\)", "", name).strip()
        words = re.sub(r"[^a-z0-9& ]", " ", base.lower()).split()
        if not words or len("".join(words)) < 4 or words[0] in AMBIGUOUS:
            continue
        if f" {' '.join(words)} " in t and not any(norm_company(base) in norm_company(p) or norm_company(p) in norm_company(base)
                                                    for p in portal_found + out):
            if not company_names_match(base):
                out.append(base)
    return out


def company_names_match(name: str) -> bool:
    key = norm_company(name)
    return any(key == norm_company(c) or (len(key) >= 4 and norm_company(c).startswith(key)) for c in company_names())


def _money(val: str, unit: str | None) -> float:
    v = float(val)
    return v * 100 if unit and unit.startswith("cr") else v


def parse_filters(text: str) -> dict:
    t = text.lower()
    f = {}
    money_ctx = bool(re.search(r"ctc|lpa|salary|pay|package|stipend|compensation|\bcr\b|crore|lakh", t))
    unit = r"(lpa|l\b|lakhs?|cr\b|crore)"
    m = re.search(r"(?:above|over|more than|greater than|>=?|at least|minimum|min\.?|atleast)\s*(?:rs\.?|inr|₹)?\s*"
                  rf"(\d+(?:\.\d+)?)\s*{unit}?(?!\s*(?:years?|yrs?|months?|%|cgpa))", t) or \
        re.search(rf"(\d+(?:\.\d+)?)\s*\+\s*{unit}", t)
    if m and (m.group(2) or money_ctx):
        f["min_ctc_lpa"] = _money(m.group(1), m.group(2))
    m = re.search(r"(?:below|under|less than|<=?|at most|maximum|max\.?)\s*(?:rs\.?|inr|₹)?\s*"
                  rf"(\d+(?:\.\d+)?)\s*{unit}?(?!\s*(?:years?|yrs?|months?|%|cgpa))", t)
    if m and (m.group(2) or money_ctx):
        f["max_ctc_lpa"] = _money(m.group(1), m.group(2))
    for pat, cat in CATEGORY_WORDS:
        if re.search(pat, t):
            f["role_category"] = cat
            break
    m = re.search(r"\bday\s*(\d)\b", t)
    if m:
        f["drive_day"] = int(m.group(1))
    m = re.search(rf"\b({CITY_RE})\b", t)
    if m:
        f["location"] = {"bengaluru": "bangalore", "gurugram": "gurgaon"}.get(m.group(1), m.group(1))
    if re.search(r"\bdec(ember)?\b", t):
        f["month"] = 12
    elif re.search(r"\bnov(ember)?\b", t):
        f["month"] = 11
    if re.search(r"deadline", t):
        f["sort_by"] = "deadline"
    elif re.search(r"\b(first|earliest|order|sequence|timeline|schedule|when|comes? (first|next)|early)\b", t):
        f["sort_by"] = "day"
    elif re.search(r"highest|top|best[- ]paying|max(imum)? ctc|most", t):
        f["sort_by"] = "ctc"
    return f


def route_and_plan(question: str, profile: dict) -> tuple[str, list[tuple[str, dict]]]:
    q = question.lower()
    companies = find_companies(question)
    filters = parse_filters(question)
    wants_prep = bool(re.search(r"\bprep|prepare|study|plan\b|strategy|should i|focus|priorit|roadmap", q))
    wants_stats = bool(re.search(r"how many offers|offers|statistic|\bstats\b|average ctc|overall|total|profiles", q))
    wants_compare = bool(re.search(r"compare|\bvs\.?\b|versus|difference between|better", q))
    personal = bool(re.search(r"\bi have\b|\bmy\b|\bi'm\b|\bi am\b|eligible|can i\b|for me", q))
    free_text = bool(re.search(r"bonus|stock|esop|rsu|equity|bond|relocation|work from home|"
                               r"wfh|selection process|insurance|benefit|perk", q))
    about_oa = bool(re.search(r"\boa\b|online assessment|pattern|interview (questions?|rounds?|experience|process)|"
                              r"rounds?\b|questions asked|topics|syllabus", q))

    stats_only = find_stats_only_companies(question, companies)
    if stats_only and not companies:
        c = stats_only[0]
        return "company", [("get_company_profile", {"company": c}), ("drive_stats", {"company": c}),
                           ("search_docs", {"query": question, "company": c})]
    if len(companies) >= 2 or (wants_compare and companies):
        return "compare", [("get_company_profile", {"company": c}) for c in companies[:4]]
    if len(companies) == 1:
        c = companies[0]
        plan = [("get_company_profile", {"company": c}), ("search_docs", {"query": question, "company": c})]
        if wants_stats:
            plan.append(("drive_stats", {"company": c}))
        return "company", plan
    if wants_prep or personal:
        cgpa, branch = profile.get("cgpa"), profile.get("branch")
        if cgpa is not None and branch:
            args = {"cgpa": cgpa, "branch": branch, "course": profile.get("course", "btech"),
                    "sort_by": filters.get("sort_by", "day")}
            if "role_category" in filters:
                args["role_category"] = filters["role_category"]
            if "min_ctc_lpa" in filters:
                args["min_ctc_lpa"] = filters["min_ctc_lpa"]
            if "location" in filters:
                args["location"] = filters["location"]
            return ("prep_plan" if wants_prep else "eligibility"), [("eligibility_check", args)]
        args = {k: v for k, v in filters.items() if k in ("role_category", "min_ctc_lpa", "max_ctc_lpa", "drive_day", "month", "location")}
        return ("prep_plan" if wants_prep else "eligibility"), [("query_placements_db", {**args, "sort_by": "day", "limit": 30})]
    if free_text and not (filters.keys() & {"min_ctc_lpa", "max_ctc_lpa", "role_category"}):
        args = {"query": question, "k": 8}
        if "location" in filters:
            args["location"] = filters["location"]
        return "semantic", [("search_docs", args)]
    structured_keys = {"min_ctc_lpa", "max_ctc_lpa", "role_category", "drive_day", "month", "sort_by", "location"}
    if wants_stats and not (filters.keys() & {"min_ctc_lpa", "role_category"}):
        return "stats", [("drive_stats", {"day": filters.get("drive_day")})]
    if filters.keys() & structured_keys and not about_oa:
        args = {k: v for k, v in filters.items() if k in structured_keys}
        plan = [("query_placements_db", {**args, "limit": 25})]
        if "drive_day" in filters:
            plan.append(("drive_stats", {"day": filters["drive_day"]}))
        return "structured", plan
    return "semantic", [("search_docs", {"query": question})]
