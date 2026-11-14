"""Agent tools. Every SQL statement is built from a fixed template with bound parameters - the LLM
only chooses filter values, never writes SQL."""
from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache

from langchain_core.tools import tool

from placement_agent import config
from placement_agent.ingest.build import COURSE_NAMES, DEPT_NAMES, norm_company, same_company

PORTAL = "Placements'26.pdf"
STATS = "Placement Statistics Till Day 5.pdf"

BRANCH_ALIASES = {
    "cse": "cse", "cs": "cse", "computer": "cse", "computer science": "cse",
    "ece": "ece", "electronics": "ece", "eee": "eee", "ee": "eee", "electrical": "eee",
    "mech": "mec", "mechanical": "mec", "mec": "mec", "civil": "civ", "civ": "civ",
    "chemical": "che", "che": "che", "ceramic": "cer", "cer": "cer", "metallurgy": "met", "metallurgical": "met",
    "met": "met", "mining": "min", "min": "min", "pharma": "phe", "pharmaceutical": "phe", "phe": "phe",
    "biochemical": "bce", "bce": "bce", "biomedical": "bme", "bme": "bme", "materials": "mst", "mst": "mst",
    "chemistry": "chy", "chy": "chy", "physics": "phy", "phy": "phy", "maths": "mat", "math": "mat",
    "mathematics": "mat", "mnc": "mat", "m&c": "mat", "mat": "mat", "architecture": "apd", "apd": "apd",
    "data science": "dse", "dse": "dse", "humanities": "hss", "hss": "hss",
}
COURSE_ALIASES = {"btech": "btech", "b.tech": "btech", "idd": "idd", "imd": "idd", "dual": "idd",
                  "mtech": "mtech", "m.tech": "mtech", "phd": "phd"}
CATEGORY_ALIASES = {
    "sde": "SDE / Software", "software": "SDE / Software", "swe": "SDE / Software", "developer": "SDE / Software",
    "data": "Data Science / ML / AI", "ml": "Data Science / ML / AI", "ai": "Data Science / ML / AI",
    "data science": "Data Science / ML / AI", "quant": "Quant", "analyst": "Analyst / Business / Consulting",
    "consulting": "Analyst / Business / Consulting", "business": "Analyst / Business / Consulting",
    "product": "Product", "hardware": "Hardware / Core Engineering", "core": "Hardware / Core Engineering",
    "vlsi": "Hardware / Core Engineering", "academia": "Academia", "multiple": "Multiple roles",
}


CITY_ALIASES = {
    "bangalore": ["bangalore", "bengaluru"], "bengaluru": ["bangalore", "bengaluru"],
    "gurgaon": ["gurgaon", "gurugram"], "gurugram": ["gurgaon", "gurugram"], "delhi": ["delhi"],
    "hyderabad": ["hyderabad"], "pune": ["pune"], "noida": ["noida"], "mumbai": ["mumbai"],
    "chennai": ["chennai"], "kolkata": ["kolkata"], "dubai": ["dubai"], "remote": ["remote", "work from home"],
    "ahmedabad": ["ahmedabad"], "jaipur": ["jaipur"], "varanasi": ["varanasi"],
}


def city_terms(location: str | None) -> list[str]:
    if not location:
        return []
    loc = location.strip().lower()
    return CITY_ALIASES.get(loc, [loc])


def norm_branch(b: str | None) -> str | None:
    if not b:
        return None
    b = b.strip().lower()
    return BRANCH_ALIASES.get(b, b if b in DEPT_NAMES else None)


def norm_course(c: str | None) -> str | None:
    if not c:
        return None
    return COURSE_ALIASES.get(c.strip().lower().replace(" ", ""), None)


def norm_category(c: str | None) -> str | None:
    if not c:
        return None
    c = c.strip().lower()
    if c in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[c]
    for k, v in CATEGORY_ALIASES.items():
        if k in c or v.lower() == c:
            return v
    return None


def _con():
    if not config.DB_PATH.exists():
        raise FileNotFoundError("Database missing. Run: python -m placement_agent.ingest.build")
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@lru_cache(maxsize=1)
def company_names() -> list[str]:
    with _con() as con:
        return [r[0] for r in con.execute("SELECT DISTINCT company FROM openings ORDER BY company")]


@lru_cache(maxsize=1)
def stats_company_names() -> list[str]:
    with _con() as con:
        return sorted({r[0] for r in con.execute("SELECT company FROM round_companies")})


def resolve_company(name: str) -> list[str]:
    """Map a user-typed company name to the canonical names in the DB."""
    return [c for c in company_names() if same_company(name, c)] or \
           [c for c in company_names() if norm_company(name) and norm_company(name) in norm_company(c)]


def _source(row) -> str:
    page = str(row["pages"]).split(",")[0]
    return f"{PORTAL} p.{page} (#{row['id']})"


def _row_summary(r) -> dict:
    return {
        "id": r["id"], "company": r["company"], "role": r["role"], "category": r["role_category"],
        "ctc_lpa": r["ctc_lpa"], "take_home_lpa": r["take_home_lpa"], "ctc_basis": r["ctc_course"],
        "min_cgpa": r["min_cgpa"], "willingness_deadline": r["willingness_deadline"],
        "drive_day": r["drive_day"], "drive_phase": r["drive_phase"], "slot_timing": r["slot_timing"],
        "location": r["location"] or "not stated", "source": _source(r),
    }


SORTS = {
    "ctc": "o.ctc_lpa IS NULL, o.ctc_lpa DESC",
    "day": "o.drive_day IS NULL, o.drive_day, o.slot_no IS NULL, o.slot_no, o.willingness_deadline",
    "deadline": "o.willingness_deadline IS NULL, o.willingness_deadline",
    "cgpa": "o.min_cgpa DESC",
}


def query_openings(company: str | None = None, role_keyword: str | None = None, role_category: str | None = None,
                   min_ctc_lpa: float | None = None, max_ctc_lpa: float | None = None,
                   drive_day: int | None = None, month: int | None = None,
                   cgpa: float | None = None, branch: str | None = None, course: str | None = None,
                   location: str | None = None, sort_by: str = "ctc", limit: int = 15) -> dict:
    where, params, notes = [], [], []
    if company:
        names = resolve_company(company)
        if not names:
            return {"rows": [], "count": 0, "notes": [f"No company matching '{company}' in the 2025-26 portal data."]}
        where.append(f"o.company IN ({','.join('?' * len(names))})")
        params += names
    if role_keyword:
        where.append("LOWER(o.role) LIKE ?")
        params.append(f"%{role_keyword.lower()}%")
    cat = norm_category(role_category)
    if role_category and cat:
        where.append("o.role_category = ?")
        params.append(cat)
    if min_ctc_lpa is not None:
        where.append("o.ctc_lpa >= ?")
        params.append(float(min_ctc_lpa))
    if max_ctc_lpa is not None:
        where.append("o.ctc_lpa <= ?")
        params.append(float(max_ctc_lpa))
    if drive_day is not None:
        where.append("o.drive_day = ?")
        params.append(int(drive_day))
    if month is not None:
        # Interview days of the 2025-26 drive all fall in December (Day 1 = 1 Dec 2025).
        where.append("((o.slot_date IS NOT NULL AND CAST(substr(o.slot_date, 6, 2) AS INTEGER) = ?) "
                     "OR (o.slot_date IS NULL AND o.drive_day IS NOT NULL AND ? = 12))")
        params += [int(month), int(month)]
        notes.append("Month filter uses the interview day (portal slot or statistics-sheet day); "
                     "openings with no announced day are excluded.")
    terms = city_terms(location)
    if terms:
        where.append("(" + " OR ".join(["LOWER(COALESCE(o.location,'') || ' ' || COALESCE(o.remarks,'')) LIKE ?"] * len(terms)) + ")")
        params += [f"%{t}%" for t in terms]
        notes.append(f"Location filter '{location}' matches the city named in a card's location/remarks; "
                     f"about 96 of 274 cards name no location and are excluded.")
    b, c = norm_branch(branch), norm_course(course) or ("btech" if (branch or cgpa is not None) else None)
    if branch and not b:
        notes.append(f"Unknown branch '{branch}' - branch filter ignored.")
    if b or c or cgpa is not None:
        sub, sp = ["e.opening_id = o.id"], []
        if b:
            sub.append("e.department = ?")
            sp.append(b)
        if c:
            sub.append("e.course = ?")
            sp.append(c)
        if cgpa is not None:
            sub.append("e.min_cgpa <= ?")
            sp.append(float(cgpa))
        where.append(f"EXISTS (SELECT 1 FROM eligibility e WHERE {' AND '.join(sub)})")
        params += sp
    sql = "SELECT o.* FROM openings o"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {SORTS.get(sort_by, SORTS['ctc'])} LIMIT ?"
    limit = max(1, min(int(limit or 15), 300))
    params.append(limit)
    with _con() as con:
        rows = [_row_summary(r) for r in con.execute(sql, params)]
        count_sql = "SELECT COUNT(*) FROM openings o" + (" WHERE " + " AND ".join(where) if where else "")
        total = con.execute(count_sql, params[:-1]).fetchone()[0]
    if role_category and not cat:
        notes.append(f"Unknown role category '{role_category}' - ignored.")
    if any(r["ctc_lpa"] is None for r in rows):
        notes.append("ctc_lpa = null means the portal says 'Refer Description/JD' (check remarks).")
    return {"rows": rows, "count": total, "shown": len(rows), "notes": notes}


def company_profile(company: str) -> dict:
    names = resolve_company(company)
    if not names:
        return {"company": company, "found": False,
                "notes": [f"'{company}' is not in the 2025-26 portal export (it may have hired only through "
                          f"internships/PPOs)."]}
    out = []
    with _con() as con:
        for o in con.execute(f"SELECT * FROM openings WHERE company IN ({','.join('?' * len(names))}) ORDER BY id", names):
            ctc = [dict(r) for r in con.execute("SELECT course, currency, ctc, take_home FROM ctc WHERE opening_id=?", (o["id"],))]
            groups = {}
            for e in con.execute("SELECT * FROM eligibility WHERE opening_id=? ORDER BY group_no", (o["id"],)):
                g = groups.setdefault(e["group_no"], {"min_cgpa": e["min_cgpa"], "courses": set(), "departments": set()})
                g["courses"].add(COURSE_NAMES.get(e["course"], e["course"]))
                g["departments"].add(e["department"])
            elig = [{"min_cgpa": g["min_cgpa"], "courses": sorted(g["courses"]), "departments": sorted(g["departments"])}
                    for g in groups.values()]
            out.append({**_row_summary(o), "ctc_by_course": ctc, "eligibility": elig, "remarks": o["remarks"],
                        "jd_file": o["jd_file"], "status": o["status"]})
    return {"company": company, "found": True, "openings": out}


def eligible_openings(cgpa: float, branch: str, course: str = "btech", role_category: str | None = None,
                      min_ctc_lpa: float | None = None, location: str | None = None, sort_by: str = "day",
                      per_day: int = 12) -> dict:
    """All eligible openings; with sort_by='day' the result keeps the top `per_day` (by CTC) for each
    interview day so that the whole timeline fits in an LLM context."""
    res = query_openings(cgpa=cgpa, branch=branch, course=course, role_category=role_category,
                         min_ctc_lpa=min_ctc_lpa, location=location, sort_by="ctc", limit=300)
    res["criteria"] = {"cgpa": cgpa, "branch": norm_branch(branch), "course": norm_course(course) or "btech"}
    rows = res["rows"]
    if sort_by == "day":
        by_day: dict = {}
        for r in rows:
            by_day.setdefault(r["drive_day"], []).append(r)
        res["per_day_counts"] = {str(d) if d else "not announced": len(v) for d, v in sorted(by_day.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}
        keep = []
        for d, v in by_day.items():
            v = v if d else sorted(v, key=lambda r: r["willingness_deadline"] or "9999")
            keep += v[:per_day]
        rows = sorted(keep, key=lambda r: (r["drive_day"] is None, r["drive_day"] or 0, r["drive_phase"] or "",
                                           -(r["ctc_lpa"] or 0)))
        res["notes"] = res.get("notes", []) + [f"Showing the top {per_day} by CTC for each interview day."]
    else:
        rows = rows[:40]
    res["rows"], res["shown"] = rows, len(rows)
    return res


def drive_statistics(day: int | None = None, company: str | None = None) -> dict:
    with _con() as con:
        rounds = [dict(r) for r in con.execute("SELECT * FROM rounds ORDER BY id")]
        for r in rounds:
            r["companies"] = [x[0] for x in con.execute("SELECT company FROM round_companies WHERE round_id=?", (r["id"],))]
            r["source"] = f"{STATS} p.{1 if r['id'] <= 3 else 2}"
        summary = {k: v for k, v in con.execute("SELECT key, value FROM summary")}
    if day is not None:
        rounds = [r for r in rounds if r["day_from"] and r["day_from"] <= int(day) <= r["day_to"]]
    if company:
        rounds = [{**r, "companies": list(dict.fromkeys(c for c in r["companies"] if same_company(company, c)))}
                  for r in rounds]
        rounds = [r for r in rounds if r["companies"]]
    return {"summary": summary, "summary_source": f"{STATS} p.2", "rounds": rounds}


def search(query: str, company: str | None = None, k: int = 5, location: str | None = None) -> dict:
    from placement_agent.retriever import get_retriever

    hits = get_retriever().search(query, k=k, company=company, location_terms=city_terms(location) or None)
    return {"hits": [{"text": h["text"][:1500], "source": f"{h['metadata']['source']} p.{h['metadata'].get('page')}"
                      + (f" (#{h['metadata']['opening_id']})" if h["metadata"].get("opening_id") else ""),
                      "company": h["metadata"].get("company"), "score": h["score"]} for h in hits]}


# ---------------------------------------------------------------- LangChain tool wrappers (for the LLM)
def _dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


@tool
def query_placements_db(company: str | None = None, role_keyword: str | None = None, role_category: str | None = None,
                        min_ctc_lpa: float | None = None, max_ctc_lpa: float | None = None,
                        drive_day: int | None = None, month: int | None = None, location: str | None = None,
                        sort_by: str = "ctc", limit: int = 15) -> str:
    """Structured search over 2025-26 placement openings (IIT BHU portal).
    Use for filters/rankings: CTC thresholds (in LPA), role category (sde, data, quant, analyst, product,
    hardware), role keyword, interview day of the drive (1-5) or month (12 = December), a company,
    location (a city such as hyderabad, bangalore, pune, gurgaon, noida, mumbai, remote).
    sort_by: 'ctc' (highest first), 'day' (earliest interview day first), 'deadline' (earliest willingness deadline)."""
    return _dump(query_openings(company=company, role_keyword=role_keyword, role_category=role_category,
                                min_ctc_lpa=min_ctc_lpa, max_ctc_lpa=max_ctc_lpa, drive_day=drive_day,
                                month=month, location=location, sort_by=sort_by, limit=limit))


@tool
def get_company_profile(company: str) -> str:
    """Everything the portal has on one company: each role, CTC/take-home per course, eligibility
    (CGPA cutoff, courses, branches), willingness deadline, interview day/slot, location, remarks. Use for
    'tell me about X' and for comparisons (call once per company)."""
    return _dump(company_profile(company))


@tool
def eligibility_check(cgpa: float, branch: str, course: str = "btech", role_category: str | None = None,
                      min_ctc_lpa: float | None = None, location: str | None = None, sort_by: str = "day") -> str:
    """Openings a student is eligible for, given CGPA, branch (e.g. cse, ece, mech) and course
    (btech, idd, mtech). Optional role_category / min_ctc_lpa / location (city) filters.
    sort_by='day' lists what comes first."""
    return _dump(eligible_openings(cgpa=cgpa, branch=branch, course=course, role_category=role_category,
                                   min_ctc_lpa=min_ctc_lpa, location=location, sort_by=sort_by))


@tool
def drive_stats(day: int | None = None, company: str | None = None) -> str:
    """Placement statistics sheet: offers and profiles per phase (internship, PPO, Day 1..5), companies per
    phase, total offers, highest and average CTC. Filter by drive day or company."""
    return _dump(drive_statistics(day=day, company=company))


@tool
def search_docs(query: str, company: str | None = None, k: int = 5, location: str | None = None) -> str:
    """Semantic/keyword search over the source documents (portal remarks, JD notes, stats, and any extra notes
    such as OA patterns). Use for free-text details: bonuses, stocks, bond, selection process.
    Optional location (city) keeps only documents that mention it."""
    return _dump(search(query, company=company, k=max(1, min(int(k), 10)), location=location))


TOOLS = [query_placements_db, get_company_profile, eligibility_check, drive_stats, search_docs]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
