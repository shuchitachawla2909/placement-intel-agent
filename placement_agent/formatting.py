"""Source numbering + deterministic answer rendering (used in offline mode and as the LLM's fallback)."""
from __future__ import annotations

import re
from datetime import datetime

STOPWORDS = {"which", "what", "companies", "company", "offer", "offers", "with", "that", "have", "there", "their",
             "about", "from", "were", "does", "give", "engineers", "engineer", "roles", "role", "software"}

FOCUS = {
    "SDE / Software": "DSA in C++ (arrays, strings, hashing, binary search, graphs, DP, greedy), OOP, OS/DBMS/CN basics, LLD for SDE-1",
    "Quant": "probability & expectation, brain-teasers, mental maths, fast C++, basic statistics",
    "Data Science / ML / AI": "ML fundamentals (bias-variance, trees/boosting, regularisation, metrics), statistics, SQL, Python/pandas, one project deep-dive",
    "Analyst / Business / Consulting": "guesstimates, case frameworks, SQL/Excel, aptitude & puzzles",
    "Product": "product sense, metrics & root-cause cases, prioritisation, guesstimates",
    "Hardware / Core Engineering": "digital electronics, CMOS/VLSI basics, computer architecture, branch core subjects",
    "Multiple roles": "read the JD and pick the closest role; DSA + aptitude as a common base",
    "Engineering (general / trainee)": "branch core fundamentals + aptitude",
    "Academia": "research statement and a teaching demo",
    "Other": "read the JD for the skill set",
}


class Sources:
    """Collects every 'source' string found in tool results and numbers them S1, S2, ..."""

    def __init__(self):
        self.order: list[str] = []

    def tag(self, src: str | None) -> str:
        if not src:
            return ""
        if src not in self.order:
            self.order.append(src)
        return f"[S{self.order.index(src) + 1}]"

    def collect(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source", "summary_source") and isinstance(v, str):
                    self.tag(v)
                else:
                    self.collect(v)
        elif isinstance(obj, list):
            for v in obj:
                self.collect(v)

    def listing(self) -> str:
        return "\n".join(f"[S{i}] {s}" for i, s in enumerate(self.order, 1))

    def as_list(self) -> list[dict]:
        return [{"tag": f"S{i}", "source": s} for i, s in enumerate(self.order, 1)]


def fmt_date(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %H:%M")
    except ValueError:
        return iso


def fmt_ctc(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else "Refer JD"


def fmt_cutoff(v) -> str:
    return "none" if not v else f"{v:g}"


def fmt_cgpa_rule(v) -> str:
    return "No CGPA cutoff" if not v else f"CGPA >= {v:g}"


def rows_table(rows: list[dict], src: Sources) -> str:
    out = ["| Company | Role | CTC (LPA) | CGPA cutoff | Location | Interview day/slot | Willingness deadline | Src |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['company']} | {r['role'][:45]} | {fmt_ctc(r['ctc_lpa'])} | {fmt_cutoff(r['min_cgpa'])} | "
                   f"{(r.get('location') or 'not stated')[:30]} | "
                   f"{r.get('drive_phase') or 'not announced'} | {fmt_date(r.get('willingness_deadline'))} | {src.tag(r['source'])} |")
    return "\n".join(out)


def profile_block(p: dict, src: Sources) -> str:
    if not p.get("found"):
        return " ".join(p.get("notes", []))
    parts = []
    for o in p["openings"]:
        ctc = "; ".join(
            f"{c['course']}: {c['currency']} {c['ctc']:,.0f}" + (f" (take-home {c['take_home']:,.0f})" if c["take_home"] else "")
            for c in o["ctc_by_course"] if c["ctc"]) or "Refer JD / remarks"
        elig = "; ".join(f"{fmt_cgpa_rule(e['min_cgpa'])} for {', '.join(e['courses'])} in " + (
            ", ".join(e["departments"]) if len(e["departments"]) <= 8
            else f"{len(e['departments'])} branches") + ("" if "cse" in e["departments"] else " (not CSE)")
            for e in o["eligibility"]) or "not listed"
        parts.append(
            f"**{o['company']} - {o['role']}** {src.tag(o['source'])}\n"
            f"- CTC: {ctc}\n- Eligibility: {elig}\n"
            f"- Willingness deadline: {fmt_date(o['willingness_deadline'])} | Interview: "
            f"{o['drive_phase'] or 'day not announced'}{' ' + o['slot_timing'] if o.get('slot_timing') else ''}\n"
            f"- Location: {o['location'] or '-'}\n- Remarks: {(o['remarks'] or '-')[:300]}")
    return "\n\n".join(parts)


def stats_block(s: dict, src: Sources, company: str | None = None) -> str:
    sm = s["summary"]
    if company:
        if not s["rounds"]:
            return f"{company} does not appear in the statistics sheet either."
        return "\n".join(f"- {company} appears in **{r['label']}** ({r['section']}) as: {', '.join(r['companies'])} "
                         f"{src.tag(r['source'])}" for r in s["rounds"])
    lines = [f"Drive totals till 5 Dec 2025: **{sm.get('total_offers')} offers**, {sm.get('total_profiles')} profiles, "
             f"highest CTC {float(sm.get('highest_ctc_lpa', 0)) / 100:g} Cr, average {sm.get('average_ctc_lpa')} LPA "
             f"{src.tag(s['summary_source'])}", "",
             "| Phase | Offers | Profiles | Companies (sample) | Src |", "|---|---|---|---|---|"]
    for r in s["rounds"]:
        sample = ", ".join(r["companies"][:8]) + (" ..." if len(r["companies"]) > 8 else "")
        lines.append(f"| {r['label']} | {r['offers']} | {r['profiles']} | {sample} | {src.tag(r['source'])} |")
    return "\n".join(lines)


def search_block(res: dict, src: Sources, n: int = 5, query: str = "") -> str:
    out = []
    terms = [t for t in re.findall(r"[a-z]{4,}", query.lower()) if t not in STOPWORDS]
    for h in res["hits"][:n]:
        text = " ".join(h["text"].split())
        head = ""
        m = re.match(r"Company: (.*?) Role: (.*?) \(category", text)
        if m:  # opening card: show company/role + the remarks sentence around the query terms
            head = f"**{m.group(1)} - {m.group(2)}**: "
            rem = text.split("Remarks:", 1)[-1].strip()
            comp = re.search(r"Compensation: (.*?) Willingness", text)
            pos = min([rem.lower().find(t) for t in terms if t in rem.lower()] or [0])
            snippet = rem[max(0, pos - 80): pos + 220] or (comp.group(1) if comp else "")
            text = ("..." if pos > 80 else "") + snippet
        out.append(f"- {head}{text[:350]}{'...' if len(text) > 350 else ''} {src.tag(h['source'])}")
    return "\n\n".join(out) or "No matching passages."


def prep_plan(rows: list[dict], profile: dict, src: Sources, per_day: int = 8, counts: dict | None = None) -> str:
    pref = set(profile.get("preferred_roles", []))
    pref_map = {"sde": "SDE / Software", "data": "Data Science / ML / AI", "quant": "Quant",
                "analyst": "Analyst / Business / Consulting", "product": "Product", "hardware": "Hardware / Core Engineering"}
    pref_cats = {pref_map[p] for p in pref if p in pref_map}

    def key(r):
        return (r["category"] not in pref_cats if pref_cats else 0, -(r["ctc_lpa"] or 0))

    groups: dict = {}
    for r in rows:
        groups.setdefault(r["drive_day"], []).append(r)
    lines = ["### Prep order - what comes first"]
    for day in sorted(groups, key=lambda d: (d is None, d or 0)):
        title = f"Day {day}" if day else "Interview day not announced (go by willingness deadline)"
        n = (counts or {}).get(str(day) if day else "not announced", len(groups[day]))
        lines.append(f"\n**{title}** - {n} eligible openings (top {min(per_day, len(groups[day]))} shown)")
        g = sorted(groups[day], key=lambda r: (r.get("drive_phase") or "", *key(r))) if day else \
            sorted(groups[day], key=lambda r: (r["willingness_deadline"] or "9999", *key(r)))
        for r in g[:per_day]:
            lines.append(f"- {r.get('drive_phase') or 'deadline ' + fmt_date(r['willingness_deadline'])}: "
                         f"{r['company']} - {r['role'][:40]} - {fmt_ctc(r['ctc_lpa'])}{' LPA' if r['ctc_lpa'] else ''} - "
                         f"cutoff {fmt_cutoff(r['min_cgpa'])} "
                         f"{src.tag(r['source'])}")
    cats = []
    for r in sorted(rows, key=lambda r: (r["drive_day"] or 99)):
        if r["category"] not in cats and r["category"] != "Other":
            cats.append(r["category"])
    lines.append("\n### What to study (general guidance - the documents have no OA patterns)")
    for c in cats[:5]:
        lines.append(f"- **{c}**: {FOCUS.get(c, FOCUS['Other'])}")
    if profile.get("strengths"):
        lines.append(f"\nYou said you're strong in {profile['strengths']} - lean on that for the Day-1 coding rounds.")
    return "\n".join(lines)
