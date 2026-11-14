"""Ingestion pipeline: PDFs -> structured SQLite + RAG documents (+ optional dense index).

Run:  python -m placement_agent.ingest.build
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from datetime import date

from placement_agent import config
from placement_agent.ingest.parse_portal import parse_portal
from placement_agent.ingest.parse_stats import parse_stats

logging.getLogger("pdfminer").setLevel(logging.ERROR)

DEPT_NAMES = {
    "cse": "Computer Science", "eee": "Electrical", "ece": "Electronics", "mec": "Mechanical", "civ": "Civil",
    "che": "Chemical", "cer": "Ceramic", "met": "Metallurgical", "min": "Mining", "phe": "Pharmaceutical",
    "bce": "Biochemical", "bme": "Biomedical", "mst": "Materials Science", "chy": "Chemistry", "phy": "Physics",
    "mat": "Mathematics & Computing", "apd": "Architecture", "dse": "Data Science", "hss": "Humanities",
}
COURSE_NAMES = {"btech": "B.Tech", "idd": "IDD/IMD", "mtech": "M.Tech", "phd": "PhD"}
CTC_PRIORITY = ["B.Tech", "IDD/IMD", "M.Tech", "PhD", "All"]

# Keyword rules for a coarse role category (derived field, used for filters such as "SDE roles").
ROLE_RULES = [
    ("Quant", r"quant|trading"),
    ("Data Science / ML / AI", r"data scien|machine learning|\bml\b|\bai\b|applied scientist|\bds\b|research scientist"),
    ("SDE / Software", r"software|sde|\bsw\b|developer|full ?stack|backend|frontend|\bui\b|engineer i\b|programmer|"
                       r"member of technical staff|swe|web|cloud|data engineer|platform engineer|technology developer"),
    ("Product", r"product manager|\bapm\b|product analyst|product designer"),
    ("Analyst / Business / Consulting", r"analyst|consult|business|associate|management trainee|young|campaign|leaders"),
    ("Hardware / Core Engineering", r"asic|hardware|\bhw\b|embedded|vlsi|silicon|design engineer|get\b|pget|graduate engineer|"
                                    r"process|quality|r&d|research & development|yield|verification|application engineer|"
                                    r"cae|material|scientist"),
    ("Academia", r"professor|faculty"),
    ("Engineering (general / trainee)", r"engineer|trainee|traine|executive"),
]


def role_category(role: str) -> str:
    r = role.lower()
    if re.search(r"multiple|refer jd|refer description", r):
        return "Multiple roles"
    for cat, pat in ROLE_RULES:
        if re.search(pat, r):
            return cat
    return "Other"


STOP = r"\b(pvt|private|ltd|limited|llp|inc|industries|technologies|technology|solutions|corporation|group|india|co)\b\.?"


def norm_company(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name.lower())
    name = re.sub(STOP, "", name)
    return re.sub(r"[^a-z0-9]", "", name)


def same_company(a: str, b: str) -> bool:
    a, b = norm_company(a), norm_company(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short)


def _pick_ctc(ctc_rows):
    for course in CTC_PRIORITY:
        for c in ctc_rows:
            if c["course"] == course and c["ctc"]:
                return c
    return None


def _lpa(amount, currency):
    if amount is None:
        return None
    return round(amount * config.FX_TO_INR.get(currency, 1.0) / 1e5, 2)


SCHEMA = """
DROP TABLE IF EXISTS openings; DROP TABLE IF EXISTS ctc; DROP TABLE IF EXISTS eligibility;
DROP TABLE IF EXISTS rounds; DROP TABLE IF EXISTS round_companies; DROP TABLE IF EXISTS summary;
CREATE TABLE openings (
  id INTEGER PRIMARY KEY, company TEXT, role TEXT, role_category TEXT, drive_type TEXT, status TEXT,
  willingness_deadline TEXT, slot_date TEXT, slot_no INTEGER, slot_timing TEXT,
  drive_day INTEGER, drive_phase TEXT, day_source TEXT,
  min_cgpa REAL, ctc_lpa REAL, take_home_lpa REAL, ctc_currency TEXT, ctc_course TEXT, ctc_source TEXT,
  location TEXT, jd_file TEXT, remarks TEXT, pages TEXT, last_updated TEXT);
CREATE TABLE ctc (opening_id INTEGER, course TEXT, currency TEXT, ctc REAL, take_home REAL);
CREATE TABLE eligibility (opening_id INTEGER, group_no INTEGER, course TEXT, department TEXT,
  min_cgpa REAL, x_pct REAL, xii_pct REAL);
CREATE TABLE rounds (id INTEGER PRIMARY KEY, label TEXT, section TEXT, offers INTEGER, profiles INTEGER,
  day_from INTEGER, day_to INTEGER);
CREATE TABLE round_companies (round_id INTEGER, company TEXT);
CREATE TABLE summary (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_elig ON eligibility(course, department);
"""


def build_db(openings: list[dict], stats: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.executescript(SCHEMA)
    day1 = date.fromisoformat(config.DRIVE_DAY1)

    rounds = stats["rounds"]
    for i, r in enumerate(rounds, 1):
        r["id"] = i
        con.execute("INSERT INTO rounds VALUES (?,?,?,?,?,?,?)",
                    (i, r["label"], r["section"], r["offers"], r["profiles"], r["day_from"], r["day_to"]))
        con.executemany("INSERT INTO round_companies VALUES (?,?)", [(i, c) for c in r["companies"]])
    for k, v in stats["summary"].items():
        con.execute("INSERT INTO summary VALUES (?,?)", (k, str(v)))

    for o in openings:
        # Which day of the drive? Prefer the portal slot; fall back to the stats sheet's day-wise lists.
        day = phase = source = None
        if o["slot_date"]:
            day = (date.fromisoformat(o["slot_date"]) - day1).days + 1
            phase, source = f"Day {day} (slot {o['slot_no']})", "portal slot"
        else:
            for r in rounds:
                if r["day_from"] and any(same_company(o["company"], c) for c in r["companies"]):
                    day, phase, source = r["day_from"], r["label"], "stats sheet"
                    break
        o.update(drive_day=day, drive_phase=phase, day_source=source, role_category=role_category(o["role"]))
        best = _pick_ctc(o["ctc"])
        o["min_cgpa"] = min((e["cgpa"] for e in o["eligibility"]), default=None)
        o["ctc_lpa"] = _lpa(best["ctc"], best["currency"]) if best else None
        o["take_home_lpa"] = _lpa(best["take_home"], best["currency"]) if best else None
        con.execute(
            "INSERT INTO openings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (o["entry_no"], o["company"], o["role"], o["role_category"], o["drive_type"], o["status"],
             o["willingness_deadline"], o["slot_date"], o["slot_no"], o["slot_timing"], day, phase, source,
             o["min_cgpa"], o["ctc_lpa"], o["take_home_lpa"], best["currency"] if best else None,
             best["course"] if best else None, o["ctc_source"], o["location"], o["jd_file"], o["remarks"],
             ",".join(map(str, o["pages"])), o["last_updated"]))
        con.executemany("INSERT INTO ctc VALUES (?,?,?,?,?)",
                        [(o["entry_no"], c["course"], c["currency"], c["ctc"], c["take_home"]) for c in o["ctc"]])
        for g, e in enumerate(o["eligibility"]):
            con.executemany("INSERT INTO eligibility VALUES (?,?,?,?,?,?,?)",
                            [(o["entry_no"], g, c, d, e["cgpa"], e["x_pct"], e["xii_pct"])
                             for c in e["courses"] for d in e["departments"]])
    con.commit()
    con.close()


def _fmt_money(v, cur):
    if v is None:
        return "Refer Description"
    return f"{cur} {v:,.0f}" + (f" (~{_lpa(v, cur)} LPA)" if cur == "INR" else f" (~{_lpa(v, cur)} LPA in INR)")


def opening_card(o: dict) -> str:
    elig = "; ".join(
        f"CGPA >= {e['cgpa']:.2f} for {', '.join(COURSE_NAMES.get(c, c) for c in e['courses'])} in "
        f"{', '.join(e['departments'])}" for e in o["eligibility"]) or "not listed"
    ctc = "; ".join(f"{c['course']}: CTC {_fmt_money(c['ctc'], c['currency'])}, take-home "
                    f"{_fmt_money(c['take_home'], c['currency'])}" for c in o["ctc"]) or "Refer Description / JD"
    slot = (f"{o['drive_phase']}, {o['slot_date']}, timing {o['slot_timing']}" if o["slot_date"]
            else (f"{o['drive_phase']} (from statistics sheet)" if o["drive_phase"] else "not announced in portal"))
    return (
        f"Company: {o['company']}\nRole: {o['role']} (category: {o['role_category']})\n"
        f"Drive: {o['drive_type']} | Status: {o['status']}\n"
        f"Eligibility: {elig}. Active backlogs allowed: {o['active_backlog']}\n"
        f"Compensation: {ctc}\n"
        f"Willingness deadline: {o['willingness_deadline']}\nInterview slot / day: {slot}\n"
        f"Location: {o['location'] or 'see remarks'}\nJD file: {o['jd_file']}\n"
        f"Remarks: {o['remarks']}"
    )


def build_documents(openings: list[dict], stats: dict) -> list[dict]:
    docs = []
    portal = config.PORTAL_PDF.name
    for o in openings:
        docs.append({
            "id": f"opening-{o['entry_no']}", "text": opening_card(o),
            "metadata": {"doc_type": "opening", "opening_id": o["entry_no"], "company": o["company"],
                         "role": o["role"], "source": portal, "page": o["pages"][0] if o["pages"] else None},
        })
    for r in stats["rounds"]:
        docs.append({
            "id": f"round-{r['id']}",
            "text": (f"Placement drive 2025-26 phase: {r['label']} ({r['section']}). Offers: {r['offers']}, "
                     f"profiles: {r['profiles']}. Companies: {', '.join(r['companies'])}."),
            "metadata": {"doc_type": "stats", "round_id": r["id"], "company": None,
                         "source": config.STATS_PDF.name, "page": 1 if r["id"] <= 3 else 2},
        })
    s = stats["summary"]
    docs.append({"id": "summary", "text": (
        f"Placement Drive 2025-26 totals until 5 Dec 2025: {s.get('total_offers')} offers, "
        f"{s.get('total_profiles')} profiles, highest CTC {s.get('highest_ctc_lpa')} LPA ({float(s.get('highest_ctc_lpa') or 0) / 100:g} Cr), "
        f"average CTC {s.get('average_ctc_lpa')} LPA."),
        "metadata": {"doc_type": "stats", "company": None, "source": config.STATS_PDF.name, "page": 2}})
    docs += load_notes()
    return docs


def load_notes() -> list[dict]:
    """Optional extra knowledge (OA patterns, interview experiences...) dropped into data/raw/notes/.
    File name = company name, e.g. notes/<CompanyName>.md."""
    if not config.NOTES_DIR.exists():
        return []
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    docs = []
    for f in sorted(config.NOTES_DIR.iterdir()):
        if f.suffix.lower() in (".md", ".txt"):
            pages = [(1, f.read_text(encoding="utf-8", errors="ignore"))]
        elif f.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            pages = [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(f).pages, 1)]
        else:
            continue
        for pno, text in pages:
            for j, chunk in enumerate(splitter.split_text(text)):
                docs.append({"id": f"note-{f.stem}-{pno}-{j}", "text": chunk,
                             "metadata": {"doc_type": "note", "company": f.stem, "source": f.name, "page": pno}})
    return docs


def main():
    t0 = time.time()
    print("Parsing portal PDF ...")
    openings = parse_portal(config.PORTAL_PDF)
    print(f"  {len(openings)} company profiles")
    stats = parse_stats(config.STATS_PDF)
    print(f"  {len(stats['rounds'])} drive phases from statistics sheet")
    build_db(openings, stats)
    docs = build_documents(openings, stats)
    (config.DATA_DIR / "docs.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    (config.DATA_DIR / "openings.json").write_text(json.dumps(openings, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {config.DB_PATH.name} and {len(docs)} RAG documents")

    from placement_agent.retriever import build_dense_index

    n = build_dense_index(docs)
    print(f"  dense index: {n}" if n else "  dense index: skipped (EMBEDDINGS not set) - BM25 only")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
