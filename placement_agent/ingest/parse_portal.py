"""Parse the placement-portal export ("Company's Visit, 2025-26", one card per company profile).

The PDF is a print of a web page. Text order (pypdf) is reliable per card, but company name and role
are both plain wrapped lines in the card header, so we use pdfplumber font info (company names are
bold, ~12.6pt) to tell them apart.
"""
from __future__ import annotations

import re
from datetime import datetime

import pdfplumber
from pypdf import PdfReader

NOISE_PATTERNS = [
    r"^\(/\)", r"Training and Placement Cell", r"Indian Institute of Technology \(Banaras",
    r"tpo@iitbhu", r"Filter & Sort", r"^\d{1,3}$", r"\(\?page=", r"^page=\d", r"Search term",
    r"Company's Visit", r"^Menu", r"Placement Brochure", r"Website designed", r"\(/credits\)", r"Know Varanasi", r"\(/contacts/\)", r"\(/company/opportunities\)", r"Need Help\?",
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS))
ENTRY_RE = re.compile(r"^(\d+) Eligibility Criteria\b")
COURSE_LABELS = r"B\.Tech|IDD/IMD|M\.Tech|PhD"
CTC_RE = re.compile(
    rf"({COURSE_LABELS})\s*(INR|AED|USD)\s*([\d,.]+)\s*(?:PA|PM)?\s*(?:(INR|AED|USD)\s*([\d,.]+)\s*(?:PA|PM)?)?"
)
ELIG_RE = re.compile(r"X : ([\d.]+)%\s*XII : ([\d.]+)%\s*CGPA : ([\d.]+)\s*Course\(s\) :\s*((?:[a-z]+\b\s*)*)")
DEPT_RE = re.compile(r"Department\(s\) :\s*((?:[a-z]{3}\b\s*)*)")
BACKLOG_RE = re.compile(r"Active backlog allowed :\s*(\S+?)\s*Total backlog\s*allowed :\s*(\S+)")
DEADLINE_RE = re.compile(r"(\w{3}), (\d{1,2}) (\w{3}) (\d{4}) \| (\d{1,2})(?::(\d{2}))? ([AP]M)")
SLOT_RE = re.compile(r"Slot : (\w{3})\w*\.? (\d{1,2}), (\d{4}), slot (\d+)")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _bold_company_names(pdf_path) -> set[str]:
    """Company names = bold words in the left header column (x < 190), grouped into lines/blocks.
    (Most pages render them at 12.6pt; a few pages are printed at a smaller scale, so size is not fixed.)"""
    names = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = [
                w for w in page.extract_words(extra_attrs=["fontname", "size"])
                if "Bold" in w["fontname"] and w["size"] >= 10.0 and w["x0"] < 190 and not w["text"].isdigit()
            ]
            words.sort(key=lambda w: (round(w["top"]), w["x0"]))
            block, last_top = [], None
            for w in words:
                if last_top is not None and w["top"] - last_top > 22:
                    names.add(_norm(" ".join(block)))
                    block = []
                block.append(w["text"])
                last_top = w["top"]
            if block:
                names.add(_norm(" ".join(block)))
    return names


def _clean_lines(pdf_path) -> list[tuple[str, int]]:
    reader = PdfReader(pdf_path)
    out = []
    for pno, page in enumerate(reader.pages, start=1):
        for line in (page.extract_text() or "").splitlines():
            line = re.sub(r"[\ue000-\uf8ff]", "", line).strip()  # drop icon-font glyphs
            if len(line) <= 1 or NOISE_RE.search(line):
                continue
            out.append((line, pno))
    return out


def _parse_deadline(text: str) -> str | None:
    m = DEADLINE_RE.search(text)
    if not m:
        return None
    _, d, mon, y, h, mi, ap = m.groups()
    h = int(h) % 12 + (12 if ap == "PM" else 0)
    return datetime(int(y), MONTHS[mon], int(d), h, int(mi or 0)).isoformat(timespec="minutes")


def _to_number(s: str) -> float | None:
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def _find_header(lines, type_idx, lo, bold_names):
    """Return (company_start, role_start) indices for the header ending just before type_idx."""
    best = None
    for start in range(type_idx - 1, max(lo, type_idx - 14) - 1, -1):
        for end in range(start + 1, min(start + 5, type_idx) + 1):
            if _norm(" ".join(l for l, _ in lines[start:end])) in bold_names:
                if best is None or start > best[0] or (start == best[0] and end > best[1]):
                    best = (start, end)
        if best:
            break
    return best  # None -> the header was pushed below the eligibility block (page break)


def _company_from_body(body_lines: list[str], bold_names: set[str]) -> str:
    """Find (and remove) the company-name lines when they were pushed into the card body."""
    anchor = body_lines.index("Willing Candidates") if "Willing Candidates" in body_lines else len(body_lines)
    for start in range(anchor - 1, -1, -1):
        for end in range(min(start + 4, anchor), start, -1):
            if _norm(" ".join(body_lines[start:end])) in bold_names:
                name = " ".join(body_lines[start:end])
                del body_lines[start:end]
                return name
    return "Unknown"


def parse_portal(pdf_path) -> list[dict]:
    lines = _clean_lines(pdf_path)
    bold_names = _bold_company_names(pdf_path)
    entry_idx = [i for i, (l, _) in enumerate(lines) if ENTRY_RE.match(l)]

    headers = []
    prev = 0
    for e in entry_idx:
        # Normal card: <company> <role...> "Placement 2025-26" "Willing Candidates" "<n> Eligibility Criteria".
        # Around page breaks the role/type lines sometimes come later in the stream; then header ends at e.
        type_idx = e - 2 if lines[e - 1][0] == "Willing Candidates" else e
        found = _find_header(lines, type_idx, prev, bold_names)
        if found is None and type_idx < e:
            # no font match (a few pages use other fonts): company = first header line after "Last Updated"
            start = next((j + 1 for j in range(type_idx - 1, max(prev, type_idx - 8) - 1, -1)
                          if lines[j][0].startswith("Last Updated")), None)
            if start is not None and start < type_idx:
                found = (start, start + 1)
        if found is None:
            type_idx = e
            found = (e, e)
        cstart, rstart = found
        headers.append((cstart, rstart, type_idx))
        prev = e + 1

    records = []
    for k, e in enumerate(entry_idx):
        cstart, rstart, type_idx = headers[k]
        body_end = headers[k + 1][0] if k + 1 < len(entry_idx) else len(lines)
        company = " ".join(l for l, _ in lines[cstart:rstart])
        role = " ".join(l for l, _ in lines[rstart:type_idx]).replace("- ", "-").strip()
        drive_type = lines[type_idx][0] if type_idx < e else ""
        body_lines = [l for l, _ in lines[e + 1:body_end]]
        if not company:
            company = _company_from_body(body_lines, bold_names)
        if type_idx == e and "Willing Candidates" in body_lines:
            # role + type were pushed below the eligibility block: recover them from the body
            w = body_lines.index("Willing Candidates")
            t = w - 1
            drive_type = body_lines[t] if t >= 0 and "2025" in body_lines[t] else ""
            r = t
            while r > 0 and r > t - 4 and not re.search(r":|^PA$|INR|^[\d,]+$|Criteria|Details|Remarks|Last Updated|CTC Take|^Home$|Description",
                                                         body_lines[r - 1]):
                r -= 1
            rec_lines = [x for x in body_lines[r:t] if x.strip()]
            if company == "Unknown" and len(rec_lines) >= 2:  # no font info: first header line = company
                company, rec_lines = rec_lines[0], rec_lines[1:]
            if not role:
                role = " ".join(rec_lines).strip()
            del body_lines[r:w + 1]
        pages = sorted({p for _, p in lines[cstart:body_end]})
        last_updated = next(
            (l.replace("Last Updated:", "").strip() for l, _ in reversed(lines[max(0, cstart - 2):cstart])
             if l.startswith("Last Updated")), None)
        m = re.search(r"\s*(Placement 20\d\d-\d\d)$", role)
        if m:  # type line glued to the role line
            role, drive_type = role[:m.start()].strip(), m.group(1)
        rec = _parse_body(body_lines)
        rec.update(
            entry_no=int(ENTRY_RE.match(lines[e][0]).group(1)), company=company, role=role or "Not specified",
            drive_type=drive_type, pages=pages, last_updated=last_updated,
        )
        records.append(rec)
    return records


def _parse_body(body_lines: list[str]) -> dict:
    text = "\n".join(body_lines)
    flat = re.sub(r"\s+", " ", text)

    eligibility = []
    depts_found = [d.split() for d in DEPT_RE.findall(flat)]
    for i, m in enumerate(ELIG_RE.finditer(flat)):
        x, xii, cgpa, courses = m.groups()
        depts = depts_found[i] if i < len(depts_found) else []
        eligibility.append({
            "x_pct": float(x), "xii_pct": float(xii), "cgpa": float(cgpa),
            "courses": courses.split(), "departments": depts,
        })
    bl = BACKLOG_RE.search(flat)

    status = next((l.split(":", 1)[1].strip() for l in body_lines if l.startswith("Status :")), None)
    deadline = None
    for i, l in enumerate(body_lines):
        if l.startswith("Willingness Deadline"):
            deadline = _parse_deadline(" ".join(body_lines[i:i + 2]))
            break
    slot_date = slot_no = None
    m = SLOT_RE.search(flat)
    if m:
        mon, d, y, s = m.groups()
        slot_date = f"{int(y):04d}-{MONTHS[mon[:3]]:02d}-{int(d):02d}"
        slot_no = int(s)
    timing = next((l.split(":", 1)[1].strip() for l in body_lines if l.startswith("Timing:")), "") or None

    # CTC table lives between "CTC Take Home" and the JD link
    ctc_part = flat.split("CTC Take", 1)[-1]
    jd_split = re.split(r"JD : Job Description", ctc_part, maxsplit=1)
    ctc_text = jd_split[0]
    ctc = []
    for m in CTC_RE.finditer(ctc_text):
        course, cur, amt, cur2, amt2 = m.groups()
        ctc.append({"course": course, "currency": cur, "ctc": _to_number(amt),
                    "take_home": _to_number(amt2) if amt2 else None})

    for c in ctc:  # a few cards have garbled numbers ("2200.86"); drop values that can't be annual pay
        if c["ctc"] is not None and c["ctc"] < 50000:
            c["ctc"] = None

    jd_file, remarks = None, ""
    if len(jd_split) > 1:
        after = jd_split[1]
        fm = re.search(r"filename=\s*([^)]*?)\s*\)", after)
        if fm:
            jd_file = fm.group(1).replace(" ", "")
            remarks = after[fm.end():]
        else:
            remarks = after
    remarks = re.sub(r"Last Updated:.*?[ap]\.m\.", "", remarks)
    remarks = re.sub(r"\s*(\.\.\.)?\s*Read\s*More\s*$", " ...", remarks.strip()).strip()

    ctc_source = "table" if any(c["ctc"] for c in ctc) else None
    if not ctc_source:  # many cards say "Refer Description" and put the CTC in the remarks
        m = re.search(r"(?:Total\s*CTC|CTC)\s*(?:\(In INR\))?\s*[:\-]*\s*(?:INR|₹|Rs\.?)\s*([\d,]{6,}?)(?=1st|\s|[A-Za-z(]|$)", remarks)
        if m and _to_number(m.group(1)) and _to_number(m.group(1)) >= 50000:
            ctc = [{"course": "All", "currency": "INR", "ctc": _to_number(m.group(1)), "take_home": None}]
            ctc_source = "remarks"

    loc = None
    lm = re.search(r"Location(?:\(s\))?\s*(?:/\s*Place of Posting)?\s*[:–-]?\s*(.+?)(?=\s+(?:Provident|Joining|Other|Read|1st|Relocation|Stocks?|Monthly|Basic|CTC|Base|Performance|Retention|Sign|ESOP|RSU|\.\.\.)|$)",
                   remarks)
    if lm:
        loc = lm.group(1).strip(" ,;:-")[:60] or None

    return {
        "eligibility": eligibility,
        "active_backlog": bl.group(1) if bl else None,
        "total_backlog": bl.group(2) if bl else None,
        "status": status, "willingness_deadline": deadline,
        "slot_date": slot_date, "slot_no": slot_no, "slot_timing": timing,
        "ctc": ctc, "ctc_source": ctc_source, "jd_file": jd_file, "remarks": remarks, "location": loc,
    }


if __name__ == "__main__":
    import json
    import sys

    from placement_agent.config import PORTAL_PDF

    recs = parse_portal(sys.argv[1] if len(sys.argv) > 1 else PORTAL_PDF)
    print(len(recs))
    for r in recs[:5]:
        print(json.dumps(r, indent=1)[:1500])
