"""Parse the T&P cell's "Placement Statistics Till Day N" PDF: one row per drive phase/day with
offers, profiles and the companies that visited in that phase."""
from __future__ import annotations

import re

import pdfplumber

LABEL_MAX_X, OFFERS_MAX_X, PROFILES_MAX_X = 130, 200, 248  # column boundaries (pt)
SKIP = re.compile(r"Placement Drive|^No\. of|^Offers|December \d|Serving the nation|TRAINING AND|Prof\.|Coordinator|"
                  r"Training and Placement|Hkkjrh|Indian Institute|Phone:")


def _lines(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            rows = {}
            for w in page.extract_words(x_tolerance=2):
                if 115 < w["top"] < 770:
                    rows.setdefault(round(w["top"] / 4), []).append(w)
            for k in sorted(rows):
                yield sorted(rows[k], key=lambda w: w["x0"])


def _split_companies(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).replace(" etc.", "").replace("etc.", "")
    out, depth, cur = [], 0, ""
    for ch in text:  # split on commas that are not inside parentheses
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth <= 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return [c for c in out if c]


def _day_range(label: str) -> tuple[int | None, int | None]:
    days = [int(d) for d in re.findall(r"Day (\d+)", label)]
    if "&" in label and re.search(r"Day (\d+) & (\d+)", label):
        a, b = map(int, re.search(r"Day (\d+) & (\d+)", label).groups())
        days += [a, b]
    return (min(days), max(days)) if days else (None, None)


def parse_stats(pdf_path) -> dict:
    rounds, section, cur, summary = [], "Pre-placement (internships & PPOs)", None, {}
    for words in _lines(pdf_path):
        text = " ".join(w["text"] for w in words)
        if SKIP.search(text):
            continue
        label = " ".join(w["text"] for w in words if w["x0"] < LABEL_MAX_X)
        offers = [w["text"] for w in words if LABEL_MAX_X <= w["x0"] < OFFERS_MAX_X and re.fullmatch(r"\d+", w["text"])]
        profiles = [w["text"] for w in words if OFFERS_MAX_X <= w["x0"] < PROFILES_MAX_X and re.fullmatch(r"\d+", w["text"])]
        comp = " ".join(w["text"] for w in words if w["x0"] >= PROFILES_MAX_X)

        if label.startswith("Total") and offers:
            summary = {"total_offers": int(offers[0]), "total_profiles": int(profiles[0])}
            m = re.search(r"Highest CTC:\s*([\d.]+)\s*Cr", comp)
            if m:
                summary["highest_ctc_lpa"] = float(m.group(1)) * 100
            m = re.search(r"Average CTC:\s*([\d.]+)\s*LPA", comp)
            if m:
                summary["average_ctc_lpa"] = float(m.group(1))
            continue
        if offers:  # a new row starts
            cur = {"label": label, "section": section, "offers": int(offers[0]),
                   "profiles": int(profiles[0]) if profiles else None, "companies_text": comp}
            rounds.append(cur)
        elif label and not comp:  # e.g. "Final Placement" section header
            section = label
        elif cur is not None:
            if label:
                cur["label"] += " " + label
            cur["companies_text"] += " " + comp

    for r in rounds:
        r["label"] = re.sub(r"- ", "-", re.sub(r"\bInternship\b$", "", r["label"])).strip()
        r["companies"] = _split_companies(r.pop("companies_text"))
        r["day_from"], r["day_to"] = _day_range(r["label"])
    return {"rounds": rounds, "summary": summary}


if __name__ == "__main__":
    import json

    from placement_agent.config import STATS_PDF

    print(json.dumps(parse_stats(STATS_PDF), indent=1)[:4000])
