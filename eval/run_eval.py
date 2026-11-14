"""Evaluate the agent on hand-written Q&A pairs.

  python -m eval.run_eval                # full agent (offline rules or the configured LLM)
  python -m eval.run_eval --verify-gold  # first check every gold 'evidence' string exists in the raw PDFs

Metrics
- answer accuracy : all must_contain facts present (each item may list alternatives "a|b") and no must_not_contain
- route accuracy  : router picked the expected route
- retrieval hit@1 / hit@5 : semantic questions, gold company among the top-k search_docs hits (retriever only)
- citation validity : answer cites at least one [S#] and every cited tag exists in the returned sources
- latency p50 / p95 per question
Results are written to eval/results_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from placement_agent import config
from placement_agent.graph import PlacementAgent
from placement_agent.ingest.build import same_company
from placement_agent.retriever import get_retriever

HERE = Path(__file__).parent


def load_qa(name: str = "qa.jsonl"):
    return [json.loads(l) for l in (HERE / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def raw_text(pdf) -> str:
    from pypdf import PdfReader

    return " ".join(" ".join((p.extract_text() or "").split()) for p in PdfReader(pdf).pages)


def verify_gold(qa) -> int:
    portal, stats = raw_text(config.PORTAL_PDF), raw_text(config.STATS_PDF)
    bad = 0
    for q in qa:
        for ev in q.get("evidence", []):
            if ev not in portal:
                print(f"  [gold] {q['id']}: evidence not found in portal PDF: {ev!r}")
                bad += 1
        for ev in q.get("evidence_stats", []):
            if ev not in stats:
                print(f"  [gold] {q['id']}: evidence not found in stats PDF: {ev!r}")
                bad += 1
    print(f"Gold evidence check: {sum(len(q.get('evidence', [])) + len(q.get('evidence_stats', [])) for q in qa) - bad} ok, {bad} missing")
    return bad


def contains(answer: str, fact: str) -> bool:
    a = answer.lower()
    return any(alt.strip().lower() in a for alt in fact.split("|"))


def pct(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-gold", action="store_true")
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--set", default="qa.jsonl", help="qa.jsonl (dev set) or heldout.jsonl")
    args = ap.parse_args()

    qa = load_qa(args.set)
    if args.only:
        qa = [q for q in qa if q["id"] in args.only.split(",")]
    if args.verify_gold and verify_gold(qa):
        sys.exit("Fix the gold set first.")

    agent = PlacementAgent(MemorySaver())
    retr = get_retriever()
    mode = f"llm={config.LLM_PROVIDER}{(':' + config.LLM_MODEL) if config.LLM_MODEL else ''}, retrieval={retr.mode}"
    print(f"Mode: {mode}\n")

    rows, lat = [], []
    for q in qa:
        res = agent.ask(q["question"], thread_id=f"eval-{uuid.uuid4().hex[:8]}")
        ans = res["answer"]
        missing = [f for f in q.get("must_contain", []) if not contains(ans, f)]
        forbidden = [f for f in q.get("must_not_contain", []) if contains(ans, f)]
        cited = set(re.findall(r"\[S(\d+)\]", ans))
        valid_tags = {s["tag"][1:] for s in res["sources"]}
        row = {"id": q["id"], "type": q["type"], "correct": not missing and not forbidden,
               "route_ok": res["route"] == q["route"], "route": res["route"],
               "citations_ok": bool(cited) and cited <= valid_tags,
               "missing": missing, "forbidden": forbidden, "latency_s": res["latency_s"],
               "tools": [t["tool"] for t in res["tools_used"]]}
        if q.get("gold_company"):
            hits = retr.search(q["question"], k=5)
            comps = [h["metadata"].get("company") or "" for h in hits]
            row["hit@1"] = any(same_company(g, c) for g in q["gold_company"] for c in comps[:1])
            row["hit@5"] = any(same_company(g, c) for g in q["gold_company"] for c in comps)
        rows.append(row)
        lat.append(res["latency_s"])
        flag = "PASS" if row["correct"] else "FAIL"
        print(f"{flag} {q['id']:<3} route={res['route']:<11} {res['latency_s']:>6.2f}s  {q['question'][:70]}"
              + (f"  missing={missing}" if missing else "") + (f"  forbidden={forbidden}" if forbidden else ""))

    sem = [r for r in rows if "hit@5" in r]
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["correct"])
    summary = {
        "mode": mode, "n_questions": len(rows),
        "answer_accuracy_pct": pct([r["correct"] for r in rows]),
        "route_accuracy_pct": pct([r["route_ok"] for r in rows]),
        "citation_validity_pct": pct([r["citations_ok"] for r in rows]),
        "retrieval_hit@1_pct": pct([r["hit@1"] for r in sem]), "retrieval_hit@5_pct": pct([r["hit@5"] for r in sem]),
        "n_retrieval_questions": len(sem),
        "latency_p50_s": round(statistics.median(lat), 3),
        "latency_p95_s": round(sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)], 3),
        "accuracy_by_type_pct": {k: pct(v) for k, v in by_type.items()},
        "docs_indexed": len(retr.docs),
    }
    print("\n" + json.dumps(summary, indent=2))
    tag = "offline" if config.LLM_PROVIDER in ("", "none") else config.LLM_PROVIDER
    summary["set"] = args.set
    (HERE / f"results_{tag}_{Path(args.set).stem}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
