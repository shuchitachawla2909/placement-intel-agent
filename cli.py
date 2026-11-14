"""Interactive CLI.   python cli.py            (chat; Ctrl+C cancels a slow question)
                     python cli.py "question"  (one-shot)"""
import sys
import time
import uuid

from placement_agent import config
from placement_agent.graph import PlacementAgent

STEP_LABELS = {"profile": "reading your profile", "router": "routing", "research": "planning tool calls",
               "tools": "running tools", "answer": "writing answer", "prep_coach": "building prep plan"}


def progress(t0):
    def on_step(node, update):
        detail = ""
        if node == "router" and update.get("route"):
            detail = f" -> {update['route']}"
        if node == "research":
            calls = [tc["name"] for m in update.get("messages", []) for tc in (getattr(m, "tool_calls", None) or [])]
            detail = f" -> {', '.join(calls)}" if calls else " -> done"
        print(f"  [{time.perf_counter() - t0:5.1f}s] {STEP_LABELS.get(node, node)}{detail}", flush=True)
    return on_step


def show(res: dict):
    print("\n" + res["answer"])
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['tag']}] {s['source']}")
    print(f"\n(route={res['route']}, tools={[t['tool'] for t in res['tools_used']]}, {res['latency_s']}s)")


def ask(agent, q, thread):
    try:
        show(agent.ask(q, thread_id=thread, on_step=progress(time.perf_counter())))
    except KeyboardInterrupt:
        print("\n(cancelled)")


def main():
    agent = PlacementAgent()
    mode = "offline rules" if config.LLM_PROVIDER in ("", "none") else f"{config.LLM_PROVIDER}:{config.LLM_MODEL or 'default'}"
    if len(sys.argv) > 1:
        ask(agent, " ".join(sys.argv[1:]), f"cli-{uuid.uuid4().hex[:6]}")
        return
    thread = "cli"
    print(f"Placement Intel Agent [{mode}] - ask about companies, CTC, eligibility, drive days.")
    print("Type 'exit' to quit; Ctrl+C cancels a question that is taking too long.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q:
            ask(agent, q, thread)


if __name__ == "__main__":
    main()
