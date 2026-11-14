"""LangGraph agent:

  START -> profile -> router -> research <-> tools -> answer -> (prep_coach) -> END

- profile:   pulls CGPA / branch / course / role preferences out of the message into persistent state
- router:    picks a route (LLM JSON output, or rules when offline / unparsable)
- research:  ReAct-style tool calling (LLM with bound tools; offline: rule planner emits the calls)
- tools:     LangGraph ToolNode executing the 5 tools
- answer:    grounded answer with [S#] citations built from tool outputs
- prep_coach: second agent that turns eligible openings into a prioritised prep plan (handoff on prep_plan)
State (messages + profile) is persisted per thread_id by a checkpointer (SQLite if available).
"""
from __future__ import annotations

import json
import operator
import re
import sys
import time
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from placement_agent import formatting as F
from placement_agent import prompts, rules
from placement_agent.llm import get_llm
from placement_agent.tools import TOOLS

MAX_TOOL_ROUNDS = 4
HISTORY_TURNS = 3


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    profile: dict
    route: str
    answer: str
    sources: list
    tools_used: list
    turns: Annotated[list, operator.add]  # chat transcript for the UI: one entry per question


# ---------------------------------------------------------------- helpers
def _turn_messages(messages: list) -> list:
    """Messages since the latest human message (inclusive)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _recent_history(messages: list, turns: int = HISTORY_TURNS) -> list:
    """Last N turns, cut at a HumanMessage so tool-call/tool-result pairs stay intact."""
    idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    start = idx[-turns] if len(idx) >= turns else 0
    return messages[start:]


def _llm_history(messages: list) -> list:
    """Recent turns for the LLM; tool outputs from earlier turns are shortened to save context."""
    hist = _recent_history(messages)
    current = _turn_messages(messages)
    cut = len(hist) - len(current)
    out = []
    for i, m in enumerate(hist):
        if i < cut and isinstance(m, ToolMessage) and len(m.content) > 1500:
            m = ToolMessage(content=m.content[:1500] + " ...[truncated]", tool_call_id=m.tool_call_id, name=m.name)
        out.append(m)
    return out


def _question(state) -> str:
    return _turn_messages(state["messages"])[0].content


def _tool_results(state) -> list[tuple[str, dict, Any]]:
    turn = _turn_messages(state["messages"])
    calls = {}
    for m in turn:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                calls[tc["id"]] = (tc["name"], tc["args"])
    out = []
    for m in turn:
        if isinstance(m, ToolMessage):
            name, args = calls.get(m.tool_call_id, (m.name, {}))
            try:
                out.append((name, args, json.loads(m.content)))
            except (json.JSONDecodeError, TypeError):
                out.append((name, args, {"error": str(m.content)}))
    return out


def _text(msg) -> str:
    """Plain text of a model reply (some providers return a list of content blocks)."""
    t = getattr(msg, "text", None)
    if isinstance(t, str) and t:  # langchain-core >= 1: str-like property
        return str(t)
    if callable(t):
        t = t()
        if isinstance(t, str) and t:
            return t
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _warn(where: str, e: Exception) -> str:
    note = f"LLM call failed in {where} ({type(e).__name__}: {str(e)[:160]}); used the offline fallback."
    print(f"[warn] {note}", file=sys.stderr)
    return note


def _offline_calls(state) -> dict:
    _, plan = rules.route_and_plan(_question(state), state.get("profile", {}))
    calls = [{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"} for n, a in plan]
    return {"messages": [AIMessage(content="", tool_calls=calls)]}


def _llm(config: RunnableConfig | None):
    """The chat model for this run, or None for offline mode.
    A request can force offline mode with configurable={"use_llm": False}."""
    if config and (config.get("configurable") or {}).get("use_llm") is False:
        return None
    return get_llm()


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- nodes
def profile_node(state: AgentState) -> dict:
    profile = dict(state.get("profile") or {})
    profile.update(rules.extract_profile(_question(state)))
    return {"profile": profile}


def router_node(state: AgentState, config: RunnableConfig = None) -> dict:
    q, profile = _question(state), state.get("profile", {})
    rule_route, _ = rules.route_and_plan(q, profile)
    llm = _llm(config)
    if llm is None:
        return {"route": rule_route}
    try:
        msg = llm.invoke([SystemMessage(prompts.ROUTER_PROMPT.format(data_note=prompts.DATA_NOTE, profile=profile)),
                          HumanMessage(q)])
    except Exception as e:  # network/quota/model errors -> rules
        _warn("router", e)
        return {"route": rule_route}
    route = _parse_json(_text(msg)).get("route")
    return {"route": route if route in rules.ROUTES else rule_route}


def research_node(state: AgentState, config: RunnableConfig = None) -> dict:
    turn = _turn_messages(state["messages"])
    rounds = sum(1 for m in turn if isinstance(m, AIMessage) and m.tool_calls)
    llm = _llm(config)
    if llm is None:  # offline: the rule planner issues all calls in one round
        return {} if rounds else _offline_calls(state)
    if rounds >= MAX_TOOL_ROUNDS:
        return {}
    sys_msg = SystemMessage(prompts.RESEARCH_PROMPT.format(route=state.get("route"), data_note=prompts.DATA_NOTE,
                                                        profile=state.get("profile", {})))
    try:
        msg = llm.bind_tools(TOOLS).invoke([sys_msg] + _llm_history(state["messages"]))
    except Exception as e:
        _warn("research", e)
        return {} if rounds else _offline_calls(state)
    return {"messages": [msg]}


def after_research(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "answer"


def answer_node(state: AgentState, config: RunnableConfig = None) -> dict:
    results = _tool_results(state)
    src = F.Sources()
    for _, _, r in results:
        src.collect(r)
    tools_used = [{"tool": n, "args": a} for n, a, _ in results]
    llm = _llm(config)
    profile = state.get("profile", {})
    if llm is not None and results:
        compact = json.dumps([{"tool": n, "args": a, "result": r} for n, a, r in results], ensure_ascii=False,
                             default=str)[:24000]
        try:
            msg = llm.invoke([SystemMessage(prompts.ANSWER_PROMPT.format(
                data_note=prompts.DATA_NOTE, profile=profile, results=compact, sources=src.listing())),
                HumanMessage(_question(state))])
            text = _text(msg)
        except Exception as e:
            note = _warn("answer", e)
            text = render_offline(_question(state), state.get("route", ""), results, src) + f"\n\n_({note})_"
    else:
        text = render_offline(_question(state), state.get("route", ""), results, src)
    cleanup = [RemoveMessage(id=m.id) for m in _turn_messages(state["messages"])
               if isinstance(m, AIMessage) and not m.tool_calls and _text(m).strip() == "DONE" and m.id]
    return {"messages": cleanup + [AIMessage(content=text)], "answer": text, "sources": src.as_list(),
            "tools_used": tools_used}


def after_answer(state: AgentState) -> str:
    return "prep_coach" if state.get("route") == "prep_plan" else "finalize"


def finalize_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """Append this question and its final answer to the transcript (shown when the chat page reloads)."""
    return {"turns": [{
        "question": _question(state), "answer": state.get("answer", ""), "sources": state.get("sources", []),
        "route": state.get("route"), "tools": [t["tool"] for t in state.get("tools_used", [])],
        "mode": "llm" if _llm(config) is not None else "offline",
        "time": time.strftime("%Y-%m-%d %H:%M"), "ts": time.time(),
    }]}


def prep_coach_node(state: AgentState, config: RunnableConfig = None) -> dict:
    """Second agent: converts the research results into a prioritised prep plan (multi-agent handoff)."""
    results = _tool_results(state)
    rows = [r for n, _, res in results if n in ("eligibility_check", "query_placements_db") for r in res.get("rows", [])]
    counts = next((res.get("per_day_counts") for n, _, res in results if res.get("per_day_counts")), None)
    if not rows:
        return {}
    profile = state.get("profile", {})
    llm = _llm(config)
    src = F.Sources()
    if llm is not None:  # the LLM answer may already cite [S#] tags - keep numbering consistent
        for s in state.get("sources", []):
            src.tag(s["source"])
    if llm is not None:
        for r in rows:
            src.tag(r["source"])
        try:
            msg = llm.invoke([SystemMessage(prompts.PREP_COACH_PROMPT.format(
                data_note=prompts.DATA_NOTE, profile=profile, rows=json.dumps(rows, default=str)[:20000],
                sources=src.listing())), HumanMessage(_question(state))])
            plan = _text(msg)
        except Exception as e:
            _warn("prep_coach", e)
            src = F.Sources()
            plan = F.prep_plan(rows, profile, src, counts=counts)
    else:
        plan = F.prep_plan(rows, profile, src, counts=counts)
    text = state.get("answer", "") + "\n\n---\n" + plan
    return {"messages": [AIMessage(content=plan, name="prep_coach")], "answer": text, "sources": src.as_list()}


# ---------------------------------------------------------------- offline renderer
def render_offline(question: str, route: str, results: list, src: F.Sources) -> str:
    if not results:
        return "I couldn't map that to the placement data. Try naming a company, a CTC range, a role type or a drive day."
    parts = []
    for name, args, res in results:
        if "error" in res:
            parts.append(f"Tool {name} failed: {res['error']}")
        elif name in ("query_placements_db", "eligibility_check"):
            crit = ", ".join(f"{k}={v}" for k, v in args.items() if v not in (None, "") and k != "limit")
            if not res["rows"]:
                parts.append(f"No openings match ({crit}). " + " ".join(res.get("notes", [])))
                continue
            if route == "prep_plan":  # the prep coach renders the details
                counts = ", ".join(f"Day {d}: {n}" if d != "not announced" else f"day not announced: {n}"
                                   for d, n in res.get("per_day_counts", {}).items())
                parts.append(f"You are eligible for **{res['count']} openings** ({crit})." + (f" By interview day - {counts}." if counts else ""))
                continue
            head = f"**{res['count']} openings** match ({crit})" + (
                f"; showing {res['shown']}." if res["shown"] < res["count"] else ".")
            parts.append(head + "\n\n" + F.rows_table(res["rows"], src) +
                         ("\n\n_" + " ".join(res["notes"]) + "_" if res.get("notes") else ""))
        elif name == "get_company_profile":
            parts.append(F.profile_block(res, src))
        elif name == "drive_stats":
            parts.append(F.stats_block(res, src, company=args.get("company")))
        elif name == "search_docs":
            if route in ("company", "compare"):  # profile already shown; only surface extra notes
                res = {"hits": [h for h in res["hits"] if not h["source"].startswith(("Placements'26", "Placement Statistics"))]}
                if not res["hits"]:
                    continue
            parts.append("Most relevant passages:\n\n" + F.search_block(res, src, query=question))
    if re.search(r"\boa\b|online assessment|pattern|interview round|questions asked|topics", question.lower()):
        has_notes = any(n == "search_docs" and any(h["source"].split(" p.")[0].lower().endswith((".md", ".txt"))
                                                   for h in r.get("hits", [])) for n, _, r in results)
        if not has_notes:
            parts.append("_Note: the 2025-26 portal/stat documents don't contain OA patterns or interview questions. "
                         "Add notes as data/raw/notes/<Company>.md (e.g. from the seniors' OA guide) and re-run ingestion "
                         "so the agent can answer this._")
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------- build + run
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("profile", profile_node)
    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("answer", answer_node)
    g.add_node("prep_coach", prep_coach_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "profile")
    g.add_edge("profile", "router")
    g.add_edge("router", "research")
    g.add_conditional_edges("research", after_research, {"tools": "tools", "answer": "answer"})
    g.add_edge("tools", "research")
    g.add_conditional_edges("answer", after_answer, {"prep_coach": "prep_coach", "finalize": "finalize"})
    g.add_edge("prep_coach", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def default_checkpointer():
    from placement_agent import config

    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        return SqliteSaver(sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False))
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


class PlacementAgent:
    def __init__(self, checkpointer=None):
        self.graph = build_graph(checkpointer if checkpointer is not None else default_checkpointer())

    def ask(self, question: str, thread_id: str = "default", on_step=None, use_llm: bool | None = None) -> dict:
        """on_step(node_name, update) is called after each graph step (used by the CLI to show progress).
        use_llm=False forces the offline rule-based mode for this question; None/True uses the configured LLM."""
        t0 = time.perf_counter()
        cfg = {"configurable": {"thread_id": thread_id}}
        if use_llm is False:
            cfg["configurable"]["use_llm"] = False
        for chunk in self.graph.stream({"messages": [HumanMessage(question)]}, cfg, stream_mode="updates"):
            if on_step:
                for node, update in chunk.items():
                    on_step(node, update or {})
        state = self.graph.get_state(cfg).values
        llm_used = use_llm is not False and get_llm() is not None
        return {"answer": state.get("answer", ""), "route": state.get("route"), "mode": "llm" if llm_used else "offline", "sources": state.get("sources", []),
                "tools_used": state.get("tools_used", []), "profile": state.get("profile", {}),
                "latency_s": round(time.perf_counter() - t0, 3)}

    def threads(self) -> list[dict]:
        """All saved conversations that have at least one answered question, newest first."""
        ids = []
        for cp in self.graph.checkpointer.list(None):
            tid = cp.config["configurable"]["thread_id"]
            if tid not in ids:
                ids.append(tid)
        out = []
        for tid in ids:
            turns = self.history(tid)
            if turns:
                out.append({"thread_id": tid, "title": turns[0]["question"][:80], "turns": len(turns),
                            "updated": turns[-1].get("time", ""), "ts": turns[-1].get("ts", 0)})
        return sorted(out, key=lambda t: t["ts"], reverse=True)

    def delete_thread(self, thread_id: str) -> None:
        self.graph.checkpointer.delete_thread(thread_id)

    def history(self, thread_id: str) -> list[dict]:
        snap = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return (snap.values or {}).get("turns", [])

    def get_profile(self, thread_id: str) -> dict:
        snap = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return (snap.values or {}).get("profile", {})

    def set_profile(self, thread_id: str, profile: dict) -> dict:
        cfg = {"configurable": {"thread_id": thread_id}}
        merged = {**self.get_profile(thread_id), **{k: v for k, v in profile.items() if v not in (None, "", [])}}
        self.graph.update_state(cfg, {"profile": merged}, as_node="profile")
        return merged
