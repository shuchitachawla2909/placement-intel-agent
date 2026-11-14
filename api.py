"""FastAPI service + demo chat page.

  uvicorn api:app --reload      -> open http://127.0.0.1:8000
  POST /chat     {"message": "...", "thread_id": "me"}
  GET  /profile/{thread_id}      PUT /profile/{thread_id} {"cgpa": 9.7, "branch": "cse"}
  GET  /health
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from placement_agent import config
from placement_agent.graph import PlacementAgent
from placement_agent.retriever import get_retriever

app = FastAPI(title="Placement Intel Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "web"), name="static")
agent = PlacementAgent()


class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = "default"
    mode: str = Field("auto", pattern="^(auto|llm|offline)$")  # offline = rule-based, no LLM calls


class ProfileIn(BaseModel):
    cgpa: float | None = Field(None, ge=0, le=10)
    branch: str | None = None
    course: str | None = None
    preferred_roles: list[str] | None = None
    strengths: str | None = None


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/health")
def health():
    llm = config.LLM_PROVIDER if config.LLM_PROVIDER in ("", "none") else f"{config.LLM_PROVIDER}:{config.LLM_MODEL or 'default'}"
    return {"status": "ok", "llm": llm or "none", "llm_available": config.LLM_PROVIDER not in ("", "none"),
            "retrieval": get_retriever().mode,
            "documents": len(get_retriever().docs)}


@app.post("/chat")
def chat(body: ChatIn):
    try:
        if body.mode == "llm" and config.LLM_PROVIDER in ("", "none"):
            raise HTTPException(400, "No LLM configured. Set LLM_PROVIDER and the API key in .env, then restart.")
        return agent.ask(body.message, thread_id=body.thread_id, use_llm=False if body.mode == "offline" else None)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@app.get("/threads")
def threads():
    return {"threads": agent.threads()}


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    agent.delete_thread(thread_id)
    return {"deleted": thread_id}


@app.get("/history/{thread_id}")
def history(thread_id: str):
    return {"thread_id": thread_id, "turns": agent.history(thread_id)}


@app.get("/profile/{thread_id}")
def get_profile(thread_id: str):
    return agent.get_profile(thread_id)


@app.put("/profile/{thread_id}")
def put_profile(thread_id: str, body: ProfileIn):
    return agent.set_profile(thread_id, body.model_dump())
