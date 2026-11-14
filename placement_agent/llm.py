"""Swappable chat model. LLM_PROVIDER=none -> offline rule-based mode (no API key needed)."""
from __future__ import annotations

import warnings
from functools import lru_cache

from placement_agent import config

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-3.5-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
}


@lru_cache(maxsize=1)
def get_llm():
    if config.LLM_PROVIDER in ("", "none", "offline"):
        return None
    from langchain.chat_models import init_chat_model

    model = config.LLM_MODEL or DEFAULT_MODELS.get(config.LLM_PROVIDER)
    if not model:
        raise ValueError(f"Set LLM_MODEL for provider '{config.LLM_PROVIDER}'")
    # some Gemini models ignore temperature and warn on every call
    warnings.filterwarnings("ignore", message=r".*fixed sampling defaults.*")
    kwargs = {}
    if config.LLM_PROVIDER in ("openai", "google_genai", "groq"):
        # fail fast instead of silently retrying for minutes on quota/network errors
        kwargs = {"timeout": config.LLM_TIMEOUT, "max_retries": config.LLM_MAX_RETRIES}
    return init_chat_model(model, model_provider=config.LLM_PROVIDER, temperature=0, **kwargs)
