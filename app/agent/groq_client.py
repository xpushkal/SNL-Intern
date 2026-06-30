"""Thin Groq wrapper: JSON-mode chat with timeout, one retry, and graceful absence.

If ``GROQ_API_KEY`` is unset the client is simply unavailable and callers fall back
to deterministic logic -- the service stays fully functional without an LLM.
"""
from __future__ import annotations

import json
import logging

from app import config

log = logging.getLogger("shl.groq")
_client = None


def available() -> bool:
    return bool(config.GROQ_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_TIMEOUT_S)
    return _client


def chat_json(system: str, user: str, *, temperature: float = 0.0) -> dict:
    """Return a parsed JSON object from the model. Raises on hard failure."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def chat_text(system: str, user: str, *, temperature: float = 0.2) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
