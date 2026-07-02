"""Turn orchestrator: run one stateless turn through the LangGraph pipeline.

The signature is fixed so ``main.py`` never changes. Any internal failure degrades to
a schema-valid safe response (defense-in-depth behind main.py's outer wrapper).
"""
from __future__ import annotations

import logging

from app import config
from app.responder import safe_fallback_response
from app.schemas import ChatResponse

log = logging.getLogger("shl.turn")


def _clamp_messages(messages: list[dict]) -> list[dict]:
    """Bound history length and per-message size so pathological input cannot pin the
    CPU in the embedding/BM25/regex path. Limits are far above any real conversation
    (the evaluator caps at 8 messages), so legitimate input is never altered."""
    out = []
    for m in (messages or [])[-config.MAX_MESSAGES:]:
        content = m.get("content") or ""
        if len(content) > config.MAX_CONTENT_CHARS:
            content = content[: config.MAX_CONTENT_CHARS]
        out.append({"role": m.get("role", "user"), "content": content})
    return out


def run_turn(messages: list[dict]) -> ChatResponse:
    """Process one stateless turn and return a schema-valid response.

    ``messages`` is the full conversation history: ``[{"role": ..., "content": ...}]``.
    """
    try:
        from app.agent.graph import GRAPH

        state = GRAPH.invoke({"messages": _clamp_messages(messages)})
        response = state.get("response")
        if isinstance(response, ChatResponse):
            return response
        log.error("graph produced no response object")
    except Exception:
        log.exception("run_turn failed")
    return safe_fallback_response()
