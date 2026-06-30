"""Turn orchestrator: run one stateless turn through the LangGraph pipeline.

The signature is fixed so ``main.py`` never changes. Any internal failure degrades to
a schema-valid safe response (defense-in-depth behind main.py's outer wrapper).
"""
from __future__ import annotations

import logging

from app.responder import safe_fallback_response
from app.schemas import ChatResponse

log = logging.getLogger("shl.turn")


def run_turn(messages: list[dict]) -> ChatResponse:
    """Process one stateless turn and return a schema-valid response.

    ``messages`` is the full conversation history: ``[{"role": ..., "content": ...}]``.
    """
    try:
        from app.agent.graph import GRAPH

        state = GRAPH.invoke({"messages": messages or []})
        response = state.get("response")
        if isinstance(response, ChatResponse):
            return response
        log.error("graph produced no response object")
    except Exception:
        log.exception("run_turn failed")
    return safe_fallback_response()
