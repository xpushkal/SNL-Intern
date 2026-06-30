"""Turn orchestrator (skeleton).

Step 1 ships a deterministic, always-valid placeholder so the API contract and the
safety wrapper are testable end-to-end. Steps 3-4 replace the body with the real
retrieval + understand/dispatch/render pipeline. The signature stays fixed so
``main.py`` never changes again.
"""
from __future__ import annotations

from app.data.catalog import load_catalog
from app.responder import build_response
from app.schemas import ChatResponse


def run_turn(messages: list[dict]) -> ChatResponse:
    """Process one stateless turn and return a schema-valid response.

    ``messages`` is the full conversation history: ``[{"role": ..., "content": ...}]``.
    """
    catalog = load_catalog()
    # Placeholder behaviour: ask one grounding question. Replaced in Step 4.
    return build_response(
        reply="Happy to help find SHL assessments. What role or skills are you hiring for?",
        items=[],
        end_of_conversation=False,
        catalog=catalog,
    )
