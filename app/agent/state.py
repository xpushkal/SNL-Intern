"""Shared agent state (LangGraph passes this dict between nodes)."""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict]       # full conversation history (stateless input)
    understanding: dict        # output of the understand node
    items: list[dict]          # selected catalog records (recommend/refine/compare)
    reply: str                 # final templated reply text
    end: bool                  # end_of_conversation
    response: Any              # assembled ChatResponse (set by the respond node)


# The structured contract produced by the understand node. Documented here so the
# deterministic fallback and the LLM path stay in lock-step.
UNDERSTANDING_KEYS = (
    "in_scope",            # bool: about SHL assessment selection?
    "intent",             # clarify | recommend | refine | compare | refuse
    "search_query",       # focused retrieval query (role/skills/JD distilled)
    "hard",               # {max_duration_minutes, languages[], test_types[]}
    "soft",               # {job_levels[], test_types[]}
    "compare_names",      # [str] assessments to compare
    "remove_names",       # [str] items to drop from the current shortlist (refine)
    "add_query",          # str|None: capability to add to the shortlist (refine)
    "clarifying_question",# str|None
    "user_done",          # bool: explicit completion ("thanks, that's all")
)


def empty_understanding() -> dict:
    return {
        "in_scope": True,
        "intent": "recommend",
        "search_query": "",
        "hard": {},
        "soft": {},
        "compare_names": [],
        "remove_names": [],
        "keep_only_names": [],   # refine: restrict the shortlist to these items
        "add_queries": [],       # refine: capabilities/products to add (one item each)
        "clarifying_question": None,
        "user_done": False,
    }
