"""The understand node: distill the whole conversation into a structured route.

Primary path = one Groq JSON call. If the key is absent, the call fails, or the JSON
is malformed, we fall back to a deterministic interpretation so the turn always
produces a valid, sensible route (never a crash, never a wrong-shaped object).
"""
from __future__ import annotations

import json
import logging
import re

from app.agent import groq_client
from app.agent.prompts import UNDERSTAND_SYSTEM
from app.agent.state import empty_understanding

log = logging.getLogger("shl.understand")

# --- deterministic signal vocabulary -------------------------------------
_ROLE_SKILL = {
    "developer", "engineer", "manager", "analyst", "designer", "sales", "nurse",
    "accountant", "consultant", "scientist", "administrator", "technician", "lead",
    "director", "executive", "graduate", "intern", "clerk", "cashier", "agent",
    "specialist", "officer", "coordinator", "supervisor", "programmer", "architect",
    "recruiter", "marketer", "teacher", "support", "leadership", "java", "python",
    "javascript", "sql", "aws", "react", "excel", "cognitive", "personality",
    "numerical", "verbal", "coding", "technical", "clerical", "data",
}
_JD_HINTS = ("experience", "responsibilities", "requirements", "skills", "proficient", "years")
_DONE = ("that's all", "thats all", "that is all", "we're done", "were done", "all set",
         "perfect", "looks good", "that works", "sounds good", "thank you", "thanks",
         "no more", "that's everything", "we are good", "good to go")
_COMPARE = ("difference between", "compare", " vs ", " versus ", "differ")
_REFINE = ("remove", "drop", "without", "instead", "replace", "add ", "also include",
           "swap", "take out", "exclude", "shorter", "make them", "keep only")
_INJECTION = ("ignore previous", "ignore all previous", "disregard", "system prompt",
              "you are now", "reveal your", "override")
_OFFTOPIC = ("weather", "recipe", "stock price", "tell me a joke", "who won",
             "write me a poem", "lawsuit", "is it legal", "sue ")


def last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def all_user_text(messages: list[dict]) -> str:
    return " ".join(
        (m.get("content") or "") for m in messages if m.get("role") == "user"
    ).strip()


def is_vague(text: str) -> bool:
    """True when there is no role/skill/JD to retrieve on."""
    t = (text or "").lower()
    if any(s in t for s in _ROLE_SKILL):
        return False
    if any(h in t for h in _JD_HINTS):
        return False
    return len(re.findall(r"[a-z0-9]+", t)) <= 5


def _normalize(u: dict) -> dict:
    base = empty_understanding()
    if isinstance(u, dict):
        for k in base:
            if k in u and u[k] is not None:
                base[k] = u[k]
    base["hard"] = base.get("hard") or {}
    base["soft"] = base.get("soft") or {}
    for k in ("compare_names", "remove_names"):
        if not isinstance(base.get(k), list):
            base[k] = []
    base["in_scope"] = bool(base.get("in_scope", True))
    base["user_done"] = bool(base.get("user_done", False))
    if base["intent"] not in {"clarify", "recommend", "refine", "compare", "refuse"}:
        base["intent"] = "recommend"
    return base


def deterministic_understand(messages: list[dict]) -> dict:
    """Keyword-based fallback route. Conservative and recall-safe."""
    u = empty_understanding()
    text = last_user(messages).lower()
    u["search_query"] = all_user_text(messages)

    if any(p in text for p in _INJECTION) or any(p in text for p in _OFFTOPIC):
        u["in_scope"] = False
        u["intent"] = "refuse"
        return u

    if any(p in text for p in _DONE) and not any(p in text for p in _REFINE):
        u["user_done"] = True

    if any(p in text for p in _COMPARE):
        u["intent"] = "compare"
        u["compare_names"] = _extract_compare_names(last_user(messages))
        return u

    has_prior_list = _has_prior_shortlist(messages)
    if has_prior_list and any(p in text for p in _REFINE):
        u["intent"] = "refine"
        u["remove_names"] = _extract_remove_names(last_user(messages))
        m = re.search(r"add (?:a |an |some )?(.+?)(?: test| assessment|s)?$", text)
        if "add" in text and m:
            u["add_query"] = m.group(1).strip()
        return u

    if u["user_done"]:
        u["intent"] = "refine"  # re-render the confirmed shortlist with end=true
        return u

    if is_vague(last_user(messages)) and not has_prior_list:
        u["intent"] = "clarify"
        u["clarifying_question"] = (
            "Happy to help. What role or key skills are you hiring for?"
        )
        return u

    u["intent"] = "recommend"
    return u


def _extract_compare_names(text: str) -> list[str]:
    t = re.sub(r"(?i)what(?:'s| is) the difference between|difference between|compare|versus|\bvs\b", "|", text)
    parts = re.split(r"\||,| and ", t)
    return [p.strip(" ?.") for p in parts if len(p.strip(" ?.")) >= 2][:4]


def _extract_remove_names(text: str) -> list[str]:
    m = re.search(r"(?i)(?:remove|drop|without|exclude|take out)\s+(?:the\s+)?(.+?)(?:[.,]|$)", text)
    return [m.group(1).strip()] if m else []


def _has_prior_shortlist(messages: list[dict]) -> bool:
    from app.agent.render import parse_prior_shortlist

    return bool(parse_prior_shortlist(messages))


def understand(messages: list[dict]) -> dict:
    """LLM-first understanding with deterministic fallback."""
    if not groq_client.available():
        return deterministic_understand(messages)
    convo = json.dumps(messages, ensure_ascii=False)
    for attempt in range(2):
        try:
            raw = groq_client.chat_json(UNDERSTAND_SYSTEM, convo)
            return _normalize(raw)
        except Exception as exc:  # parse error / timeout / API error
            log.warning("understand LLM attempt %d failed: %s", attempt + 1, exc)
    return deterministic_understand(messages)
