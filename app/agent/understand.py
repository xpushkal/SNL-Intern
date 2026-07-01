"""The understand node: distill the whole conversation into a structured route.

Primary path = one Groq JSON call. If the key is absent, the call fails, or the JSON
is malformed, we fall back to a deterministic interpretation so the turn always
produces a valid, sensible route (never a crash, never a wrong-shaped object).
"""
from __future__ import annotations

import json
import logging
import re

from app import config
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
_INJECTION = ("ignore previous", "ignore the above", "ignore all previous", "ignore your",
              "disregard", "system prompt", "your instructions", "you are now", "act as if",
              "pretend you", "jailbreak", "developer mode", "reveal your", "print your",
              "forget your instructions", "override your", "repeat the words above")
# Only unambiguous out-of-scope phrases. We bias HARD against false positives: refusing
# a valid hiring query ("a salaried manager", "a basketball coach") is worse than missing
# a novel off-topic phrasing, so bare ambiguous tokens are avoided.
_OFFTOPIC = ("weather", "recipe", "tell me a joke", "write a poem", "write me a poem",
             "who won the", "stock price", "capital of", "horoscope", "translate this",
             "is it legal", "legal advice", "lawsuit", "how much should i pay",
             "interview questions", "do my homework")


# Explicit duration cap ("under 20 minutes", "no more than 30 min"). Conservative
# phrasing only -> low false-positive risk before applying it as a HARD filter.
_DUR_RE = re.compile(
    r"(?:under|less than|at most|within|shorter than|no more than|max(?:imum)?(?: of)?)"
    r"\s+(\d{1,3})\s*(?:min|minute)"
)
# Seniority keyword -> SHL job level (SOFT ranking signal only, so approximate is fine).
_SENIORITY = {
    "graduate": "Graduate", "entry-level": "Entry-Level", "entry level": "Entry-Level",
    "junior": "Entry-Level", "mid-level": "Mid-Professional", "mid level": "Mid-Professional",
    "mid-professional": "Mid-Professional", "senior": "Professional Individual Contributor",
    "manager": "Manager", "director": "Director", "executive": "Executive",
    "supervisor": "Supervisor",
}


def extract_constraints(text: str) -> tuple[dict, dict]:
    """Deterministically pull a hard duration cap + soft seniority from free text."""
    t = (text or "").lower()
    hard: dict = {}
    if m := _DUR_RE.search(t):
        hard["max_duration_minutes"] = int(m.group(1))
    soft: dict = {}
    levels = sorted({lvl for kw, lvl in _SENIORITY.items() if kw in t})
    if levels:
        soft["job_levels"] = levels
    return hard, soft


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


def _guard(u: dict, messages: list[dict]) -> dict:
    """Reconcile the LLM route with deterministic signals to protect recall/turns."""
    # Don't waste a turn clarifying a query that clearly has a role/skill/JD.
    if u.get("intent") == "clarify" and u.get("in_scope", True):
        if not is_vague(last_user(messages)):
            u["intent"] = "recommend"
    return u


def _semantic_off_topic(text: str) -> bool:
    if not config.ENABLE_SCOPE_GUARD:
        return False
    try:
        from app.agent import scope

        return scope.is_off_topic(text)
    except Exception as exc:  # never let the guard break a turn
        log.warning("scope guard unavailable: %s", exc)
        return False


def deterministic_understand(messages: list[dict]) -> dict:
    """Keyword-based fallback route. Conservative and recall-safe."""
    u = empty_understanding()
    last = last_user(messages)
    text = last.lower()
    u["search_query"] = all_user_text(messages)
    u["hard"], u["soft"] = extract_constraints(u["search_query"])
    has_prior_list = _has_prior_shortlist(messages)

    # Scope: injection (keyword) fires anytime; off-topic (keyword + semantic) only when
    # there is no ongoing shortlist, so conversational follow-ups are never refused.
    injection = any(p in text for p in _INJECTION)
    off_topic = not has_prior_list and (
        any(p in text for p in _OFFTOPIC) or _semantic_off_topic(last)
    )
    if injection or off_topic:
        u["in_scope"] = False
        u["intent"] = "refuse"
        return u

    if any(p in text for p in _DONE) and not any(p in text for p in _REFINE):
        u["user_done"] = True

    if any(p in text for p in _COMPARE):
        u["intent"] = "compare"
        u["compare_names"] = _extract_compare_names(last)
        return u

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
    # Compact, token-frugal transcript (cap to the evaluator's 8-message window).
    convo = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages[-8:]
    )
    strict = "\n\nReturn ONLY one valid JSON object with exactly the required fields."
    for attempt in range(2):
        try:
            user = convo if attempt == 0 else convo + strict
            return _guard(_normalize(groq_client.chat_json(UNDERSTAND_SYSTEM, user)), messages)
        except json.JSONDecodeError as exc:  # only a parse error is worth retrying
            log.warning("understand JSON parse failed (attempt %d): %s", attempt + 1, exc)
            continue
        except Exception as exc:  # rate limit / timeout / API error -> fallback now
            log.warning("understand LLM error, using deterministic route: %s", exc)
            break
    # Deterministic vagueness-aware fallback (never just blindly recommend).
    return deterministic_understand(messages)
