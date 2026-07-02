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
_LANGS = ("english", "german", "french", "spanish", "italian", "portuguese", "dutch",
          "chinese", "japanese", "korean", "arabic", "hindi", "polish", "russian")
# HARD language requirement ONLY on explicit phrasing. A casual "assessed in Spanish"
# must NOT filter -- verified against trace C7, whose gold includes English-only tests
# despite mentioning Spanish. So we require "must be/available/only in <lang>" or
# "<lang> only", never a bare "in <lang>".
_LANG_RE = re.compile(
    r"(?:must be (?:available )?in|available in|only in)\s+(" + "|".join(_LANGS) + r")\b"
    r"|\b(" + "|".join(_LANGS) + r")\s+only\b"
)
_TYPE_WORDS = {"personality": "P", "behaviour": "P", "behavior": "P", "cognitive": "A",
               "ability": "A", "aptitude": "A", "knowledge": "K", "skills": "K",
               "situational": "B", "biodata": "B", "simulation": "S", "competency": "C",
               "competencies": "C"}
# A test-type becomes a HARD filter only when phrased as a constraint over the shortlist.
_TYPE_CONSTRAINT_RE = re.compile(
    r"\b(?:make (?:them|it|these)|must be|should (?:all )?be|only|all of them|change (?:them|it) to)\b")


def extract_constraints(text: str) -> tuple[dict, dict]:
    """Deterministically pull HARD constraints (duration cap / language / test-type) and
    a SOFT seniority signal from free text."""
    t = (text or "").lower()
    hard: dict = {}
    if m := _DUR_RE.search(t):
        hard["max_duration_minutes"] = int(m.group(1))
    if lm := _LANG_RE.search(t):
        lang = next(g for g in lm.groups() if g)
        hard["languages"] = [lang.capitalize()]
    if _TYPE_CONSTRAINT_RE.search(t):
        codes = sorted({c for w, c in _TYPE_WORDS.items() if re.search(rf"\b{w}\b", t)})
        if codes:
            hard["test_types"] = codes
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
    for k in ("compare_names", "remove_names", "keep_only_names", "add_queries"):
        if not isinstance(base.get(k), list):
            base[k] = []
    # Back-compat: accept a singular add_query from the LLM and fold it in.
    if u.get("add_query") and not base["add_queries"]:
        base["add_queries"] = [u["add_query"]]
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

    # Scope is evaluated on EVERY latest request, even mid-conversation: keyword
    # injection and keyword off-topic (weather / legal / hiring-advice) always fire, so a
    # legal or weather question after a recommendation is still refused. Only the semantic
    # guard stays gated to turn-1-ish, to avoid refusing follow-ups like "perfect, thanks".
    injection = any(p in text for p in _INJECTION)
    keyword_off_topic = any(p in text for p in _OFFTOPIC)
    semantic_off_topic = (not has_prior_list) and _semantic_off_topic(last)
    if injection or keyword_off_topic or semantic_off_topic:
        u["in_scope"] = False
        u["intent"] = "refuse"
        return u

    if any(p in text for p in _DONE) and not any(p in text for p in _REFINE):
        u["user_done"] = True

    if any(p in text for p in _COMPARE):
        u["intent"] = "compare"
        u["compare_names"] = _extract_compare_names(last)
        return u

    refine_triggered = has_prior_list and (
        any(p in text for p in _REFINE)
        or _KEEP_CONFIRM_RE.search(text)
        or re.search(r"(?i)\b(show|same)\b.*\b(shortlist|list|them|it)\b", text)
        or re.search(r"(?i)final (?:list|shortlist)", text)
    )
    if refine_triggered:
        u["intent"] = "refine"
        if not _KEEP_CONFIRM_RE.search(last):  # "keep the list as is" -> no-op (return prior)
            _parse_refine_ops(last, u)
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


def _split_names(s: str) -> list[str]:
    parts = re.split(r"\s*(?:,| and | & )\s*", (s or "").strip())
    return [p.strip(" .?") for p in parts if p.strip(" .?")][:6]


# Strip leading question scaffolding ("how does", "what's the", "tell me", ...).
_COMPARE_STRIP_RE = re.compile(
    r"(?i)^\s*(?:how\s+(?:do|does|are)\s+|what(?:'s| is| are)?\s+(?:the\s+)?|"
    r"can you\s+|could you\s+|tell me\s+|please\s+|i want to\s+|i'd like to\s+)+")
# Comparison keywords -> a delimiter. Bare 'with'/'to' are handled separately (the
# "compare X with/to Y" form) to avoid splitting names like "Attention to Detail".
_COMPARE_KW_RE = re.compile(
    r"(?i)difference between|different from|differs? from|compared?\s+(?:to|with)|"
    r"compare|versus|\bvs\.?\b")
_COMPARE_SPLIT_RE = re.compile(r"\s*(?:\||,| and | & )\s*")


def _extract_compare_names(text: str) -> list[str]:
    """Extract EVERY explicitly-named assessment from a comparison request. Handles
    'difference between', 'different from', 'versus'/'vs', and 'compare X with/to Y'."""
    t = _COMPARE_STRIP_RE.sub("", (text or "").strip())
    if m := re.search(r"(?i)\bcompare\s+(.+?)\s+(?:with|to)\s+(.+)$", t):
        cand = [m.group(1), m.group(2)]
    else:
        cand = _COMPARE_SPLIT_RE.split(_COMPARE_KW_RE.sub("|", t))
    names = [c.strip(" ?.") for c in cand if len(c.strip(" ?.")) >= 2]
    return [n for n in names if n.lower() not in {"the", "difference", "between"}][:4]


def _extract_remove_names(text: str) -> list[str]:
    m = re.search(r"(?i)(?:remove|drop|without|exclude|take out|get rid of)\s+(?:the\s+)?(.+?)(?:[.,]|$)", text)
    return _split_names(m.group(1)) if m else []


# "keep the list as is" / "keeping the five solutions as our stack" -> keep unchanged.
# The negative lookahead avoids swallowing "keep only ..." (a keep-only op).
_KEEP_CONFIRM_RE = re.compile(
    r"(?i)\b(?:keep|keeping|leave|retain|stick with)\b(?!\s+(?:only|just)\b)"
    r".*\b(shortlist|list|solutions|stack|five|those|them|it|as[- ]is|as our|"
    r"the same|unchanged|as they are)\b")
# Adds with no retrievable target -> not actionable (don't pollute the shortlist).
_VAGUE_ADD = {"something", "anything", "something shorter", "a shorter one", "shorter one",
              "something else", "a shorter alternative", "shorter alternative",
              "something quicker", "a quicker one", "a shorter test", "one more"}


def _clean_add(s: str) -> str:
    return re.sub(r"(?i)\s+to\b.*$", "", (s or "").strip()).strip(" .?")


def _parse_refine_ops(text: str, u: dict) -> None:
    """Fill remove_names / keep_only_names / add_query for a refine request. Supports
    add, remove, replace/swap, keep-only, set-exact-list, and multiple actions."""
    # set the exact list: "final list: X and Y" / "the shortlist should be X and Y"
    sl = re.search(
        r"(?i)(?:final (?:list|shortlist)|the (?:final )?(?:list|shortlist))\s*"
        r"(?::|should be|is|=)\s*(.+)$", text)
    if sl:
        u["keep_only_names"] = _split_names(sl.group(1))
        return
    # replace / swap X with Y  ->  remove X, add Y (spans the whole clause)
    rep = re.search(r"(?i)(?:replace|swap)\s+(.+?)\s+(?:with|for|by)\s+(.+?)(?:[.,]|$)", text)
    if rep:
        u["remove_names"] = _split_names(rep.group(1))
        add = _clean_add(rep.group(2))
        u["add_queries"] = [] if add.lower() in _VAGUE_ADD else [add]
        return
    ko = re.search(r"(?i)(?:keep only|only keep|just keep|keep just)\s+(.+?)(?:[.,]|$)", text)
    if ko:
        u["keep_only_names"] = _split_names(ko.group(1))
        return
    # Clause-aware pass so "remove X and add Y" and "remove X and Y" both parse. A clause
    # without a verb continues the previous action's target list.
    removes: list[str] = []
    adds: list[str] = []
    current: str | None = None
    for clause in re.split(r"(?i)\s*(?:,|;| and then | then | and also | and )\s*", text):
        clause = clause.strip()
        if not clause:
            continue
        rm = re.match(r"(?i)(?:remove|drop|exclude|take out|get rid of|without)\s+(?:the\s+)?(.+)$", clause)
        am = re.match(r"(?i)(?:add|also include|include|throw in)\s+(?:a |an |some )?(.+)$", clause)
        if rm:
            current = "remove"; removes += _split_names(rm.group(1))
        elif am:
            current = "add"; adds.append(_clean_add(am.group(1)))
        elif current == "remove":
            removes += _split_names(clause)
        elif current == "add":
            adds.append(_clean_add(clause))
    u["remove_names"] = removes
    u["add_queries"] = [a for a in adds if a and a.lower() not in _VAGUE_ADD]


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
