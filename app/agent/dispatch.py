"""The act node: deterministically turn an `understanding` into (items, reply, end).

Handles intent routing, the one-clarification turn-budget policy, stateless prior-list
reconstruction, positional refinement, grounded comparison, and the explicit-completion
`end_of_conversation` rule.
"""
from __future__ import annotations

import json
import logging

from app import config
from app.agent import groq_client, render
from app.agent.prompts import COMPARE_SYSTEM
from app.agent.understand import all_user_text
from app.data.catalog import load_catalog
from app.retrieval import ranking
from app.retrieval.names import resolve_name, resolve_names

log = logging.getLogger("shl.dispatch")

_REFUSE_MARKER = "I can only help you choose SHL assessments"
_ORDINALS = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4,
    "sixth": 5, "seventh": 6, "eighth": 7, "ninth": 8, "tenth": 9,
}


def count_clarifications(messages: list[dict]) -> int:
    """Prior clarifications = assistant turns with no shortlist that are not refusals.

    Deliberately does NOT treat every '?' as a clarification (a refusal also has no
    shortlist but carries the refusal marker)."""
    n = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if render._URL_RE.search(content):
            continue  # had a shortlist -> not a clarification
        if _REFUSE_MARKER in content:
            continue  # refusal
        n += 1
    return n


def dispatch(messages: list[dict], u: dict) -> tuple[list[dict], str, bool]:
    catalog = load_catalog()
    prior = render.parse_prior_shortlist(messages)

    # --- scope / refusal -------------------------------------------------
    if not u.get("in_scope", True) or u.get("intent") == "refuse":
        return [], render.render_refuse(), False

    intent = u.get("intent", "recommend")
    user_done = bool(u.get("user_done"))
    actionable = bool(u.get("remove_names") or u.get("add_query") or intent == "compare")

    # --- explicit completion with nothing new -> confirm prior shortlist --
    if user_done and not actionable and prior:
        items = prior[: config.MAX_RECS]
        return items, render.render_confirm(items), True

    # --- one-clarification turn-budget policy ----------------------------
    if intent == "clarify":
        over_budget = (
            count_clarifications(messages) >= config.CLARIFY_MAX
            or len(messages) >= config.TURN_SOFT_LIMIT
        )
        if not over_budget:
            return [], render.render_clarify(u.get("clarifying_question")), False
        intent = "recommend"  # commit with whatever we have

    # --- compare ---------------------------------------------------------
    if intent == "compare":
        items = resolve_names(u.get("compare_names", []), catalog)
        if not items:
            return [], (
                "I couldn't find those in the catalog. Which SHL assessments would you "
                "like me to compare?"
            ), False
        summary = _compare_summary(items)
        return items[: config.MAX_RECS], render.render_compare(items, summary), False

    # --- refine ----------------------------------------------------------
    if intent == "refine" and prior:
        items = _apply_refine(prior, u)
        end = user_done and bool(items)
        reply = render.render_confirm(items) if end else render.render_recommend(items, refined=True)
        return items, reply, end

    # --- recommend (default) ---------------------------------------------
    query = (u.get("search_query") or "").strip() or all_user_text(messages)
    items = ranking.search(
        query, hard=u.get("hard", {}), soft=u.get("soft", {}), k=config.RECOMMEND_FILL
    )
    end = user_done and bool(items)
    reply = render.render_confirm(items) if end else render.render_recommend(items)
    return items, reply, end


# --- helpers -------------------------------------------------------------
def _apply_refine(prior: list[dict], u: dict) -> list[dict]:
    keep = list(prior)
    for target in u.get("remove_names", []):
        idx = _resolve_in_prior(target, keep)
        if idx is not None:
            keep.pop(idx)
    add_query = (u.get("add_query") or "").strip()
    if add_query:
        existing = {r["id"] for r in keep}
        for rec in ranking.search(add_query, hard=u.get("hard", {}), soft=u.get("soft", {}),
                                  k=config.RECOMMEND_FILL):
            if rec["id"] not in existing:
                keep.append(rec)
                existing.add(rec["id"])
            if len(keep) >= config.MAX_RECS:
                break
    return keep[: config.MAX_RECS]


def _resolve_in_prior(target: str, prior: list[dict]) -> int | None:
    """Resolve 'the second one' / '#2' / a name to an index in the current shortlist."""
    t = (target or "").strip().lower()
    if not t or not prior:
        return None
    for word, idx in _ORDINALS.items():
        if word in t.split() or word == t:
            return idx if idx < len(prior) else None
    if t in {"last", "the last one", "last one"}:
        return len(prior) - 1
    digits = "".join(ch for ch in t if ch.isdigit())
    if digits:
        i = int(digits) - 1
        if 0 <= i < len(prior):
            return i
    # name match against the current shortlist only
    rec = resolve_name(target)
    if rec:
        for i, r in enumerate(prior):
            if r["id"] == rec["id"]:
                return i
    return None


def _compare_summary(items: list[dict]) -> str:
    facts = [
        {
            "name": r["name"],
            "test_type": r["test_type"],
            "categories": r.get("keys", []),
            "duration": r.get("duration_raw") or "unspecified",
            "job_levels": r.get("job_levels", []),
            "description": (r.get("description") or "")[:600],
        }
        for r in items
    ]
    if groq_client.available():
        try:
            return groq_client.chat_text(COMPARE_SYSTEM, json.dumps(facts, ensure_ascii=False))
        except Exception as exc:  # fall back to a deterministic, grounded template
            log.warning("compare LLM failed: %s", exc)
    return render.deterministic_compare(items)
