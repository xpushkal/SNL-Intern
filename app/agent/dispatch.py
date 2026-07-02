"""The act node: deterministically turn an `understanding` into (items, reply, end).

Handles intent routing, the one-clarification turn-budget policy, stateless prior-list
reconstruction, positional refinement, grounded comparison, and the explicit-completion
`end_of_conversation` rule.
"""
from __future__ import annotations

import json
import logging
import re

from app import config
from app.agent import groq_client, render
from app.agent.prompts import COMPARE_SYSTEM
from app.agent.understand import all_user_text
from app.data.catalog import load_catalog
from app.retrieval import ranking
from app.retrieval.names import resolve_name, resolve_names
from app.retrieval.rerank import rerank

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
        if render.STATE_MARK in content or render._URL_RE.search(content):
            continue  # shortlist reply or comparison -> not a clarification
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
    actionable = bool(
        u.get("remove_names") or u.get("keep_only_names") or u.get("add_queries")
        or intent == "compare"
    )

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
    # Broad recall: retrieve over the FULL conversation text, augmented by the LLM's
    # distilled query for emphasis. (Measured: the distilled query alone underperforms
    # on multi-faceted/semantic needs where gold items are inferred, e.g. "senior Rust".)
    query = (all_user_text(messages) + " " + (u.get("search_query") or "")).strip()
    if config.ENABLE_LLM_RERANK and groq_client.available():
        pool = ranking.search(query, hard=u.get("hard", {}), soft=u.get("soft", {}),
                              k=config.RERANK_POOL)
        items = rerank(query, pool, config.RECOMMEND_FILL)
    else:
        items = ranking.search(
            query, hard=u.get("hard", {}), soft=u.get("soft", {}), k=config.RECOMMEND_FILL
        )
    end = user_done and bool(items)
    reply = render.render_confirm(items) if end else render.render_recommend(items)
    return items, reply, end


# --- helpers -------------------------------------------------------------
def _apply_refine(prior: list[dict], u: dict) -> list[dict]:
    """Apply keep-only, removes, newly-introduced hard constraints, and adds to the
    current shortlist. Adds evict the lowest-ranked item so they are always retained."""
    hard = u.get("hard", {}) or {}
    soft = u.get("soft", {}) or {}
    keep = list(prior)

    # keep-only / set-exact-list: restrict the shortlist to the named items. Positional
    # references resolve within the shown list; named items resolve catalog-wide (via
    # curated aliases) so "final list: Verify G+ ..." yields the exact instrument and never
    # a fuzzy neighbour like "Verify - G+".
    keep_only = u.get("keep_only_names") or []
    if keep_only:
        resolved = _resolve_keep_only(keep_only, prior)
        if resolved:
            keep = resolved

    # removes (may be several)
    for target in u.get("remove_names", []):
        i = _resolve_in_prior(target, keep)
        if i is not None:
            keep.pop(i)

    # apply NEWLY-introduced hard constraints (duration / language / test-type) to the
    # existing shortlist -- drop items that now violate them.
    if hard:
        keep = [r for r in keep if ranking.hard_ok(r, hard)]

    # adds: one item per requested capability, evicting the lowest-ranked to make room.
    for add_query in _add_queries(u):
        existing = {r["id"] for r in keep}
        for rec in ranking.search(add_query, hard=hard, soft=soft, k=config.RECOMMEND_FILL):
            if rec["id"] in existing:
                continue
            if len(keep) >= config.MAX_RECS:
                keep.pop()  # evict lowest-ranked so the addition is retained
            keep.append(rec)
            break
    return keep[: config.MAX_RECS]


def _add_queries(u: dict) -> list[str]:
    return [q.strip() for q in (u.get("add_queries") or []) if q and q.strip()]


def _resolve_keep_only(names: list[str], prior: list[dict]) -> list[dict]:
    """Resolve keep-only / set-list targets to catalog records. Positional refs use the
    shown list; everything else resolves catalog-wide (curated aliases), so an explicitly
    named item is honoured even if it wasn't in the prior shortlist."""
    out: list[dict] = []
    seen: set[str] = set()
    for name in names:
        idx = _positional_index(name, prior)
        rec = prior[idx] if idx is not None else resolve_name(name)
        if rec and rec["id"] not in seen:
            seen.add(rec["id"])
            out.append(rec)
    return out


_POS_RE = re.compile(r"^#?\s*(\d{1,2})$|(?:item|number|option|#)\s*(\d{1,2})")


def _positional_index(target: str, prior: list[dict]) -> int | None:
    """Strict positional reference only ('the second one' / '#2' / 'last'). A name that
    merely contains a number -- e.g. 'Java 8 (New)' -- is NOT treated as a position."""
    t = (target or "").strip().lower()
    if not t or not prior:
        return None
    for word, idx in _ORDINALS.items():
        if word == t or word in t.split():
            return idx if idx < len(prior) else None
    if t in {"last", "the last one", "last one"}:
        return len(prior) - 1
    if m := _POS_RE.search(t):
        i = int(m.group(1) or m.group(2)) - 1
        if 0 <= i < len(prior):
            return i
    return None


def _resolve_in_prior(target: str, prior: list[dict]) -> int | None:
    """Resolve a positional ref OR a name to an index in the current shortlist."""
    if (idx := _positional_index(target, prior)) is not None:
        return idx
    # exact/alias/global resolution first (unambiguous)
    rec = resolve_name(target)
    if rec:
        for i, r in enumerate(prior):
            if r["id"] == rec["id"]:
                return i
    # lenient fuzzy against the SHOWN shortlist only (small set) -- the user is referring
    # to an item already in front of them, so a shorthand like "Verify G+" should match.
    from rapidfuzz import fuzz, process

    names = [r["name"] for r in prior]
    best = process.extractOne(target, names, scorer=fuzz.WRatio)
    if best and best[1] >= 70:
        return names.index(best[0])
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
