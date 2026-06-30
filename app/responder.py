"""Final-response assembly and the anti-hallucination gate.

Everything the agent produces passes through :func:`build_response`, which:
  * canonicalizes each recommendation against the catalog (verbatim URL, real type),
  * drops anything not in the catalog,
  * removes duplicates (by URL),
  * caps the list at ``MAX_RECS``,
  * returns a strictly-valid :class:`ChatResponse`.

This is the single place that enforces the "items from catalog only" + schema hard
evals, so no agent code path can leak a bad item.
"""
from __future__ import annotations

from app import config
from app.data.catalog import Catalog, norm_url
from app.schemas import ChatResponse, Recommendation


def sanitize_recommendations(items: list[dict], catalog: Catalog) -> list[Recommendation]:
    out: list[Recommendation] = []
    seen: set[str] = set()
    for item in items or []:
        canon = catalog.canonicalize(
            name=item.get("name", ""),
            url=item.get("url", ""),
            test_type=item.get("test_type", ""),
        )
        if canon is None:
            continue  # off-catalog -> dropped (never hallucinate)
        key = norm_url(canon["url"])
        if key in seen:
            continue  # de-duplicate
        seen.add(key)
        out.append(Recommendation(**canon))
        if len(out) >= config.MAX_RECS:
            break
    return out


def build_response(
    reply: str,
    items: list[dict] | None,
    end_of_conversation: bool,
    catalog: Catalog,
) -> ChatResponse:
    recs = sanitize_recommendations(items or [], catalog)
    return ChatResponse(
        reply=(reply or "").strip() or "Sorry, could you rephrase that?",
        recommendations=recs,
        end_of_conversation=bool(end_of_conversation),
    )


def safe_fallback_response(
    reply: str = "Sorry, I hit a snag. Could you rephrase your request?",
) -> ChatResponse:
    """Last-resort schema-valid response used by the global safety wrapper."""
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
