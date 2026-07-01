"""Deterministic, templated reply rendering.

Because the wire schema is fixed, the numbered list embedded in the reply text is the
ONLY carrier of cross-turn state. We therefore always print each item's catalog URL,
and reconstruct a prior shortlist by parsing those URLs back out of the most recent
assistant message (URLs are unambiguous catalog keys, unlike names with parentheses).
"""
from __future__ import annotations

import re

from app.data.catalog import load_catalog

_URL_RE = re.compile(r"https?://[^\s<>)\]]+")

# Machine marker that tags STATE-BEARING replies (recommend / refine / confirm). The
# active shortlist is reconstructed only from marked messages, so comparisons,
# clarifications, and refusals never overwrite it. Stripped in the demo UI.
STATE_MARK = "<!--shl:shortlist-->"


def parse_prior_shortlist(messages: list[dict]) -> list[dict]:
    """Records from the latest STATE-BEARING assistant message (recommend/refine/confirm).

    Skips comparisons, clarifications, and refusals (which are not marked), so those
    never replace the active shortlist."""
    catalog = load_catalog()
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if STATE_MARK not in content:
            continue  # not a shortlist-bearing reply -> skip
        out, seen = [], set()
        for u in _URL_RE.findall(content):
            rec = catalog.get_by_url(u)
            if rec and rec["id"] not in seen:
                seen.add(rec["id"])
                out.append(rec)
        return out
    return []


def _line(i: int, rec: dict) -> str:
    dur = rec.get("duration_raw") or "duration n/a"
    return f"{i}. {rec['name']} (Type {rec['test_type']}, {dur}) — {rec['url']}"


def render_recommend(items: list[dict], *, refined: bool = False) -> str:
    if not items:
        return (
            "I couldn't find a matching SHL assessment for that. Could you add a bit "
            "more detail about the role or skills?"
        )
    head = (
        "Updated shortlist:" if refined
        else (f"Here {'is' if len(items) == 1 else 'are'} "
              f"{len(items)} SHL assessment{'' if len(items) == 1 else 's'} that fit:")
    )
    body = "\n".join(_line(i, r) for i, r in enumerate(items, 1))
    return f"{head}\n{body}\n{STATE_MARK}"


def render_confirm(items: list[dict]) -> str:
    if not items:
        return "Glad I could help. Let me know if you need anything else."
    body = "\n".join(_line(i, r) for i, r in enumerate(items, 1))
    return f"Confirmed — your final SHL shortlist:\n{body}\n{STATE_MARK}"


def render_compare(items: list[dict], summary: str) -> str:
    body = "\n".join(_line(i, r) for i, r in enumerate(items, 1))
    summary = (summary or "").strip()
    return f"{summary}\n\n{body}" if summary else body


def render_clarify(question: str | None) -> str:
    return (question or "").strip() or (
        "Happy to help find SHL assessments. What role or key skills are you hiring for?"
    )


def render_refuse() -> str:
    return (
        "I can only help you choose SHL assessments from the catalog. I can't help with "
        "that request, but tell me the role or skills you're hiring for and I'll suggest "
        "relevant assessments."
    )


def deterministic_compare(items: list[dict]) -> str:
    """Fallback comparison built purely from catalog fields (no LLM)."""
    if len(items) < 2:
        return ""
    parts = []
    for r in items:
        cats = ", ".join(r.get("keys", [])) or "n/a"
        dur = r.get("duration_raw") or "unspecified duration"
        parts.append(
            f"{r['name']} is a Type {r['test_type']} assessment ({cats}), {dur}."
        )
    return " ".join(parts)
