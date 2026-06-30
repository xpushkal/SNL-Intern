"""Optional LLM reranking (feature-flagged, OFF by default).

Reorders a candidate pool against the conversation need. It can only ever reorder the
supplied candidates -- ids not in the pool are ignored -- so it cannot introduce an
off-catalog item. Any failure falls back to the deterministic input order, and unchosen
candidates are appended (never dropped) to protect recall.
"""
from __future__ import annotations

import json
import logging

from app.agent import groq_client
from app.agent.prompts import RERANK_SYSTEM

log = logging.getLogger("shl.rerank")


def rerank(need: str, candidates: list[dict], k: int) -> list[dict]:
    if not candidates or not groq_client.available():
        return candidates[:k]
    facts = [
        {
            "id": r["id"],
            "name": r["name"],
            "type": r["test_type"],
            "desc": (r.get("description") or "")[:200],
        }
        for r in candidates
    ]
    try:
        out = groq_client.chat_json(
            RERANK_SYSTEM, json.dumps({"need": need, "candidates": facts}, ensure_ascii=False)
        )
        ids = out.get("ids", []) if isinstance(out, dict) else []
        by_id = {r["id"]: r for r in candidates}
        ranked = [by_id[i] for i in ids if i in by_id]
        chosen = {r["id"] for r in ranked}
        ranked += [r for r in candidates if r["id"] not in chosen]  # keep the rest
        return ranked[:k]
    except Exception as exc:
        log.warning("rerank failed, using deterministic order: %s", exc)
        return candidates[:k]
