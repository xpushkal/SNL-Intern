"""Shared text utilities for retrieval (used by both index build and query time)."""
from __future__ import annotations

import re

# bge-* retrieval models expect this instruction prefixed to *queries* (not passages).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization, shared by BM25 build and query.

    Deliberately no stemming/plural folding: measured on the public traces it LOWERED
    scripted-final Recall@10 (0.713 -> 0.693)."""
    return _TOKEN_RE.findall((text or "").lower())
