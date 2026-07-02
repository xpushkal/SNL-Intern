"""Shared evaluation helpers: trace parsing + metrics."""
from __future__ import annotations

import glob
import os
import re

from app.data.catalog import norm_url

TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
_URL_RE = re.compile(r"<(https?://[^>]+)>")
# A user turn is the full markdown blockquote after **User** -- one OR MORE ">" lines,
# so multiline messages (e.g. C9's full job description) are captured completely.
_USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n+((?:[ \t]*>.*(?:\n|$))+)")


def _clean_block(block: str) -> str:
    lines = [re.sub(r"^\s*>\s?", "", ln).rstrip()
             for ln in block.splitlines() if ln.strip().startswith(">")]
    return " ".join(l for l in lines if l).strip()


def trace_files() -> list[str]:
    return sorted(
        glob.glob(os.path.join(TRACES_DIR, "*.md")),
        key=lambda p: (len(os.path.basename(p)), p),
    )


def parse_trace(path: str) -> dict:
    txt = open(path, encoding="utf-8").read()
    user_turns = [b for b in (_clean_block(m) for m in _USER_BLOCK_RE.findall(txt)) if b]
    # Gold = the final labeled shortlist = URLs in the last turn that lists any.
    turns = re.split(r"### Turn \d+", txt)
    gold: set[str] = set()
    for seg in reversed(turns):
        urls = _URL_RE.findall(seg)
        if urls:
            gold = {norm_url(u) for u in urls}
            break
    return {"name": os.path.basename(path), "user_turns": user_turns, "gold": gold}


def recall_at_k(gold: set[str], ranked_urls: list[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    topk = {norm_url(u) for u in ranked_urls[:k]}
    return len(gold & topk) / len(gold)
