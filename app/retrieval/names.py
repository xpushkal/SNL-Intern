"""Deterministic assessment-name resolution for comparison requests.

Resolution order (no LLM, so it can never hallucinate a match):
  1. exact normalized-name match,
  2. alias match (embedded acronyms like "OPQ"/"GSA" + initialisms), unambiguous only,
  3. fuzzy match (rapidfuzz token_set_ratio) above a conservative threshold.
"""
from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz, process

from app import config
from app.data.catalog import Catalog, load_catalog, norm_name

_STOP = {"and", "of", "the", "for", "a", "an", "to", "in", "new", "report", "test"}

# Curated aliases for genuinely ambiguous flagship acronyms (the bare acronym maps
# to many products, so we pin it to the canonical instrument).
_CURATED = {
    "opq": "Occupational Personality Questionnaire OPQ32r",
}


def _aliases_for(name: str) -> set[str]:
    aliases: set[str] = set()
    words = re.findall(r"[A-Za-z0-9]+", name)
    # initialism of significant words: "Global Skills Assessment" -> "gsa"
    sig = [w for w in words if w.lower() not in _STOP and w[0].isupper()]
    if len(sig) >= 2:
        aliases.add("".join(w[0] for w in sig).lower())
    # embedded acronym tokens: "OPQ32r" -> {"opq32r", "opq"}; "ADEPT-15" -> "adept"
    for tok in words:
        if tok.isupper() and len(tok) >= 2:
            aliases.add(tok.lower())
        m = re.match(r"^([A-Za-z]{2,})\d", tok)
        if m:
            aliases.add(m.group(1).lower())
    return {a for a in aliases if len(a) >= 2}


@lru_cache(maxsize=4)
def _alias_index(catalog_id: int) -> dict[str, str | None]:
    """alias -> record id, or None when the alias is ambiguous (maps to >1 record)."""
    catalog = load_catalog()
    index: dict[str, str | None] = {}
    for rec in catalog.recommendable:
        for alias in _aliases_for(rec["name"]):
            if alias in index and index[alias] != rec["id"]:
                index[alias] = None  # ambiguous -> unusable
            else:
                index.setdefault(alias, rec["id"])
    return index


def resolve_name(query: str, catalog: Catalog | None = None) -> dict | None:
    catalog = catalog or load_catalog()
    q = norm_name(query)
    if not q:
        return None
    # 0) curated flagship acronyms
    if q in _CURATED:
        return catalog.get_by_name(_CURATED[q])
    # 1) exact
    if rec := catalog.get_by_name(query):
        return rec
    # 2) alias (unambiguous only)
    idx = _alias_index(id(catalog))
    rid = idx.get(q)
    if rid:
        return catalog.by_id.get(rid)
    # 3) fuzzy over recommendable names (WRatio handles partial / out-of-order names)
    names = [r["name"] for r in catalog.recommendable]
    match = process.extractOne(query, names, scorer=fuzz.WRatio)
    if match and match[1] >= config.FUZZY_THRESHOLD:
        return catalog.get_by_name(match[0])
    return None


def resolve_names(queries: list[str], catalog: Catalog | None = None) -> list[dict]:
    catalog = catalog or load_catalog()
    out: list[dict] = []
    seen: set[str] = set()
    for q in queries or []:
        rec = resolve_name(q, catalog)
        if rec and rec["id"] not in seen:
            seen.add(rec["id"])
            out.append(rec)
    return out
