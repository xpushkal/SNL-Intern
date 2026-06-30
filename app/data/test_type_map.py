"""Map SHL catalog ``keys[]`` categories to the single-letter ``test_type`` codes
required by the response schema.

The provided catalog JSON has no ``test_type`` field, but its ``keys`` values are
SHL's standard category names. The sample conversation traces confirm the mapping
(e.g. "Personality & Behavior" -> ``P``). For the 39/377 multi-category records we
expose all codes and choose a single *primary* deterministically.

Primary-selection rule (documented for the interview):
  1. If the record has exactly one category -> that code (338/377 records).
  2. Otherwise the catalog lists categories alphabetically, so source order is NOT
     meaningful; we fall back to a fixed priority that ranks the most "test-like"
     (discriminating) categories first.
"""
from __future__ import annotations

from typing import Iterable

# Canonical category -> code. Spelling matches the catalog exactly
# ("Judgment", "Behavior"); common variants are tolerated below.
CATEGORY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# Tolerate UK/US spelling drift without silently dropping a category.
_VARIANTS = {
    "biodata & situational judgement": "B",
    "personality & behaviour": "P",
}

# Documented fallback priority for unresolved multi-category records.
_PRIORITY = ["K", "A", "P", "B", "C", "S", "E", "D"]


def code_for_category(category: str) -> str | None:
    if category in CATEGORY_TO_CODE:
        return CATEGORY_TO_CODE[category]
    return _VARIANTS.get(category.strip().lower())


def all_test_types(keys: Iterable[str]) -> list[str]:
    """All distinct codes for a record's categories, in priority order."""
    codes = {c for k in keys if (c := code_for_category(k))}
    return [c for c in _PRIORITY if c in codes]


def primary_test_type(keys: Iterable[str]) -> str:
    """The single primary code for a record (see module docstring for the rule)."""
    codes = all_test_types(keys)
    if not codes:
        # No recognizable category: 'K' is the catalog's dominant type and a safe
        # default. (Does not occur in the current catalog — every record maps.)
        return "K"
    return codes[0]
