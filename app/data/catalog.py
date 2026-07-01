"""Runtime catalog store: load normalized records, expose fast lookups, and provide
the canonical-validation helpers that guarantee we never emit an off-catalog item.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache

from app import config


def norm_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def norm_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


class Catalog:
    def __init__(self, records: list[dict]):
        self.records = records
        self.by_id = {r["id"]: r for r in records if r["id"]}
        self.by_url = {norm_url(r["url"]): r for r in records}
        self.by_name = {norm_name(r["name"]): r for r in records}
        # Recommendable subset (Individual Test Solutions only, unless overridden).
        self.recommendable = [
            r for r in records if r["is_individual"] or config.INCLUDE_PREPACKAGED
        ]

    # --- lookups ---------------------------------------------------------
    def get_by_url(self, url: str) -> dict | None:
        return self.by_url.get(norm_url(url))

    def get_by_name(self, name: str) -> dict | None:
        return self.by_name.get(norm_name(name))

    def get(self, name: str | None = None, url: str | None = None) -> dict | None:
        """Resolve a record by url (authoritative) then name."""
        if url and (rec := self.get_by_url(url)):
            return rec
        if name and (rec := self.get_by_name(name)):
            return rec
        return None

    # --- validation / canonicalization -----------------------------------
    def to_recommendation(self, rec: dict) -> dict:
        """Canonical {name, url, test_type} straight from the catalog (verbatim url)."""
        return {"name": rec["name"], "url": rec["url"], "test_type": rec["test_type"]}

    def canonicalize(self, name: str, url: str, test_type: str) -> dict | None:
        """Return the canonical recommendation for a (name|url) if it exists in the
        catalog AND is an Individual Test Solution; else None. This is the final gate
        applied before every response, so a pre-packaged Job Solution can never be
        returned -- including via comparison or exact-name resolution. The test_type is
        always corrected to the record's canonical value."""
        rec = self.get(name=name, url=url)
        if rec is None:
            return None
        if not rec.get("is_individual") and not config.INCLUDE_PREPACKAGED:
            return None  # pre-packaged Job Solution -> out of scope, never returned
        return self.to_recommendation(rec)


_LOCK = threading.Lock()
_INSTANCE: Catalog | None = None


@lru_cache(maxsize=1)
def _load_records() -> tuple[dict, ...]:
    path = config.NORMALIZED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Normalized catalog missing at {path}. Run `python -m app.data.ingest`."
        )
    return tuple(json.loads(path.read_text(encoding="utf-8")))


def load_catalog() -> Catalog:
    """Process-wide singleton (loaded once at startup)."""
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = Catalog(list(_load_records()))
    return _INSTANCE
