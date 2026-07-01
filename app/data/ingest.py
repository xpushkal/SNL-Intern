"""Ingest the raw SHL catalog JSON into clean, normalized records.

Run as a script to (re)build ``artifacts/catalog.normalized.json``:

    python -m app.data.ingest

Design notes
------------
* The raw file contains literal control characters inside string values, so it is
  parsed with ``strict=False``.
* Verified empirically: every gold-shortlist item across all 10 sample traces is
  present in this file, so the provided JSON *is* the recommendable universe -- no
  website scraping is required.
* Pre-packaged Job Solutions are out of scope. Exactly 7 records are such bundles
  and all end in " Solution" (e.g. "Entry Level Cashier Solution"); none appear in
  any gold shortlist. They are tagged ``is_individual=False`` (a narrow, auditable
  rule) and excluded from recommendations unless ``INCLUDE_PREPACKAGED`` is set.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app import config
from app.data.test_type_map import all_test_types, primary_test_type

_INT_RE = re.compile(r"\d+")

# Bump when the doc_text() representation changes (invalidates baked embeddings).
DOC_SCHEMA_VERSION = 1


def catalog_hash(records: list[dict]) -> str:
    """Stable digest of the catalog fields that feed retrieval + recommendations."""
    fields = ("id", "name", "url", "description", "keys", "job_levels",
              "duration_raw", "languages", "is_individual")
    payload = json.dumps(
        [{k: r.get(k) for k in fields} for r in records], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _parse_minutes(duration: str) -> int | None:
    """Best-effort minutes from strings like '30 minutes'. 'Untimed'/'' -> None."""
    if not duration:
        return None
    m = _INT_RE.search(duration)
    return int(m.group()) if m else None


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "yes"


def _is_prepackaged(name: str) -> bool:
    return name.strip().lower().endswith("solution")


def doc_text(rec: dict) -> str:
    """Searchable text for embedding + BM25 (name carries the most signal)."""
    parts = [
        rec["name"],
        rec["name"],  # repeat: weight the name in BM25 term stats
        rec.get("description", ""),
        " ".join(rec.get("job_levels", [])),
        " ".join(rec.get("keys", [])),
    ]
    return "  ".join(p for p in parts if p).strip()


def normalize_record(raw: dict) -> dict:
    name = (raw.get("name") or "").strip()
    keys = list(raw.get("keys") or [])
    return {
        "id": str(raw.get("entity_id", "")).strip(),
        "name": name,
        "url": (raw.get("link") or "").strip(),  # returned verbatim, never edited
        "description": (raw.get("description") or "").strip(),
        "job_levels": list(raw.get("job_levels") or []),
        "languages": list(raw.get("languages") or []),
        "duration_raw": (raw.get("duration") or "").strip(),
        "duration_minutes": _parse_minutes(raw.get("duration") or ""),
        "remote": _as_bool(raw.get("remote", "")),
        "adaptive": _as_bool(raw.get("adaptive", "")),
        "keys": keys,
        "test_type": primary_test_type(keys),
        "test_types": all_test_types(keys),
        "is_individual": not _is_prepackaged(name),
    }


def load_raw(path: Path | None = None) -> list[dict]:
    path = path or config.RAW_CATALOG_PATH
    return json.loads(Path(path).read_text(encoding="utf-8"), strict=False)


def build_normalized(raw_path: Path | None = None) -> list[dict]:
    records = [normalize_record(r) for r in load_raw(raw_path)]
    # Stable de-dup by id (defensive; catalog has none today).
    seen: set[str] = set()
    out = []
    for r in records:
        if r["id"] and r["id"] in seen:
            continue
        seen.add(r["id"])
        out.append(r)
    return out


def main() -> None:
    records = build_normalized()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.NORMALIZED_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    individual = sum(r["is_individual"] for r in records)
    print(f"normalized {len(records)} records -> {config.NORMALIZED_PATH}")
    print(f"  individual (recommendable): {individual}")
    print(f"  pre-packaged (excluded):    {len(records) - individual}")


if __name__ == "__main__":
    main()
