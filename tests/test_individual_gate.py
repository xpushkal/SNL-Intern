"""Pre-packaged Job Solutions can never be returned (final gate + name resolution)."""
from app.agent.turn import run_turn
from app.data.catalog import load_catalog
from app.responder import sanitize_recommendations
from app.retrieval.names import resolve_name

_PREPACKAGED = "Customer Service Phone Solution"


def _prepackaged_record():
    cat = load_catalog()
    rec = cat.by_name.get(_PREPACKAGED.lower())
    assert rec is not None and not rec["is_individual"], "fixture must be a pre-packaged item"
    return cat, rec


def test_name_resolution_rejects_prepackaged():
    assert resolve_name(_PREPACKAGED) is None


def test_sanitizer_drops_prepackaged():
    cat, rec = _prepackaged_record()
    out = sanitize_recommendations(
        [{"name": rec["name"], "url": rec["url"], "test_type": rec["test_type"]}], cat
    )
    assert out == []


def test_compare_never_returns_prepackaged():
    r = run_turn([{"role": "user", "content": f"compare {_PREPACKAGED} and OPQ32r"}])
    names = [x.name for x in r.recommendations]
    assert _PREPACKAGED not in names
    assert all("solution" not in n.lower() for n in names)
