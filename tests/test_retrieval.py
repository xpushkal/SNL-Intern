"""Retrieval: hard constraints inviolable, soft never excludes, modes work."""
from app.retrieval import ranking


def test_hard_duration_never_violated_and_may_return_fewer():
    res = ranking.search("java coding test", hard={"max_duration_minutes": 10}, k=10)
    assert all(r["duration_minutes"] is not None and r["duration_minutes"] <= 10 for r in res)


def test_hard_test_type_filter():
    res = ranking.search("leadership personality", hard={"test_types": ["P"]}, k=10)
    assert all("P" in r["test_types"] for r in res)


def test_modes_all_return_results():
    for mode in ("bm25", "dense", "hybrid"):
        res = ranking.search("python developer", k=5, mode=mode)
        assert 1 <= len(res) <= 5


def test_soft_pref_does_not_exclude():
    plain = ranking.search("project manager", k=10)
    boosted = ranking.search("project manager", soft={"job_levels": ["Manager"]}, k=10)
    assert len(plain) == len(boosted)  # soft reorders, never filters
