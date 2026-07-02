"""Battery-composition prior + shortlist-preserving question handling."""
from app.agent.turn import run_turn
from app.retrieval.ranking import FLAGSHIP_PERSONALITY
from app.data.catalog import norm_name


def _names(resp) -> list[str]:
    return [norm_name(r.name) for r in resp.recommendations]


def _java_hist(base, msg):
    return [
        {"role": "user", "content": "hiring a java developer"},
        {"role": "assistant", "content": base.reply},
        {"role": "user", "content": msg},
    ]


def test_full_shortlist_includes_flagship_personality():
    base = run_turn([{"role": "user", "content": "hiring a java developer"}])
    assert len(base.recommendations) == 10
    assert FLAGSHIP_PERSONALITY in _names(base)


def test_hard_duration_cap_excludes_flagship():
    # OPQ32r runs 25 minutes; a 10-minute cap must keep it out (hard > prior).
    r = run_turn([{"role": "user", "content": "java developer, tests under 10 minutes"}])
    assert FLAGSHIP_PERSONALITY not in _names(r)


def test_question_preserves_shortlist():
    base = run_turn([{"role": "user", "content": "hiring a java developer"}])
    r = run_turn(_java_hist(base, "Do we really need all these Java tests? Feels redundant."))
    assert [x.name for x in r.recommendations] == [x.name for x in base.recommendations]


def test_keep_named_item_adds_when_absent():
    base = run_turn([{"role": "user", "content": "hiring a java developer"}])
    assert "shl verify interactive g+" not in _names(base)
    r = run_turn(_java_hist(base, "Keep Verify G+. Locking it in."))
    assert "shl verify interactive g+" in _names(r)


def test_keep_generic_phrase_is_noop():
    base = run_turn([{"role": "user", "content": "hiring a java developer"}])
    r = run_turn(_java_hist(base, "keep the shortlist as-is"))
    assert [x.name for x in r.recommendations] == [x.name for x in base.recommendations]
