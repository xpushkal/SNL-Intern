"""Refinement: add/remove/replace/keep-only/multiple actions + constraint application."""
from app.agent.turn import run_turn


def _java_base():
    base = run_turn([{"role": "user", "content": "hiring a java developer"}])
    assert len(base.recommendations) == 10 and set(t.test_type for t in base.recommendations) == {"K"}
    return base


def _hist(base, msg):
    return [
        {"role": "user", "content": "hiring a java developer"},
        {"role": "assistant", "content": base.reply},
        {"role": "user", "content": msg},
    ]


def test_add_personality_to_full_shortlist_evicts_and_retains():
    base = _java_base()
    r = run_turn(_hist(base, "actually, add a personality test"))
    types = [t.test_type for t in r.recommendations]
    assert len(r.recommendations) == 10          # stayed full (evicted a K)
    assert "P" in types                          # the addition was retained
    assert "Updated shortlist" in r.reply


def test_replace_removes_and_adds():
    base = _java_base()
    victim = base.recommendations[0].name
    r = run_turn(_hist(base, f"replace {victim} with a personality test"))
    names = [x.name for x in r.recommendations]
    assert victim not in names
    assert "P" in [t.test_type for t in r.recommendations]


def test_keep_only_restricts_shortlist():
    base = _java_base()
    a, b = base.recommendations[0].name, base.recommendations[1].name
    r = run_turn(_hist(base, f"keep only {a} and {b}"))
    names = [x.name for x in r.recommendations]
    assert names == [a, b]


def test_multiple_actions_remove_and_add():
    base = _java_base()
    first = base.recommendations[0].name
    r = run_turn(_hist(base, "remove the first one and add a personality test"))
    names = [x.name for x in r.recommendations]
    assert first not in names
    assert "P" in [t.test_type for t in r.recommendations]


def test_add_multiple_products_retains_both():
    base = run_turn([{"role": "user", "content": "hiring a cloud engineer"}])
    r = run_turn([
        {"role": "user", "content": "hiring a cloud engineer"},
        {"role": "assistant", "content": base.reply},
        {"role": "user", "content": "add AWS and Docker"},
    ])
    names = [x.name for x in r.recommendations]
    assert any("AWS" in n for n in names) and any("Docker" in n for n in names)


def test_multiple_actions_add_two_and_remove_one():
    base = run_turn([{"role": "user", "content": "hiring a cloud engineer"}])
    first = base.recommendations[0].name
    r = run_turn([
        {"role": "user", "content": "hiring a cloud engineer"},
        {"role": "assistant", "content": base.reply},
        {"role": "user", "content": "add AWS and Docker; drop the first one"},
    ])
    names = [x.name for x in r.recommendations]
    assert any("AWS" in n for n in names) and any("Docker" in n for n in names)
    assert first not in names


def test_new_duration_constraint_applies_to_shortlist():
    from app.data.catalog import load_catalog

    cat = load_catalog()
    base = _java_base()
    r = run_turn(_hist(base, "make them under 15 minutes"))
    assert r.recommendations, "should retain the short-enough items"
    for rec in r.recommendations:
        mins = cat.get_by_url(rec.url)["duration_minutes"]
        assert mins is not None and mins <= 15
