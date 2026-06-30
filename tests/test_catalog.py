"""Catalog normalization, test_type mapping, and canonical-validation tests."""
from app.data.catalog import load_catalog
from app.data.test_type_map import all_test_types, primary_test_type


def test_catalog_loads_and_scopes():
    c = load_catalog()
    assert len(c.records) == 377
    # exactly the 7 pre-packaged bundles are out of scope
    assert len(c.recommendable) == 370
    assert all(not r["is_individual"] for r in c.records if r not in c.recommendable)


def test_test_type_mapping_known_values():
    assert primary_test_type(["Knowledge & Skills"]) == "K"
    assert primary_test_type(["Personality & Behavior"]) == "P"
    assert primary_test_type(["Ability & Aptitude"]) == "A"
    # multi-category -> deterministic priority, all codes preserved
    codes = all_test_types(["Simulations", "Knowledge & Skills"])
    assert set(codes) == {"S", "K"}
    assert primary_test_type(["Simulations", "Knowledge & Skills"]) == "K"


def test_canonical_validation_rejects_offcatalog():
    c = load_catalog()
    assert c.canonicalize("Totally Fake Test", "https://evil.com/x", "K") is None
    rec = c.recommendable[0]
    canon = c.canonicalize(rec["name"], rec["url"], rec["test_type"])
    assert canon and canon["url"] == rec["url"]  # URL returned verbatim


def test_every_record_maps_to_a_valid_test_type():
    valid = set("ABCDEKPS")
    for r in load_catalog().records:
        assert r["test_type"] in valid
        assert all(t in valid for t in r["test_types"])
