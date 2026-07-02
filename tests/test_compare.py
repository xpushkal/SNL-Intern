"""Comparison resolves EVERY explicitly-named assessment (not just the best fuzzy match)."""
from app.agent.turn import run_turn
from app.retrieval.names import resolve_name


def _names(query):
    return [x.name for x in run_turn([{"role": "user", "content": query}]).recommendations]


def test_curated_aliases_resolve():
    assert resolve_name("DSI")["name"] == "Dependability and Safety Instrument (DSI)"
    assert resolve_name("Verify G+")["name"] == "SHL Verify Interactive G+"
    assert resolve_name("OPQ")["name"] == "Occupational Personality Questionnaire OPQ32r"
    assert resolve_name("GSA")["name"] == "Global Skills Assessment"


def test_compare_contact_center_vs_customer_service_phone():
    names = _names("Contact Center Call Simulation vs Customer Service Phone Simulation")
    assert any("Contact Center Call Simulation" in n for n in names)
    assert any(n == "Customer Service Phone Simulation" for n in names)
    assert len(names) == 2


def test_compare_dsi_vs_safety_and_dependability():
    names = _names("difference between DSI and Safety & Dependability 8.0")
    assert any(n == "Dependability and Safety Instrument (DSI)" for n in names)
    assert any("Safety & Dependability 8.0" in n for n in names)
    assert len(names) == 2


def test_compare_opq_vs_gsa():
    names = _names("what is the difference between OPQ and GSA?")
    assert any("OPQ32r" in n for n in names)
    assert any(n == "Global Skills Assessment" for n in names)
    assert len(names) == 2
