"""Failure fallbacks: LLM errors degrade to a sensible deterministic route."""
from app.agent import groq_client, understand
from app.agent.turn import run_turn


def test_is_vague_detection():
    assert understand.is_vague("I need an assessment")
    assert understand.is_vague("can you help me")
    assert not understand.is_vague("hiring a java developer")
    assert not understand.is_vague("5 years experience leading sales teams")


def test_llm_failure_falls_back_to_deterministic(monkeypatch):
    # Pretend a key exists but every call raises -> must not crash, must stay valid.
    monkeypatch.setattr(groq_client, "available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(groq_client, "chat_json", boom)

    # Vague query under LLM failure -> clarify (NOT a blind recommend).
    r = run_turn([{"role": "user", "content": "I need an assessment"}])
    assert r.recommendations == [] and not r.end_of_conversation

    # Concrete query under LLM failure -> still recommends from the catalog.
    r2 = run_turn([{"role": "user", "content": "hiring a python developer"}])
    assert 1 <= len(r2.recommendations) <= 10


def test_understand_without_key_is_deterministic(monkeypatch):
    monkeypatch.setattr(groq_client, "available", lambda: False)
    u = understand.understand([{"role": "user", "content": "compare OPQ and GSA"}])
    assert u["intent"] == "compare"
