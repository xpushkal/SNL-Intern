"""Hard-eval safety: every response is schema-valid; no 4xx/5xx ever."""
from fastapi.testclient import TestClient

from app.data.catalog import load_catalog
from app.main import app
from app.responder import sanitize_recommendations

client = TestClient(app)
REQUIRED = {"reply", "recommendations", "end_of_conversation"}


def _valid(body: dict) -> bool:
    return (
        set(body) == REQUIRED
        and isinstance(body["reply"], str)
        and isinstance(body["recommendations"], list)
        and isinstance(body["end_of_conversation"], bool)
        and len(body["recommendations"]) <= 10
        and all(set(r) == {"name", "url", "test_type"} for r in body["recommendations"])
    )


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_normal_chat_is_valid():
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "java dev"}]})
    assert r.status_code == 200 and _valid(r.json())


def test_malformed_bodies_never_error():
    for bad in ({}, {"bogus": 1}, {"messages": "notalist"}, {"messages": [{"x": 1}]}, []):
        r = client.post("/chat", json=bad)
        assert r.status_code == 200, bad
        assert _valid(r.json()), bad


def test_recommendations_sanitized_against_catalog():
    cat = load_catalog()
    real = cat.recommendable[0]
    items = [
        {"name": real["name"], "url": real["url"], "test_type": real["test_type"]},
        {"name": "Hallucinated", "url": "https://nope.example/x", "test_type": "K"},
        {"name": real["name"], "url": real["url"], "test_type": real["test_type"]},  # dup
    ]
    out = sanitize_recommendations(items, cat)
    assert len(out) == 1 and out[0].url == real["url"]  # off-catalog + dup dropped
