"""DoS protection: oversized input is bounded (clamped or degraded) but the
never-4xx/5xx, always-schema-valid contract still holds."""
from fastapi.testclient import TestClient

from app import config
from app.agent.turn import _clamp_messages, run_turn
from app.main import app

client = TestClient(app)
REQUIRED = {"reply", "recommendations", "end_of_conversation"}


def test_oversized_body_degrades_to_valid_200():
    huge = "x" * (config.MAX_BODY_BYTES + 100)
    r = client.post("/chat", json={"messages": [{"role": "user", "content": huge}]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == REQUIRED and body["recommendations"] == []


def test_malformed_content_length_degrades_to_valid_200():
    r = client.post(
        "/chat",
        content=b'{"messages": []}',
        headers={"content-type": "application/json", "content-length": "not-a-number"},
    )
    assert r.status_code == 200 and set(r.json()) == REQUIRED


def test_clamp_bounds_history_and_content():
    many = [{"role": "user", "content": "y" * (config.MAX_CONTENT_CHARS + 5000)}] * (
        config.MAX_MESSAGES + 20
    )
    out = _clamp_messages(many)
    assert len(out) == config.MAX_MESSAGES
    assert all(len(m["content"]) <= config.MAX_CONTENT_CHARS for m in out)


def test_clamp_keeps_latest_messages():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(config.MAX_MESSAGES + 3)]
    out = _clamp_messages(msgs)
    assert out[-1]["content"] == msgs[-1]["content"]


def test_normal_input_is_untouched():
    msgs = [
        {"role": "user", "content": "hiring a java developer"},
        {"role": "assistant", "content": "What seniority?"},
        {"role": "user", "content": "mid-level"},
    ]
    assert _clamp_messages(msgs) == msgs


def test_giant_message_still_returns_valid_response():
    resp = run_turn([{"role": "user", "content": "java developer " * 5000}])
    assert isinstance(resp.reply, str) and len(resp.recommendations) <= config.MAX_RECS
