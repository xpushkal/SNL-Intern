"""Embedding scope guard: catches novel off-topic without false-positive refusals."""
from app.agent.turn import run_turn

_REFUSAL = "I can only help you choose SHL assessments"


def _refused(messages):
    return _REFUSAL in run_turn(messages).reply


def test_novel_off_topic_refused():
    # Not in the keyword lists -> only the semantic guard can catch these.
    assert _refused([{"role": "user", "content": "recommend a good sushi restaurant"}])
    assert _refused([{"role": "user", "content": "what movie should I watch tonight"}])


def test_valid_unusual_roles_not_refused():
    for content in (
        "assessment for a Kubernetes platform engineer",
        "hiring a wind turbine technician",
    ):
        assert not _refused([{"role": "user", "content": content}])


def test_conversational_followups_not_refused():
    base = run_turn([{"role": "user", "content": "graduate trainee battery"}])
    for content in ("perfect, thanks", "remove the second one", "add a personality test"):
        hist = [
            {"role": "user", "content": "graduate trainee battery"},
            {"role": "assistant", "content": base.reply},
            {"role": "user", "content": content},
        ]
        assert not _refused(hist)
