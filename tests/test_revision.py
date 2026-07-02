"""The pinned embedding revision is passed to both loaders and enforced by the manifest."""
import inspect

import numpy as np
import pytest

from app import config
from app.data.catalog import load_catalog
from app.retrieval import embedding, store


def test_embed_revision_is_immutable_sha():
    # Not the mutable "main"; a 40-char commit SHA.
    assert config.EMBED_REVISION != "main"
    assert len(config.EMBED_REVISION) == 40 and config.EMBED_REVISION.isalnum()


def test_load_embed_model_pins_revision(monkeypatch):
    captured = {}

    class FakeST:
        def __init__(self, name, revision=None, **kw):
            captured["name"] = name
            captured["revision"] = revision

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    embedding.load_embed_model()
    assert captured["name"] == config.EMBED_MODEL
    assert captured["revision"] == config.EMBED_REVISION


def test_runtime_loader_uses_pinned_loader(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(embedding, "load_embed_model", lambda: sentinel)
    emb = np.load(config.EMBEDDINGS_PATH)
    r = store.Retriever(load_catalog(), emb, bm25=None)
    assert r.model is sentinel  # runtime path goes through the pinned loader


def test_build_index_uses_pinned_loader():
    from app.data import build_index

    assert "load_embed_model" in inspect.getsource(build_index.main)


def test_manifest_detects_revision_mismatch(monkeypatch):
    emb, cat = np.load(config.EMBEDDINGS_PATH), load_catalog()
    monkeypatch.setattr(config, "EMBED_REVISION", "0000000000000000000000000000000000000000")
    with pytest.raises(RuntimeError, match="embedding_revision"):
        store._validate_manifest(emb, cat)
