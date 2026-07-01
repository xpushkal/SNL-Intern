"""Artifact manifest guards against embedding-model / catalog / schema mismatches."""
import numpy as np
import pytest

from app import config
from app.data.catalog import load_catalog
from app.retrieval import store


def _load():
    return np.load(config.EMBEDDINGS_PATH), load_catalog()


def test_manifest_exists_and_matches():
    emb, cat = _load()
    store._validate_manifest(emb, cat)  # should not raise for freshly-built artifacts


def test_manifest_detects_model_mismatch(monkeypatch):
    emb, cat = _load()
    monkeypatch.setattr(config, "EMBED_MODEL", "some/other-embedding-model")
    with pytest.raises(RuntimeError, match="Artifact/model mismatch"):
        store._validate_manifest(emb, cat)


def test_manifest_detects_dimension_mismatch():
    _, cat = _load()
    wrong_dim = np.zeros((377, 999), dtype="float32")
    with pytest.raises(RuntimeError, match="vector_dim"):
        store._validate_manifest(wrong_dim, cat)


def test_manifest_missing_raises(monkeypatch, tmp_path):
    emb, cat = _load()
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "nope.json")
    with pytest.raises(RuntimeError, match="manifest missing"):
        store._validate_manifest(emb, cat)
