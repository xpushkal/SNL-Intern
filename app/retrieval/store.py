"""Retriever store: loads baked artifacts once and exposes score primitives.

The catalog is tiny (377 docs), so we score *every* document with a brute-force
matrix multiply (dense) and BM25 -- no ANN index, no candidate cap. This is exact,
instant, and removes a heavy dependency (FAISS).
"""
from __future__ import annotations

import os

# Serving uses the BAKED embedding model only -- never phone home to the Hugging Face
# Hub. Without this, sentence-transformers issues a blocking HEAD request on first use
# that can hang ~30s (and break) when the network is slow/unavailable. The model is
# downloaded at build time (app.data.build_index), not here, so offline is always safe.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import pickle
import threading

import numpy as np

from app import config
from app.data.catalog import Catalog, load_catalog
from app.retrieval.text import QUERY_INSTRUCTION, tokenize


def _validate_manifest(embeddings: np.ndarray, catalog: Catalog) -> None:
    """Fail fast (with actionable guidance) if the baked artifacts don't match the
    current embedding model / catalog / doc-schema -- prevents silent mismatches."""
    from app.data.ingest import DOC_SCHEMA_VERSION, catalog_hash

    rebuild = "Rebuild artifacts:  python -m app.data.build_index"
    if not config.MANIFEST_PATH.exists():
        raise RuntimeError(f"Artifact manifest missing at {config.MANIFEST_PATH}. {rebuild}")
    man = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    checks = {
        "embedding_model": (man.get("embedding_model"), config.EMBED_MODEL),
        "embedding_revision": (man.get("embedding_revision"), config.EMBED_REVISION),
        "vector_dim": (int(man.get("vector_dim", -1)), int(embeddings.shape[1])),
        "doc_schema_version": (man.get("doc_schema_version"), DOC_SCHEMA_VERSION),
        "catalog_sha256": (man.get("catalog_sha256"), catalog_hash(catalog.records)),
    }
    mismatches = [f"{k}: manifest={a!r} current={b!r}" for k, (a, b) in checks.items() if a != b]
    if mismatches:
        raise RuntimeError(
            "Artifact/model mismatch — baked embeddings are incompatible.\n  - "
            + "\n  - ".join(mismatches) + f"\n{rebuild}"
        )


class Retriever:
    def __init__(self, catalog: Catalog, embeddings: np.ndarray, bm25):
        self.catalog = catalog
        self.embeddings = embeddings  # (N, d), L2-normalized
        self.bm25 = bm25
        self._model = None  # lazy: only loaded when a dense query is needed
        assert embeddings.shape[0] == len(catalog.records), "artifact/catalog mismatch"

    @property
    def model(self):
        if self._model is None:
            from app.retrieval.embedding import load_embed_model

            self._model = load_embed_model()  # pinned revision
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            QUERY_INSTRUCTION + (query or ""),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return vec

    def dense_scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.embed_query(query)  # cosine (normalized)

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")


_LOCK = threading.Lock()
_INSTANCE: Retriever | None = None


def load_retriever() -> Retriever:
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                if not config.EMBEDDINGS_PATH.exists() or not config.BM25_PATH.exists():
                    raise FileNotFoundError(
                        "Retrieval artifacts missing. Run `python -m app.data.build_index`."
                    )
                embeddings = np.load(config.EMBEDDINGS_PATH)
                with open(config.BM25_PATH, "rb") as fh:
                    bm25 = pickle.load(fh)
                catalog = load_catalog()
                _validate_manifest(embeddings, catalog)
                _INSTANCE = Retriever(catalog, embeddings, bm25)
    return _INSTANCE
