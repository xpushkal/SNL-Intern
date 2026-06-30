"""Retriever store: loads baked artifacts once and exposes score primitives.

The catalog is tiny (377 docs), so we score *every* document with a brute-force
matrix multiply (dense) and BM25 -- no ANN index, no candidate cap. This is exact,
instant, and removes a heavy dependency (FAISS).
"""
from __future__ import annotations

import pickle
import threading

import numpy as np

from app import config
from app.data.catalog import Catalog, load_catalog
from app.retrieval.text import QUERY_INSTRUCTION, tokenize


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
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(config.EMBED_MODEL)
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
                _INSTANCE = Retriever(load_catalog(), embeddings, bm25)
    return _INSTANCE
