"""Build and persist the retrieval artifacts (run at image-build time).

    python -m app.data.build_index

Produces, aligned 1:1 with ``catalog.normalized.json`` order:
  * ``embeddings.npy`` -- L2-normalized bge-small passage embeddings (float32).
  * ``bm25.pkl``       -- a fitted BM25Okapi over the same documents.

The dense model is downloaded here so the runtime image can run fully offline
(no Hugging Face Hub calls on cold start).
"""
from __future__ import annotations

import pickle

import numpy as np

from app import config
from app.data.catalog import load_catalog
from app.data.ingest import doc_text
from app.retrieval.text import tokenize


def main() -> None:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi

    catalog = load_catalog()
    docs = [doc_text(r) for r in catalog.records]

    print(f"embedding {len(docs)} documents with {config.EMBED_MODEL} ...")
    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode(
        docs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

    print("fitting BM25 ...")
    bm25 = BM25Okapi([tokenize(d) for d in docs])

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, embeddings)
    with open(config.BM25_PATH, "wb") as fh:
        pickle.dump(bm25, fh)

    print(f"  embeddings: {embeddings.shape} -> {config.EMBEDDINGS_PATH}")
    print(f"  bm25:       {len(docs)} docs -> {config.BM25_PATH}")


if __name__ == "__main__":
    main()
