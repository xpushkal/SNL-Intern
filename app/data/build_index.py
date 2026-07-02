"""Build and persist the retrieval artifacts (run at image-build time).

    python -m app.data.build_index

Produces, aligned 1:1 with ``catalog.normalized.json`` order:
  * ``embeddings.npy`` -- L2-normalized bge-small passage embeddings (float32).

BM25 is NOT persisted: it is refit from the normalized catalog at load time
(deterministic, ~ms), which avoids shipping a pickle that would execute arbitrary
code if tampered with. The dense model is downloaded here so the runtime image can
run fully offline (no Hugging Face Hub calls on cold start).
"""
from __future__ import annotations

import json

import numpy as np

from app import config
from app.data.catalog import load_catalog
from app.data.ingest import DOC_SCHEMA_VERSION, catalog_hash, doc_text


def main() -> None:
    from app.retrieval.embedding import load_embed_model

    catalog = load_catalog()
    docs = [doc_text(r) for r in catalog.records]

    print(f"embedding {len(docs)} documents with {config.EMBED_MODEL}@{config.EMBED_REVISION} ...")
    model = load_embed_model()  # pinned revision
    embeddings = model.encode(
        docs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, embeddings)

    manifest = {
        "embedding_model": config.EMBED_MODEL,
        "embedding_revision": config.EMBED_REVISION,
        "vector_dim": int(embeddings.shape[1]),
        "num_docs": len(docs),
        "catalog_sha256": catalog_hash(catalog.records),
        "doc_schema_version": DOC_SCHEMA_VERSION,
    }
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"  embeddings: {embeddings.shape} -> {config.EMBEDDINGS_PATH}")
    print(f"  manifest:   {manifest} -> {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
