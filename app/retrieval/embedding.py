"""Shared embedding-model loader.

Both the offline index build and the runtime retriever load the model through this one
function so the pinned revision (an immutable commit SHA) is applied identically in both
places -- guaranteeing the baked vectors and the query encoder are the same weights.
"""
from __future__ import annotations

from app import config


def load_embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL, revision=config.EMBED_REVISION)
