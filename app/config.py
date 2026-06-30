"""Central configuration and feature flags.

Every tunable lives here so behaviour is explicit, measurable, and overridable via
environment variables (12-factor). Feature flags for MMR / LLM-rerank default to OFF;
they are only switched on when the ablation harness shows a Recall@10 gain.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load a local .env during development. Never required at runtime.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a dev convenience only
    pass


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
RAW_CATALOG_PATH = Path(
    os.getenv("RAW_CATALOG_PATH", ROOT / "app" / "data" / "shl_product_catalog.json")
)
NORMALIZED_PATH = ARTIFACTS_DIR / "catalog.normalized.json"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
BM25_PATH = ARTIFACTS_DIR / "bm25.pkl"
ALIASES_PATH = ARTIFACTS_DIR / "aliases.json"

# --- Models --------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Latency budget (the evaluator caps each /chat call at 30s) ----------
LLM_TIMEOUT_S = _num("LLM_TIMEOUT_S", 10.0)        # per LLM call
TOTAL_TIMEOUT_S = _num("TOTAL_TIMEOUT_S", 25.0)    # whole-turn safety cap

# --- Conversation policy -------------------------------------------------
MAX_RECS = int(_num("MAX_RECS", 10))               # schema hard cap
RECOMMEND_FILL = int(_num("RECOMMEND_FILL", 10))   # default shortlist size
CLARIFY_MAX = int(_num("CLARIFY_MAX", 1))          # at most one clarification, ever
TURN_HARD_CAP = int(_num("TURN_HARD_CAP", 8))      # evaluator message cap
# When history length reaches this, stop clarifying and commit to a shortlist.
TURN_SOFT_LIMIT = int(_num("TURN_SOFT_LIMIT", 5))

# --- Retrieval -----------------------------------------------------------
RRF_K = int(_num("RRF_K", 60))                     # reciprocal-rank-fusion constant
SOFT_BOOST = _num("SOFT_BOOST", 0.30)              # weight of soft-preference signals
FUZZY_THRESHOLD = int(_num("FUZZY_THRESHOLD", 85)) # rapidfuzz name-match cutoff

# --- Scope ---------------------------------------------------------------
# Pre-packaged Job Solutions are out of scope (assignment). Verified: the 7 such
# records all end in " Solution" and none appear in any gold shortlist.
INCLUDE_PREPACKAGED = _flag("INCLUDE_PREPACKAGED", False)

# --- Feature flags (OFF until measured) ----------------------------------
ENABLE_LLM_RERANK = _flag("ENABLE_LLM_RERANK", False)
ENABLE_MMR = _flag("ENABLE_MMR", False)
# Relevance-dominant lambda: MMR acts only as a weak diversity tie-breaker.
MMR_LAMBDA = _num("MMR_LAMBDA", 0.8)
