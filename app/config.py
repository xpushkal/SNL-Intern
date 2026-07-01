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
# Routing/extraction is an easy task: an 8B model does it well, is much faster, and has
# a far larger free-tier token budget than 70B (measured: 70B exhausts 100k TPD in one
# eval replay). 70B remains available via env override for the grounded compare summary.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_COMPARE_MODEL = os.getenv("GROQ_COMPARE_MODEL", GROQ_MODEL)
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

# Embedding scope guard (semantic off-topic refusal on top of keyword injection checks).
ENABLE_SCOPE_GUARD = _flag("ENABLE_SCOPE_GUARD", True)
SCOPE_OUT_MIN = _num("SCOPE_OUT_MIN", 0.55)   # min out-of-scope similarity to consider
SCOPE_MARGIN = _num("SCOPE_MARGIN", 0.06)     # how much out must beat in to refuse

# Family/variant boost: lift an instrument's sibling reports toward it (SHL groups an
# instrument with several report variants that often co-occur in gold shortlists).
ENABLE_FAMILY_BOOST = _flag("ENABLE_FAMILY_BOOST", False)
FAMILY_BOOST = _num("FAMILY_BOOST", 0.5)  # fraction of the family's best score

# --- Feature flags (OFF until measured) ----------------------------------
# LLM routing is OFF by default: on the public traces the deterministic core measured
# HIGHER Recall@10 (0.575 vs ~0.35) with far lower latency, zero token cost, and 6/6
# probes. The LLM stays one env var away (ENABLE_LLM=true) for scope/routing robustness.
ENABLE_LLM = _flag("ENABLE_LLM", False)
ENABLE_LLM_RERANK = _flag("ENABLE_LLM_RERANK", False)
RERANK_POOL = int(_num("RERANK_POOL", 25))         # candidates fed to the reranker
ENABLE_MMR = _flag("ENABLE_MMR", False)
# Relevance-dominant lambda: MMR acts only as a weak diversity tie-breaker.
MMR_LAMBDA = _num("MMR_LAMBDA", 0.8)
