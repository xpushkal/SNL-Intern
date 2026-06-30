"""Rank-the-whole-catalog hybrid retrieval.

Pipeline (per the plan):
  HARD filters (never relaxed) -> score every survivor with RRF(BM25, dense)
  -> add SOFT-preference boosts (ranking-only) -> sort -> top-k.

If fewer than k survivors pass the hard filters we return fewer than k -- hard
constraints are inviolable, soft preferences only reorder. ``mode`` exposes
BM25-only / dense-only / hybrid for the ablation harness.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from app import config
from app.retrieval.store import Retriever, load_retriever

Mode = Literal["bm25", "dense", "hybrid"]


# --- constraint predicates ----------------------------------------------
def _hard_ok(rec: dict, hard: dict) -> bool:
    """A record passes only if it satisfies *every* hard constraint."""
    if not rec.get("is_individual") and not config.INCLUDE_PREPACKAGED:
        return False
    max_dur = hard.get("max_duration_minutes")
    if max_dur is not None:
        dur = rec.get("duration_minutes")
        if dur is None or dur > max_dur:  # unknown duration cannot be guaranteed
            return False
    langs = hard.get("languages") or []
    if langs:
        rec_langs = " | ".join(rec.get("languages", [])).lower()
        if not any(l.lower() in rec_langs for l in langs):
            return False
    types = hard.get("test_types") or []
    if types and not (set(types) & set(rec.get("test_types", []))):
        return False
    return True


def _soft_boost(rec: dict, soft: dict) -> float:
    """A small ranking-only nudge in [0, SOFT_BOOST]; never excludes anything."""
    signals: list[float] = []
    levels = [l.lower() for l in (soft.get("job_levels") or [])]
    if levels:
        rec_levels = {l.lower() for l in rec.get("job_levels", [])}
        signals.append(1.0 if any(l in rec_levels for l in levels) else 0.0)
    pref_types = soft.get("test_types") or []
    if pref_types:
        signals.append(1.0 if set(pref_types) & set(rec.get("test_types", [])) else 0.0)
    if not signals:
        return 0.0
    return config.SOFT_BOOST * (sum(signals) / len(signals))


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Map scores -> 0-based rank position (rank 0 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


def base_scores(retr: Retriever, query: str, mode: Mode) -> np.ndarray:
    if mode == "bm25":
        return retr.bm25_scores(query)
    if mode == "dense":
        return retr.dense_scores(query)
    # hybrid: reciprocal-rank fusion of the two rankings
    k = config.RRF_K
    rd = _ranks(retr.dense_scores(query))
    rs = _ranks(retr.bm25_scores(query))
    return 1.0 / (k + rd) + 1.0 / (k + rs)


def search(
    query: str,
    hard: dict | None = None,
    soft: dict | None = None,
    k: int | None = None,
    mode: Mode = "hybrid",
    apply_mmr: bool | None = None,
) -> list[dict]:
    """Return up to ``k`` catalog records, best first, satisfying all hard constraints."""
    hard = hard or {}
    soft = soft or {}
    k = k or config.RECOMMEND_FILL
    retr = load_retriever()
    records = retr.catalog.records

    base = base_scores(retr, query, mode)
    scored: list[tuple[float, int]] = []
    for i, rec in enumerate(records):
        if not _hard_ok(rec, hard):
            continue
        scored.append((float(base[i]) + _soft_boost(rec, soft), i))
    scored.sort(key=lambda t: t[0], reverse=True)

    use_mmr = config.ENABLE_MMR if apply_mmr is None else apply_mmr
    if use_mmr and mode != "bm25":
        idxs = _mmr([i for _, i in scored], retr.embeddings, k)
    else:
        idxs = [i for _, i in scored[:k]]
    return [records[i] for i in idxs]


def _mmr(candidate_idxs: list[int], embeddings: np.ndarray, k: int) -> list[int]:
    """Maximal Marginal Relevance as a *weak* diversity tie-breaker (lambda ~0.8).

    Candidates arrive already sorted by relevance; MMR only reshuffles near-ties.
    """
    if not candidate_idxs:
        return []
    lam = config.MMR_LAMBDA
    pool = candidate_idxs[: max(k * 3, k)]
    # relevance proxy = position in the incoming (relevance-sorted) order
    rel = {idx: 1.0 - p / max(len(pool), 1) for p, idx in enumerate(pool)}
    selected: list[int] = []
    remaining = list(pool)
    while remaining and len(selected) < k:
        best, best_score = remaining[0], -1e9
        for idx in remaining:
            if selected:
                sims = embeddings[selected] @ embeddings[idx]
                diversity = float(np.max(sims))
            else:
                diversity = 0.0
            score = lam * rel[idx] - (1 - lam) * diversity
            if score > best_score:
                best, best_score = idx, score
        selected.append(best)
        remaining.remove(best)
    return selected
