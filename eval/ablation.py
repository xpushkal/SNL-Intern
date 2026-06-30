"""Retrieval ablation: compare Recall@10 across retrieval configurations.

Uses a fixed query per trace (concatenated user turns) so the comparison isolates the
*retrieval* configuration from agent/LLM behavior. Decides whether the MMR / rerank
flags earn their place.

    python -m eval.ablation
"""
from __future__ import annotations

import statistics

from app.retrieval import ranking
from eval.common import parse_trace, recall_at_k, trace_files


def _mean_recall(mode: str, apply_mmr: bool) -> float:
    recalls = []
    for path in trace_files():
        tr = parse_trace(path)
        query = " ".join(tr["user_turns"])
        res = ranking.search(query, k=10, mode=mode, apply_mmr=apply_mmr)
        recalls.append(recall_at_k(tr["gold"], [r["url"] for r in res], 10))
    return statistics.mean(recalls)


def main() -> None:
    configs = [
        ("BM25 only", "bm25", False),
        ("Dense only", "dense", False),
        ("Hybrid (RRF)", "hybrid", False),
        ("Hybrid + MMR", "hybrid", True),
    ]
    print(f"{'configuration':20s} {'mean Recall@10':>15s}")
    print("-" * 36)
    rows = []
    for label, mode, mmr in configs:
        score = _mean_recall(mode, mmr)
        rows.append((label, score))
        print(f"{label:20s} {score:>15.3f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest: {best[0]} ({best[1]:.3f})")
    print("(LLM rerank ablation runs separately when GROQ_API_KEY is set.)")


if __name__ == "__main__":
    main()
