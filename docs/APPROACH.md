# Approach — Conversational SHL Assessment Recommender

## Problem & design stance
Build a stateless `/chat` agent that moves a hiring manager from a vague intent to a
grounded shortlist of SHL assessments — clarifying, recommending, refining, comparing,
and refusing out-of-scope asks — scored on schema hard-evals, Recall@10, and behavior
probes. My stance: **reliability and grounding first, recall second, cleverness last.**
A turn that 500s or returns a hallucinated item scores zero regardless of how good the
ranking is, so the architecture makes invalid output structurally impossible and treats
the LLM as an *enhancement* over a fully-working deterministic core.

## Data & retrieval
The provided catalog (377 records) has no `test_type`; its `keys[]` are SHL category
names, which the sample traces confirm map to the letter codes (`Personality &
Behavior`→`P`). I derive `test_type` from that mapping (priority `K>A>P>B>C>S>E>D` for
the 39 multi-category records). I verified that **all 10 trace gold shortlists are 100%
in the file**, so it is the recommendable universe — no website scraping. The 7
pre-packaged "… Solution" bundles are scoped out via a narrow, auditable rule.

Retrieval **ranks the entire catalog** (it is tiny): hard filters first (duration cap /
required language / required type — never relaxed), then **Reciprocal Rank Fusion of
BM25 and dense `bge-small` cosine**, then small soft-preference boosts, then top-10.
Soft preferences only re-rank; if fewer than 10 pass the hard filters we return fewer.
No FAISS — a brute-force matrix multiply over ~370 vectors is exact and instant.

## Agent
A 3-node LangGraph (`understand → act → respond`). `understand` is one Groq JSON call
that routes intent and extracts cumulative constraints from the whole history;
`act` is deterministic; replies are templated. Because the wire schema is fixed and the
API is stateless, the **numbered list in the reply is the only cross-turn state** — each
line prints its catalog URL, which I parse back to reconstruct the prior shortlist for
refinements like "remove the second one". Clarification is capped at one (with a
turn-budget backstop so we never starve the 8-turn cap into an empty final shortlist).
`end_of_conversation` is true only on explicit user completion — matching the traces.

## Prompt design
The routing prompt is deliberately small and contract-shaped: it returns a fixed JSON
object (`in_scope, intent, search_query, hard/soft constraints, compare/remove names,
clarifying_question, user_done`) and never writes prose or invents names. Comparison
uses a separate prompt fed **only catalog facts** (no priors), with a deterministic
field-by-field template as fallback.

## Guardrails (anti-hallucination)
Every response passes one gate: each recommendation is canonicalized against the catalog
(URL returned verbatim, type checked), off-catalog and duplicate items are dropped, the
list is capped at 10. Combined with lenient input parsing + global exception/timeout
handlers, **every call returns a schema-valid 200** — request errors, LLM failures, and
timeouts included.

## Evaluation
Three harnesses (`pytest`, `eval.replay`, `eval.ablation`). Replay mirrors the grader:
feed each trace's user turns one at a time, carry full history, score Recall@10 on the
final shortlist, plus latency and 6 behavior probes.

**Retrieval ablation (fixed query):**

| Config | Mean Recall@10 |
|---|---|
| BM25 only | 0.373 |
| Dense only | 0.468 |
| **Hybrid (RRF)** | **0.551** |
| Hybrid + MMR | 0.551 |

**Behavior probes:** 6/6 (refuses off-topic, resists injection, no turn-1 recommend on
vague, honors edits, completion sets end, no hallucinated items).
**Latency:** deterministic p50 ≈ 0.04s; with the 8B router p50 ≈ 0.5s, p95 ≈ 10s — well
inside the 30s cap.

## What didn't work / how I measured
- **MMR** gave **no** Recall@10 gain in the ablation → left **off** by default (flagged).
- **70B routing** exhausted the Groq free-tier daily token budget (100k TPD) in a single
  replay and pushed p95 latency to ~10s → switched routing to **`llama-3.1-8b-instant`**
  (separate, larger budget; faster) and slashed prompt tokens (compact transcript,
  trimmed system prompt, no retry on rate-limits).
- **LLM query distillation** did not beat the deterministic broad query on the
  multi-faceted, semantic traces (e.g. a "senior Rust" query whose gold items are
  inferred), so retrieval uses the full conversation text; the LLM contributes routing,
  constraint typing, and refinement, with the deterministic path as the measured floor
  (replay Recall@10 ≈ 0.47–0.55).

## AI-tool usage
Built with AI-assisted coding (Claude Code) for scaffolding, retrieval/agent plumbing,
and the eval harness; all design decisions (hybrid-rank-all, hard/soft split,
state-in-reply, flags-off-until-measured, model choice) are documented in
`docs/DECISIONS.md` and were driven by the measurements above.

## Stack
Groq `llama-3.1-8b-instant` · `bge-small-en-v1.5` (local, baked) · BM25 · FastAPI +
Pydantic v2 · LangGraph · rapidfuzz · NumPy · Hugging Face Spaces (Docker).
