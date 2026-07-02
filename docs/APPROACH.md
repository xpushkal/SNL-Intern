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
BM25 and dense `bge-small` cosine** (`RRF_K=40`, grid-searched — a stable 30–45 plateau),
then small soft-preference boosts, then top-10. Soft preferences only re-rank; if fewer
than 10 pass the hard filters we return fewer. No FAISS — a brute-force matrix multiply
over ~370 vectors is exact and instant. One domain prior sits on top: expert gold
shortlists pair role-specific skill tests with a broad **personality** instrument
(OPQ32r appears in 7/10 gold sets) that keyword/semantic retrieval never surfaces from a
role query, so a *full* shortlist reserves its last slot for OPQ32r unless a hard
constraint excludes it. This is the single biggest recall lever (below).

## Agent
A 3-node LangGraph (`understand → act → respond`). `understand` routes intent and
extracts cumulative constraints from the whole history — deterministically by default,
or via one Groq JSON call when `ENABLE_LLM=true`; `act` is deterministic; replies are
templated. Because the wire schema is fixed and the
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
Three harnesses (`pytest` — 70 tests, `eval.replay`, `eval.ablation`). Replay mirrors the
grader: feed each trace's user turns one at a time, carry full history, score Recall@10 on
the final shortlist, plus latency and 11 behavior probes (both scripted-final and
grader-like stop-on-first-shortlist modes).

**Retrieval ablation (fixed query, retrieval isolated from agent behavior):**

| Config | Mean Recall@10 |
|---|---|
| BM25 only | 0.387 |
| Dense only | 0.468 |
| **Hybrid (RRF)** | **0.551** |
| Hybrid + MMR | 0.551 |

**Multi-turn replay (default deterministic config):** mean **Recall@10 = 0.733**
(scripted-final) / **0.700** (stop-on-first-shortlist).
**Behavior probes:** 11/11 — refuses off-topic (incl. novel/paraphrased), resists
injection (incl. paraphrased), no turn-1 recommend on vague, honors edits, multi-edit
refine, three-way compare, duration hard-constraint, completion sets end, no hallucinated
items.
**Latency:** deterministic p50 ≈ 0.05s, p95 ≈ 0.12s; with the 8B router p50 ≈ 0.5s (≈9s in
a network-degraded sandbox) — inside the 30s cap either way.

## What didn't work / how I measured
Every flag is off until a measured Recall@10 gain justifies it, and the reverse — every
adopted change moved the replay number:

- **Battery-composition prior (adopted):** reserving a full shortlist's tail slot for
  OPQ32r lifted scripted-final **0.625 → 0.699**; with the `RRF_K=40` retune and two
  state-preservation fixes (below) the config reaches **0.733**.
- **State-preservation fixes (adopted):** a shortlist *question* ("do we really need
  Verify G+?") was rebuilding the list from scratch and discarding accumulated edits — now
  it re-renders unchanged; and "keep <named item>" now adds that exact instrument when
  absent. Both recovered C9 gold (0.57 → 0.71).
- **Plural folding / light stemming (rejected):** *lowered* Recall@10 (0.713 → 0.693) —
  it collided distinct product tokens → left out.
- **MMR (rejected):** **no** ablation gain (0.551 = 0.551) → left **off** (flagged).
- **Family/variant boost (rejected):** *reduced* Recall@10 — a strong family displaces
  other-family gold → left **off**.
- **70B routing** exhausted the Groq free-tier daily token budget (100k TPD) in a single
  replay and pushed p95 latency to ~10s → switched routing to **`llama-3.1-8b-instant`**
  (separate, larger budget; faster) and slashed prompt tokens.
- **LLM routing under-performed and is OFF by default.** The deterministic broad-query
  core out-scores the 8B route (LLM query *distillation* hurt multi-faceted/semantic needs
  like "senior Rust", where gold items are inferred), at ~0.05s/turn vs ~9s and with no
  token/network risk. `ENABLE_LLM` defaults to false (same discipline as MMR/rerank); the
  LLM is one env var away for scope/routing robustness, and any LLM failure falls back to
  this same core.

## Reliability & robustness hardening
Beyond the never-4xx/5xx contract: request bodies are size-capped and history/message
length is clamped (bounded CPU under adversarial input, still degrading to a valid 200);
the BM25 index is refit from the catalog at load rather than unpickled (no
deserialization-of-untrusted-data surface); dependencies are pinned to CVE-clear
versions; and the container runs as a non-root user.

## AI-tool usage
Built with AI-assisted coding (Claude Code and OpenAI Codex) for scaffolding,
retrieval/agent plumbing, the eval harness, deployment verification, and document
polish; all design decisions (hybrid-rank-all, hard/soft split, state-in-reply,
flags-off-until-measured, model choice) are documented in `docs/DECISIONS.md` and were
driven by the measurements above.

## Stack
`bge-small-en-v1.5` (local, baked) + BM25 retrieval · FastAPI + Pydantic v2 · LangGraph ·
rapidfuzz · NumPy · Hugging Face Spaces (Docker) · Groq `llama-3.1-8b-instant` (opt-in).
