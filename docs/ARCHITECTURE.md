# Architecture

## Request lifecycle

```
POST /chat  (stateless: full history every call)
   │
   ▼
FastAPI layer (app/main.py)
   • lenient request parsing (unknown fields ignored → never a 422)
   • run pipeline in a worker thread under a whole-turn timeout
   • RequestValidationError + Exception handlers → schema-valid 200
   ▼
Turn orchestrator (app/agent/turn.py) — defense-in-depth try/except
   ▼
LangGraph: understand → act → respond  (app/agent/graph.py)
   │
   ├─ understand (app/agent/understand.py)
   │     deterministic keyword router by default (vagueness-aware) →
   │     {in_scope, intent, search_query, hard, soft, compare_names,
   │      remove_names, add_query, clarifying_question, user_done}
   │     optional (ENABLE_LLM): one Groq JSON call over the WHOLE history,
   │     with the deterministic router as the fallback on any failure
   │
   ├─ act (app/agent/dispatch.py) — deterministic
   │     refuse | clarify(≤1) | recommend | refine | compare
   │     reconstructs prior shortlist from the last assistant reply
   │
   └─ respond (app/agent/render.py + app/responder.py)
         templated reply + anti-hallucination/schema gate
   ▼
Retrieval (app/retrieval/*) — rank the FULL catalog
   HARD filters (never relaxed) → RRF(BM25, dense) → SOFT boosts → top-10
   [MMR / LLM-rerank optional, OFF by default]
```

## Why these choices

**Stateless state lives in the reply.** The wire schema is fixed
(`reply`, `recommendations`, `end_of_conversation`), and the grader replays the full
history each call. So the **numbered list embedded in the reply text is the only
cross-turn state carrier**. Every recommendation line prints its catalog URL;
`render.parse_prior_shortlist` reads those URLs back out of the most recent assistant
message to reconstruct the prior shortlist (URLs are unambiguous keys, unlike names
with parentheses). This is what makes "remove the second one" work without a database.

**≤2 LLM calls per turn.** One `understand` call routes + extracts; replies are
templated; only `compare` (and the optional reranker) make a second call. Groq is
sub-second, so a turn finishes well inside the 30s cap with timeout headroom.

**Rank the whole catalog.** With ~370 recommendable items, there is no need for an ANN
index or a candidate cap — we score every document with a brute-force matrix multiply
(dense, `embeddings @ query`) plus BM25, fuse with Reciprocal Rank Fusion, then add
soft boosts. Exact, instant, and one fewer dependency (no FAISS).

**Hard vs soft.** `understand` tags each constraint. Hard constraints (`max_duration`,
required `languages`, required `test_types`) are applied as filters *before* scoring
and are never relaxed — if fewer than 10 survive, we return fewer than 10. Soft
preferences (`job_levels`, preferred types) only add a small ranking boost and can
never exclude an item, which protects recall.

## Reliability / fallback matrix

| Failure | Handling | Result |
|---|---|---|
| Malformed request body | lenient schema + `RequestValidationError` handler | 200, safe reply |
| `understand` LLM error/timeout | 1 stricter retry → deterministic router | valid route |
| Deterministic router on vague input | vagueness check → clarify (not blind recommend) | empty recs, clarify |
| Rerank disabled/fails | deterministic fusion order | top-10 |
| Compare LLM fails | deterministic field-by-field template | grounded compare |
| Off-catalog / duplicate item | `responder` canonicalization gate | dropped |
| Whole turn exceeds budget | `asyncio.wait_for` in `main.py` | 200, safe reply |

## test_type derivation

The catalog has no `test_type`; its `keys[]` are SHL category names (trace-confirmed:
`Personality & Behavior` → `P`). `app/data/test_type_map.py` maps each category to its
letter and exposes all codes plus a single primary. 338/377 records have one category;
for the 39 multi-category records the catalog lists categories alphabetically (so order
is not meaningful) and we apply a documented priority `K>A>P>B>C>S>E>D`.

## Scope

Verified empirically: all 10 sample-trace gold shortlists are 100% present in the
provided catalog, so it *is* the recommendable universe (no website scraping). The 7
pre-packaged Job Solutions (all literally "… Solution") are tagged `is_individual=False`
and excluded from recommendations, but remain in the catalog for validation.
