# Decision Log (ADRs)

Short, interview-defensible records of the non-obvious choices.

### 1. Treat the provided JSON as the catalog universe (no website scraping)
**Decision.** Use the supplied `shl_product_catalog.json` (377 records) as the single
source of truth; do not scrape shl.com.
**Why.** Verified empirically that all 10 sample-trace gold shortlists are 100% present
in this file — it *is* the recommendable universe. Scraping would add fragility for no
recall benefit. The "verify-first" branch of the plan resolved to "no scrape."

### 2. Derive `test_type` from `keys[]`
**Decision.** Map SHL category names → single-letter codes; keep all codes, pick one
primary by documented priority `K>A>P>B>C>S>E>D` for the 39 multi-category records.
**Why.** The schema requires `test_type`, the data has none, and the traces confirm the
mapping (`Personality & Behavior` → `P`). Categories are listed alphabetically, so
source order is not a reliable "primary" signal — hence an explicit priority.

### 3. Exclude pre-packaged Job Solutions via a narrow, auditable rule
**Decision.** Tag the 7 records ending in "… Solution" as `is_individual=False` and
exclude them from recommendations (kept in the catalog for validation).
**Why.** The assignment scopes pre-packaged Job Solutions out. These 7 are an exact,
inspectable set and appear in no gold shortlist, so exclusion is recall-safe and
scope-correct — not a fuzzy guess.

### 4. The LLM is an enhancement, not a dependency
**Decision.** A deterministic keyword router fully implements every behavior; the Groq
`understand` call overrides it when available.
**Why.** Guarantees the service works (and passes hard evals) even with no key, a bad
key, rate limits, or an outage. Reliability is scored on *every* turn.

### 5. Cross-turn state lives in the reply text
**Decision.** Print each item's URL in the numbered reply; reconstruct the prior
shortlist by parsing those URLs from the last assistant message.
**Why.** The wire schema is fixed and the API is stateless — the reply is the only
channel that survives to the next turn. URLs are unambiguous catalog keys.

### 6. `end_of_conversation` only on explicit completion
**Decision.** Default `false`; `true` only when the user explicitly ends ("perfect,
that's what we need", "thanks, that's all").
**Why.** Matches the sample traces (recommend turns are `false`; the closing turn is
`true`) and keeps refine flows open. Confirmed against C1/C10.

### 7. Rank the whole catalog; no FAISS
**Decision.** Brute-force dense (`embeddings @ query`) + BM25 over all records.
**Why.** ~370 tiny docs → exact scoring is instant and removes a heavyweight,
wheel-finicky dependency. Larger candidate pools are free, which protects recall.

### 8. Hybrid (RRF) baseline; MMR and LLM-rerank behind flags, OFF by default
**Decision.** Default retrieval is BM25⊕dense via RRF. `ENABLE_MMR` and
`ENABLE_LLM_RERANK` are off.
**Why.** The ablation (`python -m eval.ablation`) shows hybrid (0.551) > dense (0.468)
> BM25 (0.373), and MMR adds nothing (0.551). We only turn a flag on if measured.

### 9. Asymmetric schema strictness
**Decision.** Output models use `extra="forbid"` and a catalog-validation gate; input
models are lenient (ignore unknown fields, soft role handling).
**Why.** Strict output passes the hard evals; lenient input prevents an automatic 422
on a slightly-off request (a failed turn we couldn't intercept).

### 10. At most one clarification, with a turn-budget backstop
**Decision.** Clarify only when truly vague, exactly once; if we've already clarified or
history nears the 8-message cap, commit to a shortlist.
**Why.** Balances the two opposing probes ("no turn-1 recommend for vague" vs.
turn-starvation → empty final shortlist → Recall 0). Prior-clarification detection uses
our own no-shortlist/non-refusal marker, not a naive "?" check.

### 11. Python 3.11 + Docker pin
**Decision.** Develop and deploy on Python 3.11.
**Why.** The system default (3.14) lacks wheels for torch/sentence-transformers; 3.11
is fully supported and matches the deployment image.
