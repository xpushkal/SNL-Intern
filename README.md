---
title: SHL Assessment Recommender
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Conversational SHL Assessment Recommender

A stateless conversational agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of **SHL assessments** —
clarifying when needed, refining on request, comparing on demand, and refusing
anything outside the SHL catalog. Built for the SHL AI Intern take-home.

```
POST /chat ──► FastAPI (strict schema, never-500) ──► LangGraph agent
                                                         understand ─► act ─► respond
                                                              │          │
                                                   Groq (LLM, optional)  │
                                                              ▼          ▼
                                              hybrid retrieval over the full catalog
                                              BM25 ⊕ dense (bge-small) → RRF → top-10
```

## Highlights

- **Reliability-first.** Every response is schema-valid; request-validation errors,
  agent exceptions, and timeouts all degrade to a valid 200 (no 4xx/5xx ever).
- **Deterministic by default.** The agent clarifies, recommends, refines, compares, and
  refuses fully deterministically — no LLM required. On the public traces this measured
  *higher* Recall@10 (**0.575**) than the LLM route, at ~0.05s/turn. The Groq LLM is an
  opt-in enhancement (`ENABLE_LLM=true`) for scope/routing robustness.
- **Hard constraints are inviolable.** Duration caps / required languages / required
  types filter; soft preferences only re-rank. The shortlist returns <10 rather than
  relax a hard constraint.
- **Grounded by construction.** The catalog is the only source of truth; every
  returned `name`/`url`/`test_type` is validated against it (no hallucinated items),
  and URLs are returned verbatim.
- **Measured, not assumed.** MMR and LLM-rerank are behind flags, **off by default**,
  enabled only if the ablation shows a Recall@10 gain.

## API

`GET /health` → `{"status": "ok"}` (200)

`POST /chat` — stateless; send the full history each call:

```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}'
```

```json
{
  "reply": "Here are 10 SHL assessments that fit:\n1. Java 8 (New) (Type K, 30 minutes) — https://www.shl.com/...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

`recommendations` is empty while clarifying or refusing, and 1–10 items once the agent
commits. `end_of_conversation` is `true` only when the user explicitly ends the task.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build catalog + retrieval artifacts (one-time; baked into the image for deploy)
python -m app.data.ingest
python -m app.data.build_index

# (optional) enable the LLM routing path
echo "GROQ_API_KEY=gsk_..." > .env
echo "ENABLE_LLM=true" >> .env

# Run
uvicorn app.main:app --port 8000
```

Then open **http://localhost:8000** for a minimal chat demo UI, or
**http://localhost:8000/docs** for the interactive API console. (`/` and `/docs` are
convenience surfaces; the graded API is `/health` + `/chat`.)

## Evaluation

```bash
python -m eval.replay     # multi-turn Recall@10 + latency + 6 behavior probes
python -m eval.ablation   # BM25 vs dense vs hybrid vs +MMR Recall@10 table
pytest -q                 # 22 unit/contract tests
```

See [docs/APPROACH.md](docs/APPROACH.md) for the 2-page design write-up,
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the deep dive, and
[docs/DECISIONS.md](docs/DECISIONS.md) for the trade-off log.

## Project layout

```
app/        FastAPI + LangGraph agent + retrieval + data pipeline
artifacts/  baked catalog + embeddings + BM25
eval/       replay harness, ablation, probes, the 10 sample traces
tests/      pytest suite
docs/       APPROACH / ARCHITECTURE / DECISIONS
```

## Stack

Groq (Llama-3.3-70B) · `BAAI/bge-small-en-v1.5` local embeddings · BM25 (`rank_bm25`) ·
FastAPI + Pydantic v2 · LangGraph · rapidfuzz · NumPy. Target host: Hugging Face Spaces (Docker).
