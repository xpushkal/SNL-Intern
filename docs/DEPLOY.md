# Deployment — Hugging Face Spaces (Docker)

The service is container-ready. The image bakes the embedding model and retrieval
artifacts at build time, so there are **no runtime downloads** and cold start fits the
evaluator's 2-minute health window.

## One-time

1. Create a new **Space** → SDK: **Docker** → blank.
2. Add the Space's YAML header to the top of `README.md` (HF requires it):

   ```yaml
   ---
   title: SHL Assessment Recommender
   emoji: 🧭
   colorFrom: indigo
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Set a **Space secret**: `GROQ_API_KEY = gsk_...`
   (Optional: `GROQ_MODEL`, `ENABLE_LLM_RERANK`, etc. — see `app/config.py`.)

## Push

```bash
git remote add space https://huggingface.co/spaces/<user>/<space>
git push space main
```

The build runs `python -m app.data.ingest && python -m app.data.build_index`
(downloads `bge-small`, builds embeddings + BM25) and starts uvicorn on port 7860.

## Verify

```bash
curl https://<user>-<space>.hf.space/health        # {"status":"ok"}
curl -s https://<user>-<space>.hf.space/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hiring a java developer"}]}'
```

## Submission checklist
1. **Public endpoint** — `/health` and `/chat` reachable on the Space URL (above).
2. **Approach PDF (≤2 pages)** — `make approach-html` then open `docs/APPROACH.html`
   and *Save as PDF* (or `pandoc … --pdf-engine=weasyprint` if installed). Source of
   truth is `docs/APPROACH.md`.
3. Paste both into the submission form.

## Notes
- **Free-tier token budget.** Routing uses `llama-3.1-8b-instant` (fast, large free
  TPD). If Groq rate-limits, the agent degrades to the deterministic path — still valid
  and grounded — so the endpoint never fails.
- **Local Docker test:** `docker build -t shl . && docker run -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY shl`
