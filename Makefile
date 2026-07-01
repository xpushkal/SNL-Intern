.PHONY: setup build run test eval ablation clean

PY ?= .venv/bin/python

setup:           ## create venv + install deps
	python3.11 -m venv .venv && $(PY) -m pip install -U pip -r requirements.txt

build:           ## normalize catalog + build retrieval artifacts
	$(PY) -m app.data.ingest
	$(PY) -m app.data.build_index

run:             ## serve the API on :8000
	$(PY) -m uvicorn app.main:app --port 8000 --reload

test:            ## run the pytest suite (deterministic)
	$(PY) -m pytest tests/ -q

eval:            ## multi-turn Recall@10 + latency + behavior probes
	$(PY) -m eval.replay

ablation:        ## retrieval ablation table (BM25/dense/hybrid/+MMR)
	$(PY) -m eval.ablation

approach-html:   ## render the 2-page approach doc -> print-ready HTML (open, Save as PDF)
	pandoc docs/APPROACH.md -o docs/APPROACH.html --standalone --embed-resources \
	  --css docs/approach.css --metadata title="SHL Assessment Recommender — Approach"

clean:
	rm -rf artifacts/embeddings.npy artifacts/bm25.pkl __pycache__ */__pycache__ .pytest_cache
