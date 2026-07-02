# Hugging Face Spaces (Docker SDK). Python pinned to 3.11 for ML wheels.
FROM python:3.11-slim

# Bake all model/artifact caches into the image -> ZERO downloads at runtime
# (fast, deterministic cold start within the evaluator's 2-min health window).
ENV HF_HOME=/app/.cache/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user (defense in depth: a compromised process cannot write
# outside /app or install anything). UID 1000 matches HF Spaces' recommendation.
RUN useradd --create-home --uid 1000 appuser && chown appuser:appuser /app
COPY --chown=appuser:appuser . .
USER appuser

# Normalize the catalog, download the embedding model, and build the retrieval
# artifacts at BUILD time so the running container needs no network.
RUN python -m app.data.ingest && python -m app.data.build_index

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
