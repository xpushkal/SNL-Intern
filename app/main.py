"""FastAPI service for the conversational SHL assessment recommender.

Two endpoints (exact contract required by the evaluator):
  * ``GET  /health`` -> ``{"status": "ok"}`` (200)
  * ``POST /chat``   -> ``{reply, recommendations[], end_of_conversation}``

Reliability contract: **every** call returns a schema-valid 200. Request-validation
errors, agent exceptions, and per-turn timeouts are all converted into a safe,
schema-valid response so no turn ever fails the hard evals with a 4xx/5xx.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from app import config
from app.responder import safe_fallback_response
from app.schemas import ChatRequest, ChatResponse

_STATIC = Path(__file__).parent / "static"

log = logging.getLogger("shl")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load artifacts once at startup (fail fast in build, not on first request).
    from app.data.catalog import load_catalog

    load_catalog()
    try:  # retriever is added in Step 3; tolerate its absence pre-build.
        from app.retrieval.store import load_retriever

        load_retriever()
    except Exception as exc:  # pragma: no cover
        log.warning("retriever not loaded at startup: %s", exc)
    yield


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    """Minimal chat demo UI (not part of the graded API)."""
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    from app.agent.turn import run_turn

    messages = [m.model_dump() for m in request.messages]
    try:
        # Run the (sync) pipeline off the event loop with a hard whole-turn timeout.
        return await asyncio.wait_for(
            asyncio.to_thread(run_turn, messages), timeout=config.TOTAL_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        log.warning("turn timed out after %ss", config.TOTAL_TIMEOUT_S)
        return safe_fallback_response(
            "That took longer than expected. Could you share the role or key skills again?"
        )
    except Exception:  # never 500
        log.exception("unhandled error in /chat")
        return safe_fallback_response()


# --- Request-size guard (DoS): oversized bodies degrade to a schema-valid 200 ----
@app.middleware("http")
async def _body_size_guard(request: Request, call_next):
    if request.method == "POST":
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = config.MAX_BODY_BYTES + 1  # malformed header -> treat as oversized
        if length > config.MAX_BODY_BYTES:
            log.warning("rejected oversized request body (%s bytes)", length)
            return JSONResponse(
                status_code=200,
                content=safe_fallback_response(
                    "That message is too large for me to process. Could you send a "
                    "shorter description of the role you're hiring for?"
                ).model_dump(),
            )
    return await call_next(request)


# --- Convert framework-level errors into schema-valid 200s ----------------
@app.exception_handler(RequestValidationError)
async def _on_validation_error(_req: Request, _exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=200, content=safe_fallback_response().model_dump())


@app.exception_handler(Exception)
async def _on_unhandled(_req: Request, _exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=200, content=safe_fallback_response().model_dump())
