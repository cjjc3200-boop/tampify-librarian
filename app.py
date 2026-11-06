"""FastAPI service exposing the TampifyAI Librarian.

Endpoints:
  GET  /health          liveness + index status
  POST /ask             {question, k?} -> grounded answer + sources
  POST /reindex         rebuild the vector index from the brain

The app loads the embedding model and index at startup so user requests only
ever hit the warm (~20ms) path, not the ~7s cold load.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from librarian import answer as answer_mod  # noqa: E402
from librarian import store  # noqa: E402

app = FastAPI(title="TampifyAI Librarian", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the business")
    k: int = Field(4, ge=1, le=20, description="How many chunks to retrieve")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    cost_usd: float


@app.on_event("startup")
def _warm_index():
    """Load the embedding model and open the index at startup.

    Tracing showed the first query otherwise pays a ~7s cold-start cost to
    load the model. Warming here means user requests only ever hit the warm
    (~20ms) path.
    """
    try:
        store.collection()
        store.embed(["warmup"])  # force the sentence-transformer to load
    except Exception:
        # Not indexed yet -- /ask returns a clear error instead of crashing.
        pass


@app.get("/health")
def health():
    try:
        n = store.collection().count()
        return {"status": "ok", "indexed_chunks": n}
    except Exception:
        return {"status": "not_indexed", "indexed_chunks": 0}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        result = answer_mod.answer(req.question, k=req.k)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Librarian unavailable (is the brain indexed?): {e}",
        )
    return AskResponse(
        answer=result["answer"],
        sources=list(dict.fromkeys(result["sources"])),
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
    )


@app.post("/reindex")
def reindex():
    from librarian.ingest import build_chunks

    n = store.index(build_chunks())
    return {"status": "reindexed", "chunks": n}
