"""Hybrid retrieval: vector similarity fused with keyword overlap."""

import re
from dataclasses import dataclass

from . import config
from .store import collection, embed

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class Result:
    text: str
    source: str
    score: float


def retrieve(query: str, k: int | None = None, tracer=None) -> list[Result]:
    """Return the top-k chunks, fusing vector and keyword rankings (RRF).

    Vector search catches meaning ("how much is the sales agent" matches the
    pricing table). Keyword overlap catches exact terms the embedding may blur
    ("$1,000", "Gro Yoga"). Reciprocal-rank fusion combines both without
    needing the two scores to share a scale.
    """
    k = k or config.TOP_K
    col = collection()

    # Pull a wider candidate pool than k so fusion has room to work.
    pool = max(k * 3, 10)

    if tracer is not None:
        with tracer.span("embed_query"):
            q_vec = embed([query])
    else:
        q_vec = embed([query])

    if tracer is not None:
        with tracer.span("vector_search", pool=pool):
            res = col.query(query_embeddings=q_vec, n_results=pool)
    else:
        res = col.query(query_embeddings=q_vec, n_results=pool)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    by_id = {cid: (doc, meta) for cid, doc, meta in zip(ids, docs, metas)}

    # Ranking 1: vector order (as returned)
    vector_order = list(ids)
    # Ranking 2: keyword overlap order
    q_tokens = _tokens(query)
    keyword_scored = sorted(
        ids,
        key=lambda cid: len(q_tokens & _tokens(by_id[cid][0])),
        reverse=True,
    )

    rrf: dict[str, float] = {}
    K_CONST = 60  # standard RRF damping
    for ranking in (vector_order, keyword_scored):
        for rank, cid in enumerate(ranking):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K_CONST + rank)

    top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    results = [
        Result(text=by_id[cid][0], source=by_id[cid][1]["source"], score=score)
        for cid, score in top
    ]

    if tracer is not None:
        with tracer.span("rrf_fuse", candidates=len(ids), kept=len(results)):
            pass

    return results
