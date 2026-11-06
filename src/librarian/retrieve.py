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


def retrieve(query: str, k: int | None = None) -> list[Result]:
    """Return the top-k chunks, fusing vector + keyword rankings (RRF)."""
    k = k or config.TOP_K
    col = collection()

    pool = max(k * 3, 10)
    res = col.query(query_embeddings=embed([query]), n_results=pool)
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    by_id = {cid: (doc, meta) for cid, doc, meta in zip(ids, docs, metas)}

    vector_order = list(ids)
    q_tokens = _tokens(query)
    keyword_scored = sorted(
        ids,
        key=lambda cid: len(q_tokens & _tokens(by_id[cid][0])),
        reverse=True,
    )

    rrf: dict[str, float] = {}
    K_CONST = 60
    for ranking in (vector_order, keyword_scored):
        for rank, cid in enumerate(ranking):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (K_CONST + rank)

    top = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [
        Result(text=by_id[cid][0], source=by_id[cid][1]["source"], score=score)
        for cid, score in top
    ]
